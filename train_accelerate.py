# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
# Modifications for Hugging Face Accelerate by user request.


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# No longer needed: from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
# No longer needed: import torch.multiprocessing as mp
# No longer needed: from torch.distributed import init_process_group
# No longer needed: from torch.nn.parallel import DistributedDataParallel
from accelerate import Accelerator # Added
from accelerate.utils import set_seed # Added

from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE

from bigvgan import BigVGAN
from discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)

from utils import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_audio,
)
import torchaudio as ta
from pesq import pesq
from tqdm import tqdm
import auraloss

# torch.backends.cudnn.benchmark = False # Can potentially be enabled depending on use case

# Removed rank argument
def train(a, h):
    # Initialize Accelerator
    # Add gradient_accumulation_steps to h or a (here assuming added to h)
    accelerator = Accelerator(
        gradient_accumulation_steps=h.gradient_accumulation_steps,
        mixed_precision=h.mixed_precision, # e.g., "fp16" or "bf16"
        log_with="tensorboard", # Integrate tensorboard logging
        project_dir=os.path.join(a.checkpoint_path, "logs") # Specify project dir for Accelerate logging state
    )

    # Set seed using Accelerate's utility which handles distributed settings
    set_seed(h.seed)

    # Device is handled by Accelerator
    device = accelerator.device

    # Define BigVGAN generator - placement handled by accelerator.prepare
    generator = BigVGAN(h)

    # Define discriminators - placement handled by accelerator.prepare
    mpd = MultiPeriodDiscriminator(h)

    # Define additional discriminators - placement handled by accelerator.prepare
    if h.get("use_mbd_instead_of_mrd", False):
        if accelerator.is_main_process:
            print(
                "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
            )
        mrd = MultiBandDiscriminator(h)
    elif h.get("use_cqtd_instead_of_mrd", False):
        if accelerator.is_main_process:
            print(
                "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
            )
        mrd = MultiScaleSubbandCQTDiscriminator(h)
    else:
        mrd = MultiResolutionDiscriminator(h)

    # Loss functions
    if h.get("use_multiscale_melloss", False):
        if accelerator.is_main_process:
            print(
                "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
            )
        # Move loss modules to device if they have parameters (this one doesn't, but good practice)
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        ).to(device)
    else:
        fn_mel_loss_singlescale = F.l1_loss

    # Print the model & number of parameters on main process
    if accelerator.is_main_process:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {a.checkpoint_path}")

    # Checkpoint scanning remains the same, but loading needs care
    cp_g, cp_do = None, None
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(
            a.checkpoint_path, prefix="g_", renamed_file="bigvgan_generator.pt"
        )
        cp_do = scan_checkpoint(
            a.checkpoint_path,
            prefix="do_",
            renamed_file="bigvgan_discriminator_optimizer.pt",
        )

    # Load model checkpoints *before* accelerator.prepare
    steps = 0
    last_epoch = -1
    state_dict_do = None # Keep track to load optimizers later

    if cp_g is not None:
        if accelerator.is_main_process:
             print(f"Loading generator models from checkpoint {cp_g}")
        state_dict_g = load_checkpoint(cp_g, device='cpu') # Load to CPU first
        generator.load_state_dict(state_dict_g["generator"])
    # Load discriminator models IF cp_do exists (since they are in the same file)
    if cp_do is not None:
        if accelerator.is_main_process:
             print(f"Loading discriminator models from checkpoint {cp_do}")
        # Load the state_dict_do once
        state_dict_do = load_checkpoint(cp_do, device='cpu') # Load to CPU
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        # Extract steps and epoch here
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
    else:
        if accelerator.is_main_process:
             print("No valid combined (do_) checkpoint found. Starting training from scratch or only generator weights.")
             # Reset steps/epoch if only generator was loaded
             if cp_g is not None:
                 print("WARNING: Loaded generator checkpoint but not optimizer/discriminator state. Resetting steps/epoch.")
             steps = 0
             last_epoch = -1

     # 2. Define Optimizers (AFTER potential model loading, BEFORE prepare)
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    # 3. Load Optimizer States (if they exist) - AFTER creating optimizers, BEFORE creating schedulers, BEFORE prepare
    if state_dict_do is not None:
        try:
            optim_g.load_state_dict(state_dict_do["optim_g"])
            optim_d.load_state_dict(state_dict_do["optim_d"])
            if accelerator.is_main_process:
                print("Loaded optimizer states.")
        except Exception as e:
             if accelerator.is_main_process:
                 print(f"Warning: Could not load optimizer state: {e}. Continuing with fresh optimizer state.")


    # 4. Define Schedulers (AFTER creating optimizers AND loading their states, BEFORE prepare)
    # Pass last_epoch which was loaded from state_dict_do
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    # 5. Load Scheduler States (Optional but recommended for exact resume) - AFTER creating schedulers, BEFORE prepare
    if state_dict_do is not None:
        try:
            if "scheduler_g" in state_dict_do:
                 scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
                 if accelerator.is_main_process: print("Loaded scheduler_g state.")
            if "scheduler_d" in state_dict_do:
                 scheduler_d.load_state_dict(state_dict_do["scheduler_d"])
                 if accelerator.is_main_process: print("Loaded scheduler_d state.")
        except Exception as e:
             if accelerator.is_main_process:
                 print(f"Warning: Could not load scheduler state: {e}. Continuing potentially from incorrect step count if only last_epoch was used.")

    # Define training and validation datasets
    training_filelist, validation_filelist, list_unseen_validation_filelist = (
        get_dataset_filelist(a)
    )

    # No explicit shuffle needed, DataLoader with sampler handles it
    # No explicit device needed for dataset if data loading is CPU-bound
    trainset = MelDataset(
        training_filelist,
        h,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        shuffle=True, # Shuffle should be True for training
        fmax_loss=h.fmax_for_loss,
        # device=device, # Keep data loading on CPU
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
        is_seen=True,
    )

    # Removed DistributedSampler creation

    # Create DataLoaders (validation loaders only needed on main process for eval)
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True, # Shuffle=True is usually desired, Accelerate handles distributed sampling
        # sampler=train_sampler, # Removed sampler
        batch_size=h.batch_size, # This is per-process batch size
        pin_memory=True,
        drop_last=True,
    )

    validation_loader = None
    list_unseen_validation_loader = []
    if accelerator.is_main_process:
        validset = MelDataset(
            validation_filelist, h, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate,
            h.fmin, h.fmax, False, False, fmax_loss=h.fmax_for_loss, fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir, is_seen=True,
        )
        validation_loader = DataLoader(
            validset, num_workers=1, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True
        )

        for i in range(len(list_unseen_validation_filelist)):
            unseen_validset = MelDataset(
                list_unseen_validation_filelist[i], h, h.segment_size, h.n_fft, h.num_mels, h.hop_size,
                h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, fmax_loss=h.fmax_for_loss,
                fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=False,
            )
            unseen_validation_loader = DataLoader(
                unseen_validset, num_workers=1, shuffle=False, sampler=None, batch_size=1, pin_memory=True, drop_last=True
            )
            list_unseen_validation_loader.append(unseen_validation_loader)


    # Prepare components with Accelerate
    # Note: We prepare validation loaders too, although only used by main process.
    # This is generally good practice if validation might be distributed later.
    # 6. Prepare components with Accelerate
    if accelerator.is_main_process: print("Preparing components with Accelerate...")
    (
        generator, mpd, mrd, optim_g, optim_d, scheduler_g, scheduler_d, train_loader,
        validation_loader, *list_unseen_validation_loader
    ) = accelerator.prepare(
        generator, mpd, mrd, optim_g, optim_d, scheduler_g, scheduler_d, train_loader,
        validation_loader, *list_unseen_validation_loader
    )
    if accelerator.is_main_process: print("Preparation complete.")

    # Tensorboard logger initialization
    sw = None
    if accelerator.is_main_process:
        # Accelerator integrates TensorBoard, initialize tracker
        #import pdb; pdb.set_trace()
        config_dict = dict(h)
        accelerator.init_trackers(
             project_name="bigvgan_training", # Adjust as needed
             #config=h # Log hyperparameters
        )
        # Get the underlying SummaryWriter if needed for direct calls (e.g., add_figure, add_audio)
        # Note: This might vary slightly based on Accelerate version. Check documentation.
        # Usually, you'd use accelerator.log()
        try:
            sw = accelerator.get_tracker("tensorboard").writer
        except Exception as e:
            print(f"Could not get Tensorboard writer from Accelerate tracker: {e}")
            # Fallback to manual SummaryWriter if needed, ensure log dir is correct
            log_dir = os.path.join(a.checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            sw = SummaryWriter(log_dir=log_dir)

        if a.save_audio: # Also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, "samples"), exist_ok=True)

    # Validation loop (only runs on main process)
    def validate(accelerator, a, h, loader, mode="seen", current_step=-1): # pass accelerator and step
        assert current_step >= 0

        # Ensure validation runs only on the main process
        if not accelerator.is_main_process:
             return

        # Get the raw model underlying DDP/FSDP/etc.
        unwrapped_generator = accelerator.unwrap_model(generator)
        unwrapped_generator.eval()
        torch.cuda.empty_cache() # Keep cache clearing

        val_err_tot = 0
        val_pesq_tot = 0
        val_mrstft_tot = 0

        # Modules for evaluation metrics - move to main process device
        pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).to(accelerator.device)
        loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=accelerator.device)

        if a.save_audio:
            # Create directories for the specific step
            gt_dir = os.path.join(a.checkpoint_path, "samples", f"gt_{mode}")
            gen_dir = os.path.join(a.checkpoint_path, "samples", f"{mode}_{current_step:08d}")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(gen_dir, exist_ok=True)

        with torch.no_grad():
            print(f"Step {current_step} {mode} speaker validation...")
            # Use tqdm only on the main process
            for j, batch in enumerate(tqdm(loader, desc=f"Validation {mode}")):
                # Data should already be on the correct device from DataLoader if prepared,
                # but explicit .to() is safer if loader wasn't prepared or for manual data handling.
                x, y, _, y_mel = batch
                x = x.to(accelerator.device)
                y = y.to(accelerator.device)
                y_mel = y_mel.to(accelerator.device)

                # Use the unwrapped generator model
                y_g_hat = unwrapped_generator(x)

                # Calculations remain largely the same, ensure tensors are on accelerator.device
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                    h.win_size, h.fmin, h.fmax_for_loss,
                )
                min_t = min(y_mel.size(-1), y_g_hat_mel.size(-1))
                val_err_tot += F.l1_loss(y_mel[...,:min_t], y_g_hat_mel[...,:min_t]).item()

                if not "nonspeech" in mode:
                    y_16k = pesq_resampler(y)
                    y_g_hat_16k = pesq_resampler(y_g_hat.squeeze(1))
                    try:
                        y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                        y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                        val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
                    except Exception as e:
                         print(f"\n PESQ calculation failed for item {j}: {e}")


                min_t_wav = min(y.size(-1), y_g_hat.size(-1))
                val_mrstft_tot += loss_mrstft(y_g_hat[...,:min_t_wav], y[...,:min_t_wav]).item()

                # Log audio and figures using accelerator.log or sw directly
                if j % a.eval_subsample == 0 and sw is not None:
                    if current_step >= 0:
                        try:
                            sw.add_audio(f"gt_{mode}/y_{j}", y[0].cpu(), current_step, h.sampling_rate)
                            sw.add_figure(f"gt_{mode}/y_spec_{j}", plot_spectrogram(x[0].cpu()), current_step) # Ensure tensor is on CPU for plotting
                            if a.save_audio:
                                save_audio(y[0].cpu(), os.path.join(gt_dir, f"{j:04d}_step{current_step}.wav"), h.sampling_rate)
                        except Exception as log_e:
                            print(f"Error logging GT sample {j}: {log_e}")


                    try:
                        sw.add_audio(f"generated_{mode}/y_hat_{j}", y_g_hat[0].cpu(), current_step, h.sampling_rate)
                        y_hat_spec = mel_spectrogram(
                            y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                            h.win_size, h.fmin, h.fmax,
                        )
                        sw.add_figure(f"generated_{mode}/y_hat_spec_{j}", plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), current_step)

                        spec_delta = torch.clamp(torch.abs(x[0].cpu() - y_hat_spec.squeeze(0).cpu()), min=1e-6, max=1.0)
                        sw.add_figure(f"delta_dclip1_{mode}/spec_{j}", plot_spectrogram_clipped(spec_delta.numpy(), clip_max=1.0), current_step)

                        if a.save_audio:
                            save_audio(y_g_hat[0, 0].cpu(), os.path.join(gen_dir, f"{j:04d}.wav"), h.sampling_rate)
                    except Exception as log_e:
                        print(f"Error logging generated sample {j}: {log_e}")


            val_err = val_err_tot / (j + 1)
            val_pesq = val_pesq_tot / (j + 1) if not "nonspeech" in mode else 0 # Avoid division by zero if skipped
            val_mrstft = val_mrstft_tot / (j + 1)

            # Log evaluation metrics using accelerator.log or sw directly
            if sw is not None:
                sw.add_scalar(f"validation_{mode}/mel_spec_error", val_err, current_step)
                sw.add_scalar(f"validation_{mode}/pesq", val_pesq, current_step)
                sw.add_scalar(f"validation_{mode}/mrstft", val_mrstft, current_step)
            # Or use accelerator.log:
            # accelerator.log({
            #     f"validation_{mode}/mel_spec_error": val_err,
            #     f"validation_{mode}/pesq": val_pesq,
            #     f"validation_{mode}/mrstft": val_mrstft
            # }, step=current_step)

        # Set model back to train mode
        unwrapped_generator.train()


    # Initial validation if resuming
    if steps != 0 and accelerator.is_main_process and not a.debug:
        if not a.skip_seen and validation_loader:
            validate(accelerator, a, h, validation_loader, mode=f"seen_{train_loader.dataset.name}", current_step=steps)
        for i in range(len(list_unseen_validation_loader)):
            validate(accelerator, a, h, list_unseen_validation_loader[i], mode=f"unseen_{list_unseen_validation_loader[i].dataset.name}", current_step=steps)

    if a.evaluate:
         if accelerator.is_main_process:
              print("Evaluation finished.")
         accelerator.wait_for_everyone() # Ensure all processes finish before exiting
         # Clean up trackers
         if accelerator.is_main_process:
              accelerator.end_training()
         exit()

    # Main training loop
    generator.train()
    mpd.train()
    mrd.train()

    # Use total_batch_size for reference if needed, but batch size in loader is per-process
    total_batch_size = h.batch_size * accelerator.num_processes * h.gradient_accumulation_steps
    if accelerator.is_main_process:
        print(f"Total effective batch size: {total_batch_size}")
        print(f"Gradient accumulation steps: {h.gradient_accumulation_steps}")
        print(f"Mixed precision: {h.mixed_precision}")

    global_step = steps # Use a separate counter for optimizer steps


    for epoch in range(max(0, last_epoch), a.training_epochs):
        if accelerator.is_main_process:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        # No need for train_sampler.set_epoch(epoch), Accelerate handles DataLoader shuffling

        # Inner loop iterates `gradient_accumulation_steps` times per optimizer step
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            is_sync_step = ((step + 1) % h.gradient_accumulation_steps == 0) or (step + 1 == len(train_loader))
            
            # Context manager for gradient accumulation synchronization
            # No autocast needed, handled by accelerator.prepare and accelerator.backward
            with accelerator.accumulate(generator), accelerator.accumulate(mpd), accelerator.accumulate(mrd):
                # Data to device handled by `accelerator.prepare(train_loader)`
                # Or manually: x = batch[0].to(device) ...
                x, y, _, y_mel = batch
                # y = y.to(device, non_blocking=True) # Already on device
                # y_mel = y_mel.to(device, non_blocking=True) # Already on device
                y = y.unsqueeze(1)

                y_g_hat = generator(x)
                # y_g_hat is potentially in lower precision (fp16/bf16) due to `prepare`

                # Discriminator Optimization
                # Need to run D forwards in full precision potentially, or ensure loss handles mixed types
                # Usually, D input requires detach() and potentially casting if needed.
                # Autocast context is implicitly handled by accelerator.

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MRD
                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f
                # Scale loss for accumulation
                loss_disc_all_scaled = loss_disc_all / h.gradient_accumulation_steps

                # Backward pass for discriminator
                optim_d.zero_grad(set_to_none=True) # Zero grads before backward
                if global_step >= a.freeze_step: # Check global_step for freezing
                    accelerator.backward(loss_disc_all_scaled) # retain_graph=True?
                    # Clipping and stepping happen only on sync steps
                else:
                     # If D is frozen, don't compute gradients or step
                     pass # No backward pass needed


                # Generator Optimization
                # L1 Mel-Spectrogram Loss
                lambda_melloss = h.get("lambda_melloss", 45.0)
                # Cast Mel Spectrogram inputs explicitly if needed, although L1 usually handles mixed prec ok
                # y_g_hat might be float16, calculate mel spec from it
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1).float(), # Ensure input to mel_spectrogram is float32 if needed
                    h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax_for_loss,
                )

                if h.get("use_multiscale_melloss", False):
                    # fn_mel_loss_multiscale expects waveforms
                    loss_mel = fn_mel_loss_multiscale(y.float(), y_g_hat.float()) * lambda_melloss # Ensure float32 input if loss func requires
                else:
                    # Ensure y_mel and y_g_hat_mel are compatible (e.g., both float32)
                    loss_mel = fn_mel_loss_singlescale(y_mel.float(), y_g_hat_mel.float()) * lambda_melloss


                # MPD loss
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat) # y_g_hat is not detached here
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

                # MRD loss
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat) # y_g_hat is not detached here
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

                if global_step >= a.freeze_step:
                    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                else:
                    # print("[WARNING] using regression loss only for G...") # Avoid printing every step
                    loss_gen_all = loss_mel

                # Scale loss for accumulation
                loss_gen_all_scaled = loss_gen_all / h.gradient_accumulation_steps

                # Backward pass for generator
                optim_g.zero_grad(set_to_none=True) # Zero grads before backward
                accelerator.backward(loss_gen_all_scaled)

                # Optimizer step and scheduler step only if accumulation cycle is complete
                if is_sync_step:
                    # Clip gradients before stepping optimizer
                    clip_grad_norm = h.get("clip_grad_norm", 1000.0) # Default to 1000

                    grad_norm_g = 0.0
                    grad_norm_mpd = 0.0
                    grad_norm_mrd = 0.0

                    if accelerator.sync_gradients: # Redundant check inside is_sync_step but explicit
                         # Clip G gradients
                        if len(list(generator.parameters())) > 0:
                            grad_norm_g = accelerator.clip_grad_norm_(generator.parameters(), clip_grad_norm)
                        optim_g.step() # Step G right after clipping G
                        scheduler_g.step() # Step scheduler G

                        # Process Discriminator (if not frozen)
                        if global_step >= a.freeze_step:
                            params_to_clip_d = itertools.chain(mpd.parameters(), mrd.parameters())
                            if any(p.grad is not None for p in itertools.chain(mpd.parameters(), mrd.parameters())):
                                  grad_norm_d = accelerator.clip_grad_norm_(params_to_clip_d, clip_grad_norm)
                            else:
                                  grad_norm_d = 0.0
                            optim_d.step() # Step D right after clipping D
                            scheduler_d.step() # Step scheduler D

                    # Increment the actual step counter
                    global_step += 1

                    # Logging, Checkpointing, Validation (on main process, tied to global_step)
                    if accelerator.is_main_process:
                        # STDOUT logging
                        if global_step % a.stdout_interval == 0:
                            mel_error_log = loss_mel.item() / lambda_melloss if lambda_melloss > 0 else loss_mel.item()
                            print(
                                f"Steps: {global_step:d} "
                                f"Epoch: {epoch + 1} ({step+1}/{len(train_loader)}) "
                                f"Gen Loss Total: {loss_gen_all.item():4.3f} " # Log unscaled loss
                                f"Disc Loss Total: {loss_disc_all.item():4.3f} " # Log unscaled loss
                                f"Mel Error: {mel_error_log:4.3f} "
                                f"s/step: {(time.time() - start_b) / h.gradient_accumulation_steps :4.3f} " # Approximate time per optimizer step
                                f"LR: {optim_g.param_groups[0]['lr']:.2e} "
                                f"G_grad: {grad_norm_g:.2f} D_grad: {grad_norm_mpd + grad_norm_mrd:.2f}" # Log norms after clip
                            )

                        # Tensorboard summary logging (use accelerator.log or sw directly)
                        if global_step % a.summary_interval == 0 and sw is not None:
                             mel_error_log = loss_mel.item() / lambda_melloss if lambda_melloss > 0 else loss_mel.item()
                             log_data = {
                                  "training/gen_loss_total": loss_gen_all.item(), # Log unscaled loss
                                  "training/mel_spec_error": mel_error_log,
                                  "training/fm_loss_mpd": loss_fm_f.item(),
                                  "training/gen_loss_mpd": loss_gen_f.item(),
                                  "training/disc_loss_mpd": loss_disc_f.item(),
                                  "training/grad_norm_mpd": grad_norm_mpd, # Log norm after clip
                                  "training/fm_loss_mrd": loss_fm_s.item(),
                                  "training/gen_loss_mrd": loss_gen_s.item(),
                                  "training/disc_loss_mrd": loss_disc_s.item(),
                                  "training/grad_norm_mrd": grad_norm_mrd, # Log norm after clip
                                  "training/grad_norm_g": grad_norm_g,     # Log norm after clip
                                  "training/learning_rate_d": scheduler_d.get_last_lr()[0],
                                  "training/learning_rate_g": scheduler_g.get_last_lr()[0],
                                  "training/epoch": epoch + 1,
                             }
                             # accelerator.log(log_data, step=global_step) # Use accelerator's logging
                             for k, v in log_data.items(): sw.add_scalar(k, v, global_step) # Or direct SW calls


                        # Checkpointing (on main process)
                        if global_step % a.checkpoint_interval == 0 and global_step != 0:
                            accelerator.wait_for_everyone() # Ensure all processes are ready before saving

                            if accelerator.is_main_process:
                                # Unwrap models before saving state dict
                                unwrapped_generator = accelerator.unwrap_model(generator)
                                unwrapped_mpd = accelerator.unwrap_model(mpd)
                                unwrapped_mrd = accelerator.unwrap_model(mrd)

                                checkpoint_path_g = f"{a.checkpoint_path}/g_{global_step:08d}"
                                save_checkpoint(
                                    checkpoint_path_g, {"generator": unwrapped_generator.state_dict()}
                                )

                                checkpoint_path_do = f"{a.checkpoint_path}/do_{global_step:08d}"
                                save_checkpoint(
                                    checkpoint_path_do,
                                    {
                                        "mpd": unwrapped_mpd.state_dict(),
                                        "mrd": unwrapped_mrd.state_dict(),
                                        "optim_g": optim_g.state_dict(),
                                        "optim_d": optim_d.state_dict(),
                                        # Optional: save scheduler states if needed for exact resume
                                        # "scheduler_g": scheduler_g.state_dict(),
                                        # "scheduler_d": scheduler_d.state_dict(),
                                        "steps": global_step, # Save the global step count
                                        "epoch": epoch,
                                        "config": h, # Save config for reference
                                        "build_version": "accelerate_adapted"
                                    },
                                )
                                print(f"Checkpoints saved at step {global_step}")


                        # Validation (on main process)
                        if global_step % a.validation_interval == 0 and global_step != 0:
                             if not a.debug:
                                  accelerator.wait_for_everyone() # Ensure training is paused everywhere
                                  validate(accelerator, a, h, validation_loader, mode=f"seen_{train_loader.dataset.name}", current_step=global_step)
                                  for i in range(len(list_unseen_validation_loader)):
                                       validate(accelerator, a, h, list_unseen_validation_loader[i], mode=f"unseen_{list_unseen_validation_loader[i].dataset.name}", current_step=global_step)
                                  # Ensure model is back in train mode after validation
                                  generator.train()
                                  mpd.train()
                                  mrd.train()

            # Reset timer for next step measurement
            if accelerator.is_main_process and is_sync_step:
                 start_b = time.time()


        if accelerator.is_main_process:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")

    # End training
    if accelerator.is_main_process:
        print("Finished training.")
        if sw is not None:
            accelerator.end_training() # Clean up trackers


def main():
    print("Initializing Training Process with Accelerate..")

    parser = argparse.ArgumentParser()

    # Keep original args
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--input_wavs_dir", default="LibriTTS")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LibriTTS/train.txt")
    parser.add_argument("--input_validation_file", default="LibriTTS/val.txt")
    parser.add_argument("--list_input_unseen_wavs_dir", nargs="+", default=[])
    parser.add_argument("--list_input_unseen_validation_file", nargs="+", default=[])
    parser.add_argument("--checkpoint_path", default="exp/bigvgan_accelerate")
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=10000, type=int)
    parser.add_argument("--stdout_interval", default=100, type=int) # Adjust defaults maybe
    parser.add_argument("--checkpoint_interval", default=10000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=5000, type=int)
    parser.add_argument("--freeze_step", default=0, type=int, help="freeze D for the first specified steps.")
    parser.add_argument("--fine_tuning", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool, help="debug mode. skips validation loop.")
    parser.add_argument("--evaluate", default=False, type=bool, help="only run evaluation from checkpoint.")
    parser.add_argument("--eval_subsample", default=5, type=int, help="subsampling during evaluation.")
    parser.add_argument("--skip_seen", default=False, type=bool, help="skip seen dataset validation.")
    parser.add_argument("--save_audio", default=False, type=bool, help="save audio during validation.")

    # Add Accelerate specific args
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--mixed_precision", default="no", type=str, choices=["no", "fp16", "bf16"], help="Mixed precision mode (no, fp16, bf16).")

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Add accelerate args to hparams dict
    h.gradient_accumulation_steps = a.gradient_accumulation_steps
    h.mixed_precision = a.mixed_precision

    # Let Accelerate handle num_gpus and batch size per GPU
    # No need to manually divide batch size here

    build_env(a.config, "config.json", a.checkpoint_path)

    # Remove manual seeding here, Accelerate handles it in train()
    # No need for mp.spawn

    # Call train directly, Accelerate handles distributed launch
    train(a, h)


if __name__ == "__main__":
    main()