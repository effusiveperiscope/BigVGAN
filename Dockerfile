
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

Run apt-get update && apt-get install -y \
    software-properties-common
Run add-apt-repository universe
Run apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libjpeg-dev \
    libpng-dev \
    gcc

Run mkdir /root/BigVGAN
Workdir /root/BigVGAN

Copy requirements.txt /root/BigVGAN/

Run pip3 install --upgrade pip
Run pip3 install numpy==1.23.5 librosa==0.10.1 \
    scipy==1.11.3 tensorboard==2.15.1 soundfile==0.12.1 matplotlib==3.8.1 \
    pesq==0.0.4 auraloss==0.4.0 tqdm==4.66.1 accelerate==0.25.0

Run ldconfig && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/*

Copy *py /root/BigVGAN/
Copy *ipynb /root/BigVGAN/
Copy *yaml /root/BigVGAN/
Copy alias_free_torch /root/BigVGAN/alias_free_torch