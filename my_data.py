#!/usr/bin/env python3

import os
import random
import argparse
import logging
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field

# --- Configuration ---
DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.95, 0.03, 0.02)  # Train, Val, Test
DEFAULT_OUTPUT_DIR = '.'
DEFAULT_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
OUTPUT_FILENAMES = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt'
}
# CHANGE: New delimiter for spec argument
SPEC_DELIMITER = '='

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Directory Specification ---
@dataclass
class DirectorySpec:
    """Specification for searching a directory."""
    path: str
    match_word: Optional[str] = None
    # Future requirements can be added here

# --- Core Functions ---

def find_audio_files(
    directory_specs: List[DirectorySpec],
    audio_extensions: Set[str]
) -> List[str]:
    """
    Recursively searches directories specified by DirectorySpec objects
    for audio files, applying optional match word filtering.

    Args:
        directory_specs: A list of DirectorySpec objects defining search paths
                         and criteria.
        audio_extensions: A set of lowercase audio file extensions to look for
                          (e.g., {'.wav', '.mp3'}).

    Returns:
        A list of absolute paths to the found audio files.
    """
    found_files = []
    processed_paths = set() # Avoid processing duplicates if specs overlap

    for spec in directory_specs:
        # Ensure path exists before making it absolute to avoid errors on invalid input
        if not os.path.exists(spec.path):
             logging.warning(f"Specified path does not exist: {spec.path}. Skipping.")
             continue
        # Make path absolute *after* existence check
        abs_dir_path = os.path.abspath(spec.path)
        if not os.path.isdir(abs_dir_path):
            logging.warning(f"Path is not a directory: {spec.path}. Skipping.")
            continue

        logging.info(
            f"Searching in: {abs_dir_path}"
            f"{f' (matching files containing *{spec.match_word}*)' if spec.match_word else ''}"
        )

        match_word_lower = spec.match_word.lower() if spec.match_word else None

        for root, _, files in os.walk(abs_dir_path, followlinks=True):
            for filename in files:
                # Use os.path.splitext for reliable extension splitting
                _ , ext = os.path.splitext(filename)
                ext_lower = ext.lower()

                # 1. Check extension
                if ext_lower not in audio_extensions:
                    continue

                # 2. Check match word (if specified) - check within the full filename
                if match_word_lower and match_word_lower not in filename.lower():
                    continue

                # 3. Construct absolute path and check for duplicates
                file_path = os.path.join(root, filename)
                abs_file_path = os.path.abspath(file_path)

                if abs_file_path not in processed_paths:
                    found_files.append(abs_file_path)
                    processed_paths.add(abs_file_path)
                # else: # Optional: log duplicate detection
                #    logging.debug(f"Skipping duplicate file path: {abs_file_path}")


    logging.info(f"Found {len(found_files)} unique audio files matching criteria.")
    return found_files


def split_files(
    file_list: List[str],
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SEED
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits a list of file paths into train, validation, and test sets.

    Args:
        file_list: List of file paths to split.
        split_ratios: Tuple of (train_ratio, val_ratio, test_ratio).
                      Must sum close to 1.0.
        seed: Random seed for shuffling.

    Returns:
        A tuple containing three lists: (train_files, val_files, test_files).
    """
    if not file_list:
        return [], [], []

    if not (0.999 < sum(split_ratios) < 1.001):
         raise ValueError(f"Split ratios must sum to 1.0. Got: {split_ratios} (sum={sum(split_ratios)})")

    if any(r < 0 for r in split_ratios):
        raise ValueError(f"Split ratios cannot be negative. Got: {split_ratios}")

    # Shuffle list reproducibly
    random.seed(seed)
    shuffled_list = file_list[:] # Create a copy to shuffle
    random.shuffle(shuffled_list)

    # Calculate split indices
    n_total = len(shuffled_list)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    # n_test is the remainder to ensure all files are used
    n_test = n_total - n_train - n_val

    # Handle potential rounding issues ensuring test set isn't negative
    if n_test < 0 :
         n_val = n_total - n_train # Adjust val size if test becomes negative
         n_test = 0
         logging.warning("Split calculation resulted in negative test samples."
                         f"Adjusting val size to {n_val}, test size to {n_test}.")
    elif n_train + n_val > n_total:
        n_val = n_total - n_train
        n_test = 0
        logging.warning("Split calculation resulted in train+val > total."
                         f"Adjusting val size to {n_val}, test size to {n_test}.")


    # Perform the split
    train_files = shuffled_list[:n_train]
    val_files = shuffled_list[n_train : n_train + n_val]
    test_files = shuffled_list[n_train + n_val :]

    logging.info(f"Splitting {n_total} files -> "
                 f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Sanity check
    assert len(train_files) + len(val_files) + len(test_files) == n_total, \
        "Internal error: Split counts do not match total file count."

    return train_files, val_files, test_files


def write_split_files(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    output_dir: str
) -> None:
    """
    Writes the file paths for each split into separate text files.

    Args:
        train_files: List of training file paths.
        val_files: List of validation file paths.
        test_files: List of test file paths.
        output_dir: Directory where the output .txt files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Writing split files to directory: {os.path.abspath(output_dir)}")

    split_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, file_list in split_data.items():
        output_filename = OUTPUT_FILENAMES[split_name]
        output_path = os.path.join(output_dir, output_filename)
        try:
            # Use utf-8 encoding for broader compatibility
            with open(output_path, 'w', encoding='utf-8') as f:
                for file_path in file_list:
                    f.write(f"{file_path}\n")
            logging.info(f"Successfully wrote {len(file_list)} paths to {output_path}")
        except IOError as e:
            logging.error(f"Failed to write to {output_path}: {e}")


# --- Argument Parsing and Main Execution ---

def parse_directory_spec(spec_string: str) -> DirectorySpec:
    """
    Parses a 'path[=match_word]' string into a DirectorySpec object.
    Uses '=' as the delimiter.
    """
    parts = spec_string.split(SPEC_DELIMITER, 1) # Split only on the first '='
    path = parts[0]
    match_word = parts[1] if len(parts) > 1 else None

    # Basic validation
    if not path:
        raise ValueError(f"Invalid directory specification: '{spec_string}'. Path cannot be empty.")
    # Optional: Check if path looks somewhat valid (e.g., doesn't start/end with delimiter if match word is present)
    # This check might be too strict depending on allowed path/match word characters
    # if match_word is not None and (spec_string.startswith(SPEC_DELIMITER) or spec_string.endswith(SPEC_DELIMITER)):
    #      raise ValueError(f"Invalid directory specification format: '{spec_string}'. Check delimiter usage.")

    return DirectorySpec(path=path.strip(), match_word=match_word.strip() if match_word else None)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Dataset utility to find audio files recursively, filter them optionally,
        and create train/validation/test splits (saving file paths to text files).
        """,
        formatter_class=argparse.RawTextHelpFormatter # Keep newline formatting in help
    )

    # CHANGE: Updated help text and metavar for --spec
    parser.add_argument(
        '--spec',
        action='append',  # Allows specifying multiple times
        required=True,
        metavar='PATH[=MATCH_WORD]',
        help=f"""Directory specification. Format: 'directory_path' or 'directory_path{SPEC_DELIMITER}match_word'.
Use this argument multiple times for multiple directories/specifications.
The match word is case-insensitive and checks for substring presence in the filename.
Example:
  --spec /data/sounds
  --spec C:\\Users\\Audio\\Clips{SPEC_DELIMITER}background
  --spec /archive/speech{SPEC_DELIMITER}clean_audio
        """
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the train.txt, val.txt, test.txt files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling and splitting (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        '-r', '--split-ratios',
        type=float,
        nargs=3,
        default=DEFAULT_SPLIT_RATIOS,
        metavar=('TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO'),
        help=f"Train, validation, test split ratios (must sum to 1.0) (default: {' '.join(map(str, DEFAULT_SPLIT_RATIOS))})"
    )
    parser.add_argument(
        '-e', '--extensions',
        type=str,
        nargs='+',
        default=list(DEFAULT_AUDIO_EXTENSIONS),
        help=f"List of audio file extensions to search for (case-insensitive). "
             f"Include the leading dot. (default: {' '.join(DEFAULT_AUDIO_EXTENSIONS)})"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose logging (DEBUG level)."
    )


    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Process Arguments ---
    try:
        directory_specs = [parse_directory_spec(spec) for spec in args.spec]
    except ValueError as e:
        parser.error(f"Error parsing --spec argument: {e}") # argparse will print usage and exit

    # Validate and normalize extensions
    audio_extensions = set()
    for ext in args.extensions:
        if not ext.startswith('.'):
            logging.warning(f"Extension '{ext}' does not start with a dot. Adding '.' automatically -> '.{ext.lower()}'")
            audio_extensions.add('.' + ext.lower())
        else:
            audio_extensions.add(ext.lower())

    logging.info(f"Using audio extensions: {audio_extensions}")


    # Validate split ratios
    try:
        # Check sum within the function called later, but check positive here
        if any(r < 0 for r in args.split_ratios):
            raise ValueError("Split ratios cannot be negative.")
        # Warn if sum is not exactly 1, but allow proceeding (split_files handles adjustment)
        if not (0.999 < sum(args.split_ratios) < 1.001):
             logging.warning(f"Split ratios {args.split_ratios} do not sum exactly to 1.0 (sum={sum(args.split_ratios)}). "
                             "The test set size will be adjusted to account for the remainder.")
    except ValueError as e:
        parser.error(str(e)) # argparse handles exit

    # --- Run Workflow ---
    all_audio_files = find_audio_files(directory_specs, audio_extensions)

    if not all_audio_files:
        logging.warning("No audio files found matching the criteria. No split files will be generated.")
        return # Exit gracefully

    try:
        train_files, val_files, test_files = split_files(
            all_audio_files,
            args.split_ratios,
            args.seed
        )
    except ValueError as e:
         # Handle errors from split_files (like ratio sum error if not caught before)
         logging.error(f"Error during file splitting: {e}")
         parser.exit(1) # Exit with error status


    write_split_files(train_files, val_files, test_files, args.output_dir)

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()