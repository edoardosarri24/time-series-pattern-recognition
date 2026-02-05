import argparse
import os
import sys
import requests
import zipfile
import io
import random

# Configuration
OUTPUT_DIR = "../../data/"
OUTPUT_FILE = "input.txt"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
GROUPS = ['train', 'test']
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

# Functions
def get_args():
    parser = argparse.ArgumentParser(description="Download UCI HAR data for analysis.")
    parser.add_argument("--multiplier", type=int, default=1, help="Multiply dataset size X times by generating noisy variations (Data Augmentation).")
    parser.add_argument("--noise", type=float, default=0.01, help="Maximum amplitude of random noise added to augmented data (default: 0.01).")
    return parser.parse_args()

def load_signals(zip_ref, group):
    """
    Loads the 6 signal files for a given group (train or test).
    Returns a list of lists, where each inner list contains the lines of the file.
    """
    signals_data = []
    for signal in SIGNALS:
        # Construct path inside zip: e.g., UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt
        filename = f"UCI HAR Dataset/{group}/Inertial Signals/{signal}_{group}.txt"
        try:
            with zip_ref.open(filename) as f:
                # Read all lines, decode, strip whitespace
                lines = [line.decode('utf-8').strip() for line in f]
                signals_data.append(lines)
        except KeyError:
            print(f"Error: Could not find {filename} in zip archive.")
            sys.exit(1)
    return signals_data

def process_group(zip_ref, group, f_out, multiplier, noise_level):
    """
    Reads signals for a group, transposes them to (timestamp, dimension), and writes to f_out.
    Applies Data Augmentation if multiplier > 1.
    """
    # Load all 6 files -> shape [6, num_samples] where each sample is a string of 128 floats
    raw_signals = load_signals(zip_ref, group)
    
    # Check consistency
    num_samples = len(raw_signals[0])
    for i in range(1, 6):
        if len(raw_signals[i]) != num_samples:
            print(f"Error: Mismatch in sample counts for {group}. Signal 0 has {num_samples}, signal {i} has {len(raw_signals[i])}.")
            sys.exit(1)
            
    # Optimization: Pre-parse strings to floats once
    # parsed_signals structure: [dim_idx][sample_idx][time_idx]
    parsed_signals = []
    for dim_idx in range(6):
        dim_data = []
        for i in range(num_samples):
             # Split line by whitespace and convert to float
             values = list(map(float, raw_signals[dim_idx][i].split()))
             if len(values) != 128:
                 print(f"Warning: Sample {i} in dimension {dim_idx} has {len(values)} timestamps instead of 128.")
             dim_data.append(values)
        parsed_signals.append(dim_data)

    total_timestamps = 0
    
    # Generate data: Original (m=0) + Augmented variants (m > 0)
    for m in range(multiplier):
        # Iterate through each window/sample
        for i in range(num_samples):
            # We transpose: 128 timestamps, each with 6 values
            window_len = 128
            
            # Access pre-parsed data for this sample across all 6 dimensions
            # sample_dims[dim][t]
            sample_dims = [parsed_signals[d][i] for d in range(6)]
            
            for t in range(window_len):
                row_vals = []
                for d in range(6):
                    val = sample_dims[d][t]
                    # Apply small random noise for augmented copies
                    if m > 0:
                        val += random.uniform(-noise_level, noise_level)
                    row_vals.append(f"{val:.6f}")
                
                f_out.write(" ".join(row_vals) + "\n")
                total_timestamps += 1

# Main
def main():
    args = get_args()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    zip_path = os.path.join(OUTPUT_DIR, "UCI_HAR_Dataset.zip")

    # Download if not exists
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {DATASET_URL}.")
        try:
            with requests.get(DATASET_URL, stream=True) as r:
                r.raise_for_status()
                total_length = r.headers.get('content-length')

                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    if total_length is None: # no content length header
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            sys.stdout.write(f"\rDownloading {downloaded / (1024*1024):.2f} MB")
                            sys.stdout.flush()
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for chunk in r.iter_content(chunk_size=8192):
                            dl += len(chunk)
                            f.write(chunk)
                            done = int(50 * dl / total_length)
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl//1024} KB")
                            sys.stdout.flush()
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nCritical Error during download: {e}", file=sys.stderr)
            # Clean up partial file
            if os.path.exists(zip_path):
                os.remove(zip_path)
            sys.exit(1)
    else:
        print(f"Using cached dataset at {zip_path}")
    print(f"Extracting and processing... (Multiplier: {args.multiplier}x, Noise: +/-{args.noise})")
    try:
        with zipfile.ZipFile(zip_path) as z:
            with open(output_path, "w") as f_out:
                # Process train, then test
                for group in GROUPS:
                    process_group(z, group, f_out, args.multiplier, args.noise)
        print(f"Success! Dataset generated at: {output_path}")
        # Cleanup zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"Critical Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()