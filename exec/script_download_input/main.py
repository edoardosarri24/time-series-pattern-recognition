import argparse
import os
import sys
import requests
import zipfile
import io

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

def process_group(zip_ref, group, f_out):
    """
    Reads signals for a group, transposes them to (timestamp, dimension), and writes to f_out.
    """
    # Load all 6 files -> shape [6, num_samples] where each sample is a string of 128 floats
    raw_signals = load_signals(zip_ref, group)
    # Check consistency
    num_samples = len(raw_signals[0])
    for i in range(1, 6):
        if len(raw_signals[i]) != num_samples:
            print(f"Error: Mismatch in sample counts for {group}. Signal 0 has {num_samples}, signal {i} has {len(raw_signals[i])}.")
            sys.exit(1)
    total_timestamps = 0
    # Iterate through each window/sample
    for i in range(num_samples):
        # Parse the 128 values for each of the 6 dimensions
        # dims will be a list of 6 lists, each containing 128 floats
        dims = []
        for dim_idx in range(6):
            # Split line by whitespace (handles multiple spaces)
            values = list(map(float, raw_signals[dim_idx][i].split()))
            if len(values) != 128:
                print(f"Warning: Sample {i} in dimension {dim_idx} has {len(values)} timestamps instead of 128.")
            dims.append(values)
        # Now write them transposed: 128 timestamps, each with 6 values
        # We assume all dimensions have same length (128)
        window_len = len(dims[0])
        for t in range(window_len):
            row_vals = [str(dims[d][t]) for d in range(6)]
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
    print("Extracting and processing...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            with open(output_path, "w") as f_out:
                # Process train, then test
                for group in GROUPS:
                    process_group(z, group, f_out)
        print(f"Success! Dataset generated at: {output_path}")
        # Cleanup zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"Critical Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()