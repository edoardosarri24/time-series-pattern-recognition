import argparse
import os
import sys
import requests
import zipfile
import random

# Configuration
OUTPUT_DIR = "../../data/"
OUTPUT_FILE = "input.txt"
QUERY_FILE = "query.txt"
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
    parser.add_argument("--query-length", type=int, default=64, help="Length of the query pattern (default: 64).")
    parser.add_argument("--seed", type=int, default=78, help="Random seed for query selection (default: 78).")
    parser.add_argument("--sigma", type=float, default=0.01, help="Standard deviation for query noise (default: 0.01).")
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

# Main
def main():
    args = get_args()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    query_path = os.path.join(OUTPUT_DIR, QUERY_FILE)
    zip_path = os.path.join(OUTPUT_DIR, "UCI_HAR_Dataset.zip")

    # Download if not exists
    if not os.path.exists(zip_path):
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
        except Exception as e:
            print(f"\nCritical Error during download: {e}", file=sys.stderr)
            # Clean up partial file
            if os.path.exists(zip_path):
                os.remove(zip_path)
            sys.exit(1)
        print() # Newline after progress bar
    else:
        print(f"Using cached dataset at {zip_path}")

    # Data Augumentation
    print(f"Extracting and processing... (Multiplier: {args.multiplier}x, Noise: +/-{args.noise})")
    try:
        with zipfile.ZipFile(zip_path) as z:
            # 1. Load all raw data into memory
            raw_groups = []
            total_base_timestamps = 0
            for group in GROUPS:
                raw = load_signals(z, group)
                # Parse to floats to avoid repeated parsing
                num_samples = len(raw[0])
                # Check consistency
                for i in range(1, 6):
                    if len(raw[i]) != num_samples:
                        print(f"Error: Mismatch in sample counts for {group}.")
                        sys.exit(1)
                parsed = []
                for dim_idx in range(6):
                    dim_data = []
                    for i in range(num_samples):
                        values = list(map(float, raw[dim_idx][i].split()))
                        if len(values) != 128:
                            print(f"Warning: Sample {i} in dimension {dim_idx} has {len(values)} timestamps.")
                        dim_data.append(values)
                    parsed.append(dim_data)
                group_timestamps = num_samples * 128
                total_base_timestamps += group_timestamps
                raw_groups.append((parsed, num_samples))
            total_timestamps = total_base_timestamps * args.multiplier
            # 2. Pick Query Index
            random.seed(args.seed)
            window_len = 128
            num_windows = total_timestamps // window_len
            max_window_idx = (total_timestamps - args.query_length) // window_len
            if max_window_idx < 0:
                raise ValueError("Dataset smaller than query length")
            selected_window_idx = random.randint(0, max_window_idx)
            query_start_idx = selected_window_idx * window_len
            print(f"Selected Query Start Index: {query_start_idx} (Window {selected_window_idx})")
            
            # 3. Generate Output and Capture Query
            query_data = []
            global_idx = 0
            with open(output_path, "w") as f_out:
                for m in range(args.multiplier):
                    for (parsed_signals, num_samples) in raw_groups:
                        for i in range(num_samples):
                            # Transpose: [dim][i] -> timestamps
                            sample_dims = [parsed_signals[d][i] for d in range(6)]
                            for t in range(128):
                                row_vals = []
                                row_floats = [] # for query capture
                                for d in range(6):
                                    val = sample_dims[d][t]
                                    if m > 0:
                                        val += random.uniform(-args.noise, args.noise)
                                    row_vals.append(f"{val:.6f}")
                                    row_floats.append(val)
                                f_out.write(" ".join(row_vals) + "\n")
                                # Capture query
                                if query_start_idx <= global_idx < query_start_idx + args.query_length:
                                    query_data.append(row_floats)
                                global_idx += 1
                    # Progress indication
                    sys.stdout.write(f"\rGenerated copy {m+1}/{args.multiplier}")
                    sys.stdout.flush()
            print("\nDataset generation complete.")

            # 4. Write Query File
            with open(query_path, "w") as f_q:
                for row in query_data:
                    # Add Gaussian Noise to query as per spec
                    noisy_row = []
                    for val in row:
                        val += random.gauss(0, args.sigma)
                        noisy_row.append(f"{val:.6f}")
                    f_q.write(" ".join(noisy_row) + "\n")
            print(f"Dataset saved to: {output_path}")
            print(f"Query saved to: {query_path}")
            # Cleanup zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)

    except Exception as e:
        print(f"\nCritical Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
