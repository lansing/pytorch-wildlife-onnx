import struct
from pathlib import Path
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)

# --- 1. Define command-line arguments ---
parser = argparse.ArgumentParser(description="Remove Ultralytics metadata header from a TensorRT engine file.")
parser.add_argument('--ultralytics_engine_path_file', type=str, default='exported_models/ultralytics_engine_path.txt',
                    help='Path to the file containing the path of the Ultralytics-exported engine.')
parser.add_argument('--output_dir', type=str, default='exported_models', help='Directory to save the pure engine file.')

args = parser.parse_args()

# --- Setup Output Directory ---
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

try:
    # Read the path to the Ultralytics-exported engine
    with open(args.ultralytics_engine_path_file, 'r') as f:
        ultralytics_engine_path = Path(f.read().strip())

    if not ultralytics_engine_path.exists():
        LOGGER.error(f"Ultralytics engine file not found: {ultralytics_engine_path}")
        exit(1)

    LOGGER.info(f"Processing Ultralytics engine: {ultralytics_engine_path}")

    # Determine the output path for the pure engine
    pure_engine_name = f"{ultralytics_engine_path.stem}.pure{ultralytics_engine_path.suffix}"
    pure_engine_path = output_dir / pure_engine_name

    with open(ultralytics_engine_path, 'rb') as f_in:
        # Read the first 4 bytes to get the metadata length
        metadata_len_bytes = f_in.read(4)
        if len(metadata_len_bytes) < 4:
            LOGGER.error("File is too small to contain metadata length. It might already be a pure engine or corrupted.")
            # Assume it's already pure and copy it over
            with open(pure_engine_path, 'wb') as f_out:
                f_in.seek(0)
                f_out.write(f_in.read())
            LOGGER.info(f"Copied original file as pure engine: {pure_engine_path}")
            exit(0)

        # Interpret as a signed little-endian integer
        metadata_length = struct.unpack('<i', metadata_len_bytes)[0]

        if metadata_length < 0 or metadata_length > f_in.seek(0, os.SEEK_END) - 4:
            LOGGER.error(f"Invalid metadata length ({metadata_length}). File might be corrupted or not an Ultralytics engine with metadata. Copying as is.")
            with open(pure_engine_path, 'wb') as f_out:
                f_in.seek(0)
                f_out.write(f_in.read())
            LOGGER.info(f"Copied original file as pure engine due to invalid metadata length: {pure_engine_path}")
            exit(0)

        f_in.seek(4 + metadata_length) # Move past the 4 bytes length + metadata itself

        # Read the rest of the file, which is the pure TensorRT engine
        pure_engine_data = f_in.read()

    # Write the pure engine data to a new file
    with open(pure_engine_path, 'wb') as f_out:
        f_out.write(pure_engine_data)

    LOGGER.info(f"Successfully removed Ultralytics header. Pure TensorRT engine saved to: {pure_engine_path}")

except FileNotFoundError:
    LOGGER.error(f"The file '{args.ultralytics_engine_path_file}' was not found.")
    exit(1)
except Exception as e:
    LOGGER.error(f"An error occurred: {e}")
    exit(1)

