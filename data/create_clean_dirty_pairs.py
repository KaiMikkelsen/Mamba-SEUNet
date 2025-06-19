import os
import numpy as np
import librosa
import soundfile as sf
import random

# --- Configuration ---
# Set your specific input directories
CLEAN_GUITAR_INPUT_DIR = "/Users/kaimikkelsen/guitar_noise/guitar noise datasets/noisy_data/audio_mono-pickup_mix"
HUM_INPUT_DIR = "/Users/kaimikkelsen/guitar_noise/guitar noise datasets/noisy_data/recorded_hums"

# Output directories for generated audio
NOISY_OUTPUT_DIR = "generated_datasets_identical_names/training_noisy_inputs" # Output for mixed guitar + hum
CLEAN_TARGET_OUTPUT_DIR = "generated_datasets_identical_names/training_clean_targets" # Output for original clean guitar

# IMPORTANT: Verify Mamba-SEUNet's expected sample rate!
# Mamba-SEUNet was designed for speech, often 16kHz. Guitar is typically 44.1kHz or 48kHz.
# You MUST ensure your generated audio matches the model's expectation.
# If the model expects 16kHz, change this to 16000. Librosa will handle resampling during load.
# Current setting assumes you want to keep the original sample rate of your guitar recordings
# or resample everything to 44.1kHz if sources vary.
SAMPLE_RATE = 16000

NUM_AUGMENTATIONS_PER_CLEAN_FILE = 10 # Generate multiple variations (different hum/SNR) for each clean guitar file
MIN_SNR_DB = -5  # Minimum Signal-to-Noise Ratio (guitar to hum)
MAX_SNR_DB = 10  # Maximum Signal-to-Noise Ratio (guitar to hum)

# Create output directories if they don't exist
os.makedirs(NOISY_OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEAN_TARGET_OUTPUT_DIR, exist_ok=True)

def load_audio(file_path, sr, mono=True):
    """
    Load audio with librosa, ensuring consistent sample rate and mono.
    Handles various audio formats.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=mono)
        if audio.ndim > 1:
            audio = audio.flatten() # Ensure mono if loaded as stereo
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def adjust_length(audio, target_length):
    """
    Pad or trim audio to target_length. If padding, it loops the audio.
    """
    if audio is None:
        return None

    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        # Loop the audio to reach target_length
        num_repeats = (target_length + len(audio) - 1) // len(audio)
        looped_audio = np.tile(audio, num_repeats)
        return looped_audio[:target_length]
    return audio

def mix_signals(clean_signal, noise_signal, snr_db):
    """
    Mix clean signal with noise at a given SNR.
    Normalizes the mixed signal to prevent clipping.
    """
    if clean_signal is None or noise_signal is None:
        return None

    clean_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise_signal**2)

    # Handle cases where signals might be silent
    if clean_power == 0:
        if noise_power == 0:
            return np.zeros_like(clean_signal)
        else: # Only noise, normalize it
            return noise_signal / np.max(np.abs(noise_signal)) if np.max(np.abs(noise_signal)) > 0 else noise_signal

    # Calculate target noise power based on desired SNR
    target_noise_power = clean_power / (10**(snr_db / 10))

    # Scale noise signal to target power
    if noise_power > 0:
        scaled_noise = noise_signal * np.sqrt(target_noise_power / noise_power)
    else:
        scaled_noise = np.zeros_like(noise_signal) # No noise if original noise power is zero

    mixed_signal = clean_signal + scaled_noise

    # Normalize mixed signal to prevent clipping
    max_abs_val = np.max(np.abs(mixed_signal))
    if max_abs_val > 0:
        mixed_signal = mixed_signal / max_abs_val
    return mixed_signal

# --- Main script execution ---
# Get list of all clean guitar files
clean_guitar_files = [os.path.join(CLEAN_GUITAR_INPUT_DIR, f)
                      for f in os.listdir(CLEAN_GUITAR_INPUT_DIR)
                      if f.lower().endswith(('.wav', '.flac', '.aiff', '.mp3'))]

# Get list of all hum files
hum_files = [os.path.join(HUM_INPUT_DIR, f)
             for f in os.listdir(HUM_INPUT_DIR)
             if f.lower().endswith(('.wav', '.flac', '.aiff', '.mp3'))]

if not clean_guitar_files:
    print(f"Error: No clean guitar audio files found in {CLEAN_GUITAR_INPUT_DIR}. Please check the path and contents.")
    exit()
if not hum_files:
    print(f"Error: No hum audio files found in {HUM_INPUT_DIR}. Please check the path and contents.")
    exit()

print(f"Found {len(clean_guitar_files)} clean guitar files.")
print(f"Found {len(hum_files)} hum files.")

# Load all hums once to avoid repeated disk I/O
loaded_hums = [load_audio(h_path, SAMPLE_RATE) for h_path in hum_files]
loaded_hums = [h for h in loaded_hums if h is not None] # Filter out any failed loads

if not loaded_hums:
    print("Error: Failed to load any hum audio files. Check file integrity or format.")
    exit()

total_generated_files = 0
for i, clean_file_path in enumerate(clean_guitar_files):
    print(f"Processing clean guitar file {i+1}/{len(clean_guitar_files)}: {os.path.basename(clean_file_path)}")
    clean_guitar_audio = load_audio(clean_file_path, SAMPLE_RATE)

    if clean_guitar_audio is None:
        continue # Skip if audio loading failed

    for j in range(NUM_AUGMENTATIONS_PER_CLEAN_FILE):
        # Randomly select a hum audio
        random_hum_audio = random.choice(loaded_hums)

        # Adjust hum length to match guitar audio length (looping if necessary)
        hum_to_mix = adjust_length(random_hum_audio, len(clean_guitar_audio))

        if hum_to_mix is None:
            print(f"Warning: Skipping augmentation {j} for {os.path.basename(clean_file_path)} due to hum processing error.")
            continue

        # Mix clean guitar and hum at a random SNR
        snr_db = np.random.uniform(MIN_SNR_DB, MAX_SNR_DB)
        mixed_signal = mix_signals(clean_guitar_audio, hum_to_mix, snr_db)

        if mixed_signal is None:
            print(f"Warning: Skipping augmentation {j} for {os.path.basename(clean_file_path)} due to mixing error.")
            continue

        # Generate a unique base filename that will be used for BOTH noisy and clean
        base_filename_without_ext = os.path.splitext(os.path.basename(clean_file_path))[0]
        # Ensure unique ID is safe for filenames
        unique_file_identifier = f"{base_filename_without_ext}_aug{j}_snr{snr_db:.1f}".replace('.', '_').replace('-', 'neg')

        # The key change: The output filename is now identical for both
        output_filename = f"{unique_file_identifier}.wav"

        noisy_output_path = os.path.join(NOISY_OUTPUT_DIR, output_filename)
        clean_target_path = os.path.join(CLEAN_TARGET_OUTPUT_DIR, output_filename) # Same filename!

        try:
            # Save the mixed signal (input to model)
            sf.write(noisy_output_path, mixed_signal, SAMPLE_RATE)
            # Save the original clean guitar signal (target for model)
            sf.write(clean_target_path, clean_guitar_audio, SAMPLE_RATE)
            total_generated_files += 1
            # print(f"Generated: {noisy_output_path} | Target: {clean_target_path}") # Uncomment for verbose output
        except Exception as e:
            print(f"Error saving files for {clean_file_path}, augmentation {j}: {e}")

print("\nDataset generation complete!")
print(f"Total audio pairs generated: {total_generated_files}")
print(f"Generated noisy audio files are in: {NOISY_OUTPUT_DIR}")
print(f"Generated clean target audio files are in: {CLEAN_TARGET_OUTPUT_DIR}")