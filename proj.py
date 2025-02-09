import numpy as np
import wave
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

###############################################################################
def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal x
    without using any FFT library.
    
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j 2*pi*k*n / N)
    """
    N = len(x)
    # Convert to complex128 for safety
    x = np.asarray(x, dtype=np.complex128)
    X = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        s = 0.0 + 0.0j
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s
    return X

def idft(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of a 1D signal X
    without using any FFT library.
    
    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j 2*pi*k*n / N)
    """
    N = len(X)
    X = np.asarray(X, dtype=np.complex128)
    x = np.zeros(N, dtype=np.complex128)
    
    for n in range(N):
        s = 0.0 + 0.0j
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            s += X[k] * np.exp(angle)
        x[n] = s / N
    return x

###########################################################################################

# 1) Load Audio Signals
# Using the provided function to load .wav files and extract discrete signals.
################################################################################
def record_audio(file_name="output.wav", duration=2, sample_rate=44100):
    """
    Record audio from the microphone and save it as a .wav file.
    
    Parameters:
    - file_name: Name of the output file (default: "output.wav").
    - duration: Recording duration in seconds (default: 5 seconds).
    - sample_rate: Sampling rate in Hz (default: 44100 Hz).
    """
    print(f"Recording for {duration} seconds...")

    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    
    print("Recording complete. Saving to file...")

    # Save audio data as a WAV file
    write(file_name, sample_rate, audio_data)
    print(f"Audio saved as {file_name}!")


def get_discrete_signal(file_path, resample_rate=None):
    """Load a .wav file and return its samples, time axis, and original sample rate."""
    with wave.open(file_path, 'r') as wav_file:
        original_sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / original_sample_rate
        n_channels = wav_file.getnchannels()
        raw_data = wav_file.readframes(n_frames)

        # Convert buffer to int16
        signal = np.frombuffer(raw_data, dtype=np.int16)

        # If there are multiple channels, we'll pick the first channel
        if n_channels > 1:
            signal = signal[::n_channels]

        # Create time array
        time_values = np.linspace(0, duration, len(signal), endpoint=False)

        # Optionally resample if a different rate is specified
        if resample_rate and resample_rate != original_sample_rate:
            from scipy.signal import resample
            num_samples = int(duration * resample_rate)
            signal = resample(signal, num_samples)
            time_values = np.linspace(0, duration, len(signal), endpoint=False)

    return signal, time_values, original_sample_rate

    ################################


# DFT using numpy's FFT
def dft(signal):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal using numpy's FFT.
    """
    return np.fft.fft(signal)

# IDFT using numpy's IFFT
def idft(frequency_signal):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of a 1D signal using numpy's IFFT.
    """
    return np.fft.ifft(frequency_signal)

# Compute energy of the signal
def compute_energy(signal):
    """
    Compute the energy of a discrete-time signal using its squared magnitude.
    """
    return np.sum(np.abs(signal) ** 2)

####################################################

def filter_frequency_range(signal, sample_rate, freq_min=50, freq_max=5000):
    """Remove frequencies outside the range [freq_min, freq_max] by zeroing them in the spectrum."""
    # 3a. Transform to frequency domain via DFT
    X = dft(signal)
    N = len(X)

    # Frequency resolution per bin
    freq_resolution = sample_rate / N

    # Copy spectrum for filtering
    X_filtered = np.copy(X)

    # 3b. Zero out frequencies outside the specified range
    for k in range(N):
        # Compute the frequency in Hz for index k
        freq = k * freq_resolution
        # For k > N/2, consider negative frequencies
        if k > N // 2:
            freq = (k - N) * freq_resolution
        # Check if freq is outside the passband
        if (abs(freq) < freq_min) or (abs(freq) > freq_max):
            X_filtered[k] = 0

    # 3c. Transform back to time domain via IDFT
    filtered_signal = idft(X_filtered)
    return filtered_signal


def classify_voice(signal, sample_rate, avg_energy_male, avg_energy_female, low_freq=50, high_freq=5000):
    # Step 1: Filter the signal
    filtered_signal = filter_frequency_range(signal, sample_rate, freq_min=low_freq, freq_max=high_freq)

    # Step 2: Compute the energy of the filtered signal
    energy_value = compute_energy(filtered_signal)

    # Step 3: Compute the difference between the signal energy and the average reference energies
    diff_male = abs(energy_value - avg_energy_male)
    diff_female = abs(energy_value - avg_energy_female)

    # Step 4: Classify based on which energy is closer
    if diff_male < diff_female:
        return "Male"
    else:
        return "Female"


if __name__ == "__main__":
    file_name = "test.wav"  # Output file name
    duration = 4  # Record for 5 seconds
    sample_rate = 44100  # Sampling rate in Hz

    record_audio(file_name, duration, sample_rate)

    # File paths for male and female samples
    male_file_paths = [
        "./mahyar.wav",
        "./afshin.wav",

    ]

    female_file_paths = [
        "./setayesh.wav",
        "./kiana.wav"
    ]

    # Load signals and compute filtered energy for each sample
    male_energies = []
    for file_path in male_file_paths:
        signal, time_values, sr = get_discrete_signal(file_path)
        filtered_signal = filter_frequency_range(signal, sr, freq_min=50, freq_max=5000)
        male_energies.append(compute_energy(filtered_signal))

    female_energies = []
    for file_path in female_file_paths:
        signal, time_values, sr = get_discrete_signal(file_path)
        filtered_signal = filter_frequency_range(signal, sr, freq_min=50, freq_max=5000)
        female_energies.append(compute_energy(filtered_signal))

    # Compute average energies
    avg_energy_male = np.mean(male_energies)
    avg_energy_female = np.mean(female_energies)

    print(f"Average Male Energy: {avg_energy_male}")
    print(f"Average Female Energy: {avg_energy_female}")

    # Classify a new voice recording
    new_file_path = "./test.wav"  # Update with correct file path
    new_signal, new_time, new_sr = get_discrete_signal(new_file_path)

    # Classify the new signal based on its energy
    result = classify_voice(new_signal, new_sr, avg_energy_male, avg_energy_female)
    print(f"Classification Result: {result}")

    # Visualization for the input signal to classify

    # Compute FFT for the input signal and its filtered version
    new_fft = np.fft.fft(new_signal)
    new_filtered = filter_frequency_range(new_signal, new_sr, freq_min=50, freq_max=5000)
    new_filtered_fft = np.fft.fft(new_filtered)
    freqs_new = np.fft.fftfreq(len(new_signal), 1 / new_sr)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    # 1. Plot the input signal in the time domain
    axs[0].plot(new_time, new_signal, label="Original Signal")
    axs[0].plot(new_time, new_filtered, label="Filtered Signal", color='r', alpha=0.7)
    axs[0].set_title("Input Signal Before and After Filtering (Time Domain)")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # 2. Plot the input signal in the frequency domain (FFT before filtering)
    axs[1].plot(freqs_new[:len(freqs_new) // 2], np.abs(new_fft[:len(freqs_new) // 2]), label="Original FFT")
    axs[1].set_title("Input Signal in Frequency Domain (Before Filtering)")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()

    # 3. Plot the filtered signal in the frequency domain (FFT after filtering)
    axs[2].plot(freqs_new[:len(freqs_new) // 2], np.abs(new_filtered_fft[:len(freqs_new) // 2]), label="Filtered FFT", color='r')
    axs[2].set_title("Filtered Signal in Frequency Domain (After Filtering)")
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Magnitude")
    axs[2].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()
