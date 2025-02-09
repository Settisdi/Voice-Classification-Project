Voice Classification Using DFT & Signal Processing

This project implements a voice classification system that distinguishes between male and female voices based on signal energy and frequency characteristics. It utilizes fundamental signal processing concepts, including Discrete Fourier Transform (DFT), Inverse Discrete Fourier Transform (IDFT), and energy computation to analyze and classify voice signals.

📌 Features

✅ Load and process .wav audio files
✅ Implement DFT & IDFT from scratch (without libraries)
✅ Apply frequency-domain filtering to remove noise
✅ Compute signal energy
✅ Classify voices as male or female based on energy comparison

🛠️ Installation & Setup

🔹 Clone the Repository
git clone https://github.com/settisdi/Voice-Classification-Project.git
cd voice-classification
🔹 Install Dependencies 
pip install (libraries)
🔹 Run the Code
python proj.py

📝 Methodology

1️⃣ Load Audio Signals
.wav files are read and converted into discrete-time signals.
The sample rate and time values are extracted for processing.
2️⃣ Implement DFT & IDFT
DFT transforms signals from the time domain to the frequency domain.
IDFT reconstructs signals back into the time domain after processing.
3️⃣ Apply Frequency Filtering
Noise is removed by zeroing out unwanted frequencies (e.g., keeping only 50–5000 Hz).
4️⃣ Compute Signal Energy
The energy of the signal is calculated to determine its intensity and characteristics.
5️⃣ Classify as Male or Female
The system compares the computed energy with reference values of male and female voices.
A decision is made based on predefined thresholds.
📊 Results & Analysis

The classification system correctly identifies male and female voices based on their energy levels.
The DFT visualization shows clear differences between voice types.
Noise filtering significantly improves accuracy.
