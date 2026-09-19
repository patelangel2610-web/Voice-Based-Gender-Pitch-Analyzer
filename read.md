# Voice-Based Gender and Pitch Analyzer

A Python-based Signals and Systems mini-project that estimates human voice pitch (Fundamental Frequency, F0) using FFT and Autocorrelation, and classifies it based on predefined frequency thresholds.

## Features
* **Real-Time Audio Capture:** Records 3 seconds of mono audio using `sounddevice`.
* **Signal Preprocessing:** Includes DC offset removal, Hann windowing, and RMS-based silence rejection.
* **Frequency Analysis (FFT):** Computes the Fast Fourier Transform to analyze the 0–300 Hz frequency spectrum.
* **Autocorrelation Pitch Detection:** Accurately isolates the fundamental period ($T_0$) to avoid harmonic octave errors.
* **Biological Overlap Zone:** Classifies pitch into Male (<145 Hz), Female (>185 Hz), or an Androgynous Overlap Zone (145–185 Hz).
* **Tkinter GUI:** Non-freezing interface built with Python threading and Matplotlib integration.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python main.py`
3. Click **START** and hold a steady vowel sound (like "Ahhhh") for 3 seconds.