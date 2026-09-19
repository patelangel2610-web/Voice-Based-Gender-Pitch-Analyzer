import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- CONFIGURATION VARIABLES ---
SAMPLE_RATE = 44100
DURATION = 3          # Recording duration in seconds
LOWER_BOUND = 145.0   # Below this is classified as strictly Male Estimate
UPPER_BOUND = 185.0   # Above this is classified as strictly Female Estimate
RMS_THRESHOLD = 0.01  # Minimum audio volume to be considered "voice"

class VoiceAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice-Based Gender / Pitch Analyzer")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f0f0")

        # --- TOP HEADER ---
        title_label = tk.Label(root, text="VOICE-BASED GENDER / PITCH ANALYZER", font=("Arial", 20, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        disclaimer = tk.Label(root, text=f"Threshold Bounds: {LOWER_BOUND} Hz to {UPPER_BOUND} Hz\n(Note: Human pitch ranges overlap in the middle. Estimates do not determine biological gender.)", 
                              font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555")
        disclaimer.pack()

        # --- STATUS AND RESULTS ---
        self.status_label = tk.Label(root, text="Status: Ready", font=("Arial", 14), bg="#f0f0f0", fg="blue")
        self.status_label.pack(pady=5)

        self.pitch_label = tk.Label(root, text="Pitch: --- Hz | F0: --- Hz", font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.pitch_label.pack(pady=5)

        # --- CLASSIFICATION PANELS ---
        self.panels_frame = tk.Frame(root, bg="#f0f0f0")
        self.panels_frame.pack(pady=10, fill="x", padx=50)

        self.left_panel = tk.Frame(self.panels_frame, bg="lightgray", width=400, height=100, relief="ridge", bd=4)
        self.left_panel.pack(side="left", expand=True, fill="both", padx=10)
        self.left_panel.pack_propagate(False)
        self.left_label_1 = tk.Label(self.left_panel, text="LOWER PITCH", font=("Arial", 16, "bold"), bg="lightgray")
        self.left_label_1.pack(pady=(20, 0))
        self.left_label_2 = tk.Label(self.left_panel, text="Male Estimate", font=("Arial", 14), bg="lightgray")
        self.left_label_2.pack()

        self.right_panel = tk.Frame(self.panels_frame, bg="lightgray", width=400, height=100, relief="ridge", bd=4)
        self.right_panel.pack(side="right", expand=True, fill="both", padx=10)
        self.right_panel.pack_propagate(False)
        self.right_label_1 = tk.Label(self.right_panel, text="HIGHER PITCH", font=("Arial", 16, "bold"), bg="lightgray")
        self.right_label_1.pack(pady=(20, 0))
        self.right_label_2 = tk.Label(self.right_panel, text="Female Estimate", font=("Arial", 14), bg="lightgray")
        self.right_label_2.pack()

        # --- BUTTONS ---
        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="START (Record 3s)", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=20, command=self.start_recording)
        self.start_btn.pack(side="left", padx=10)

        self.reset_btn = tk.Button(btn_frame, text="RESET", font=("Arial", 14, "bold"), bg="#f44336", fg="white", width=10, command=self.reset_gui)
        self.reset_btn.pack(side="left", padx=10)

        # --- MATPLOTLIB GRAPHS ---
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)
        
        self.reset_plots()

    def reset_plots(self):
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()
        
        self.ax1.set_title("Time-Domain Voice Signal")
        self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Amplitude")

        self.ax2.set_title("Frequency-Domain Representation (0–300 Hz)")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")

        self.ax3.set_title("Autocorrelation for Pitch Detection")
        self.ax3.set_xlabel("Lag / Period (seconds)")
        self.ax3.set_ylabel("Normalized Autocorrelation")

        self.canvas.draw()

    def reset_gui(self):
        self.status_label.config(text="Status: Ready", fg="blue")
        self.pitch_label.config(text="Pitch: --- Hz | F0: --- Hz")
        self.set_panel_colors("reset")
        self.reset_plots()
        self.start_btn.config(state="normal")

    def set_panel_colors(self, state):
        # Colors: reset (gray), male (blue), female (pink), overlap (yellow)
        if state == "reset":
            bg_l, bg_r = "lightgray", "lightgray"
        elif state == "male":
            bg_l, bg_r = "#add8e6", "lightgray"  # Light blue left
        elif state == "female":
            bg_l, bg_r = "lightgray", "#ffb6c1"  # Light pink right
        elif state == "overlap":
            bg_l, bg_r = "#ffe4b5", "#ffe4b5"    # Yellow for both sides
        else:
            bg_l, bg_r = "lightgray", "lightgray"

        self.left_panel.config(bg=bg_l)
        self.left_label_1.config(bg=bg_l)
        self.left_label_2.config(bg=bg_l)
        
        self.right_panel.config(bg=bg_r)
        self.right_label_1.config(bg=bg_r)
        self.right_label_2.config(bg=bg_r)

    def start_recording(self):
        self.start_btn.config(state="disabled")
        self.reset_gui()
        self.status_label.config(text="Status: Recording... Speak now!", fg="red")
        
        # Run audio capture in a background thread to prevent GUI freezing
        threading.Thread(target=self.record_and_analyze, daemon=True).start()

    def record_and_analyze(self):
        try:
            # 1. Acquire Signal
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            audio = audio_data.flatten()

            # 2. Preprocess Signal
            audio = audio - np.mean(audio)  # Remove DC offset
            
            # Check for silence using RMS energy
            rms = np.sqrt(np.mean(audio**2))
            if rms < RMS_THRESHOLD:
                self.root.after(0, self.update_failure, "Status: No voice detected (Too quiet)")
                return

            # Apply Hann Window
            windowed_audio = audio * np.hanning(len(audio))

            # 3. FFT (0 - 300 Hz)
            N = len(windowed_audio)
            freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
            fft_mag = np.abs(np.fft.rfft(windowed_audio))
            
            # Limit to 0-300Hz for plotting
            valid_freq_idx = np.where(freqs <= 300)[0]
            freqs_300 = freqs[valid_freq_idx]
            fft_mag_300 = fft_mag[valid_freq_idx]

            # 4. Autocorrelation (FFT-based for speed)
            fvi = np.fft.fft(windowed_audio, n=2*N)
            acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi)))[:N]
            acf = acf / acf[0]  # Normalize

            # 5. Extract Fundamental Period (T0) and Pitch (F0)
            # Search only between 70 Hz and 300 Hz
            min_lag = int(SAMPLE_RATE / 300)
            max_lag = int(SAMPLE_RATE / 70)
            
            valid_acf = acf[min_lag:max_lag]
            if len(valid_acf) == 0:
                self.root.after(0, self.update_failure, "Status: No reliable pitch found")
                return
                
            peak_idx_in_valid = np.argmax(valid_acf)
            actual_peak_lag = peak_idx_in_valid + min_lag
            
            t0 = actual_peak_lag / SAMPLE_RATE
            f0 = SAMPLE_RATE / actual_peak_lag

            # Sanity check on confidence
            if acf[actual_peak_lag] < 0.2:
                self.root.after(0, self.update_failure, "Status: No reliable pitch (Unvoiced/Noise)")
                return

            # 6. Push data back to main thread for GUI update
            time_axis = np.linspace(0, DURATION, len(audio))
            lags_axis = np.arange(len(acf)) / SAMPLE_RATE
            
            self.root.after(0, self.update_success, audio, time_axis, freqs_300, fft_mag_300, lags_axis, acf, f0, t0, actual_peak_lag)

        except Exception as e:
            self.root.after(0, self.update_failure, f"Error: {str(e)}")

    def update_failure(self, msg):
        self.status_label.config(text=msg, fg="red")
        self.start_btn.config(state="normal")

    def update_success(self, audio, time_axis, freqs, fft_mag, lags, acf, f0, t0, peak_lag):
        # 1. Update Pitch Label
        self.pitch_label.config(text=f"Pitch: {f0:.2f} Hz | F0: {f0:.2f} Hz")

        # 2. Apply Classification with Overlap Zone
        if f0 < LOWER_BOUND:
            self.set_panel_colors("male")
            self.status_label.config(text="Status: Voice detected (Clear Lower Pitch)", fg="green")
        elif f0 > UPPER_BOUND:
            self.set_panel_colors("female")
            self.status_label.config(text="Status: Voice detected (Clear Higher Pitch)", fg="green")
        else:
            self.set_panel_colors("overlap")
            self.status_label.config(text="Status: Voice detected (Overlap / Androgynous Zone)", fg="#d2691e")

        # 3. Update Plots
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()

        # Plot 1: Time Domain
        self.ax1.plot(time_axis, audio, color="blue", alpha=0.7)
        self.ax1.set_title("Time-Domain Voice Signal")
        self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Amplitude")

        # Plot 2: Frequency Domain (FFT)
        self.ax2.plot(freqs, fft_mag, color="purple")
        self.ax2.axvline(x=f0, color="red", linestyle="--", label=f"Estimated F0 ({f0:.1f} Hz)")
        self.ax2.set_title("Frequency-Domain Representation (0–300 Hz)")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")
        self.ax2.legend(loc="upper right")

        # Plot 3: Autocorrelation
        display_lag_limit = int(SAMPLE_RATE * 0.02) 
        self.ax3.plot(lags[:display_lag_limit], acf[:display_lag_limit], color="green")
        self.ax3.axvline(x=t0, color="red", linestyle="--", label=f"Peak T0 = {t0:.4f}s")
        self.ax3.plot(t0, acf[peak_lag], 'ro')
        self.ax3.set_title("Autocorrelation for Pitch Detection")
        self.ax3.set_xlabel("Lag / Period (seconds)")
        self.ax3.set_ylabel("Normalized Autocorrelation")
        self.ax3.legend(loc="upper right")

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
        
        self.start_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAnalyzerGUI(root)
    root.mainloop()