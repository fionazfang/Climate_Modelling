
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, filtfilt


class BipolarSeesawModel:
    def __init__(self, ngrip_data, tau=1000, dt=50, alpha=0.6):
        self.tau = tau      # Characteristic timescale
        self.dt = dt        # Time step for uniform interpolation
        self.alpha = alpha  # Red noise parameter

        self.process_ngrip_data(ngrip_data)

    def process_ngrip_data(self, ngrip_data):
        ngrip_data = ngrip_data[(ngrip_data["age"] >= 25000) & (ngrip_data["age"] <= 65000)]
        ngrip_data = ngrip_data.sort_values("age", ascending=True)
        self.t = np.arange(ngrip_data["age"].min(), ngrip_data["age"].max(), self.dt)
        self.TN = np.interp(self.t, ngrip_data["age"], ngrip_data["temperature"])

        self.TN = self.highpass_filter(self.TN, cutoff=1 / 8000)

    def highpass_filter(self, data, cutoff):
        nyquist = 1 / (2 * self.dt)  # Nyquist frequency
        b, a = butter(2, cutoff / nyquist, btype='high')
        return filtfilt(b, a, data)

    def compute_ts_convolution(self):
        t_prime = np.arange(0, 10 * self.tau, self.dt)
        kernel = np.exp(-t_prime / self.tau) / self.tau  # Exponential decay kernel
        kernel /= np.sum(kernel * self.dt)               # Normalize
        TS = -fftconvolve(self.TN, kernel, mode='full')[:len(self.TN)] * self.dt
        self.TS = TS
        return TS

    def generate_red_noise(self):
        red_noise = np.zeros_like(self.TS)
        white_noise = np.random.normal(0, 0.4 * np.std(self.TS), size=len(self.TS))

        for i in range(1, len(self.TS)):
            red_noise[i] = self.alpha * red_noise[i - 1] + white_noise[i]

        self.red_noise = red_noise
        self.TS_noisy = self.TS + red_noise
        return red_noise

    def run(self):
        self.compute_ts_convolution()
        self.generate_red_noise()
        return self.t, self.TN, self.TS, self.TS_noisy


# Load NGRIP data
ngrip_data = pd.read_excel("kindler2014_ngrip_temperature.xlsx")

# Load EPICA data
file_path = "EDC_dD_temp_estim.tab"
df_epica = pd.read_csv(file_path, sep="\t", skiprows=20)  # Skipping metadata rows
df_epica = df_epica[["Age model [ka]", "delta T [°C]"]].dropna()
df_epica.columns = ["age", "delta_T"]
df_epica["age"] *= 1000  # Multiply by 1000 to get years
df_epica = df_epica[(df_epica["age"] >= 25000) & (df_epica["age"] <= 65000)]
df_epica = df_epica.sort_values("age")

def highpass_filter(data, cutoff, dt):
    nyquist = 1 / (2 * dt)  # Nyquist frequency
    b, a = butter(2, cutoff / nyquist, btype='high')
    return filtfilt(b, a, data)

dt_epica = np.mean(np.diff(df_epica["age"]))  # Calculate mean timestep
df_epica["delta_T"] = highpass_filter(df_epica["delta_T"], cutoff=1 / 8000, dt=dt_epica)



# Run the model with different tau values
scenarios = [500, 1120, 2000, 4000]
models = [BipolarSeesawModel(ngrip_data, tau=tau, alpha=0.6) for tau in scenarios]
results = [model.run() for model in models]

# Plot results
fig, axes = plt.subplots(nrows=len(scenarios) + 2, figsize=(10, 3 * (len(scenarios) + 2)), sharex=True)

kyrBP = results[0][0]  # Time axis

# Plot NGRIP Temperature (Northern Hemisphere)
axes[0].plot(kyrBP, results[0][1], 'k', label="NGRIP Temperature")
axes[0].set_ylabel("T_N Amplitude")
axes[0].legend()
axes[0].set_ylim(-10, 10)

# Plot EPICA Temperature (Southern Hemisphere)
axes[1].plot(df_epica["age"], df_epica["delta_T"], 'firebrick', label="EPICA ΔT")
axes[1].set_ylabel("EPICA ΔT (°C)")
axes[1].legend()
axes[1].set_ylim(-10, 10)

# Plot the modeled scenarios for different τ values
for i, (tau, result) in enumerate(zip(scenarios, results)):
    _, TN, TS, TS_noisy = result
    axes[i + 2].plot(kyrBP, TS, 'royalblue', label=f"T_S (τ={tau})")
    axes[i + 2].plot(kyrBP, TS_noisy, 'mediumvioletred', alpha=0.4, label="T_S* (with red noise)")
    axes[i + 2].set_ylabel("T_S Amplitude")
    axes[i + 2].legend()
    axes[i + 2].set_ylim(-10, 10)

axes[-1].set_xlabel("Age (ya BP)")
plt.tight_layout()
plt.show()
