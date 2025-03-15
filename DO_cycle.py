import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

class BipolarSeesawModel:
    def __init__(self, tau=1000, dt=50, T_max=40000, alpha=0.6, TN=None):
        self.tau = tau      # Characteristic timescale (years)
        self.dt = dt       # Time step (years)
        self.T_max = T_max    # Simulation length (years)
        self.t = np.arange(0, self.T_max, self.dt)  # Time array
        self.alpha = alpha    # Red noise parameter

        # Generate T_N only if not provided (for comparison)
        self.TN = TN if TN is not None else self.generate_tn()

    def generate_tn(self):
        """
        An on-off signal for T_N
        """
        TN = np.zeros_like(self.t)
        current_state = 1
        t_switch = 0

        while t_switch < self.T_max:
            duration = np.random.uniform(2000, 5000)
            next_switch = t_switch + duration
            TN[(self.t >= t_switch) & (self.t < next_switch)] = current_state
            current_state *= -1      # Switch polarity between 1 and -1
            t_switch = next_switch

        return TN

    def compute_ts_convolution(self):
        """
        T_S(t) = - (1/tau) ∫ T_N(t - t') e^(-t'/tau) dt'
        """
        t_prime = np.arange(0, 10 * self.tau, self.dt)  # Kernel range - i use (0, 10τ) to approximate (0, t)
        kernel = np.exp(-t_prime / self.tau) / self.tau  # Exponential decay kernel
        kernel /= np.sum(kernel * self.dt)         # Normalize to ensure the weight sums to 1

        # Compute convolution using FFT
        TS = -fftconvolve(self.TN, kernel, mode='full')[:len(self.TN)] * self.dt
        self.TS = TS
        return TS

    def generate_red_noise(self):
        """
        r(t) = alpha * r(t - Δt) + ε(t)
        """
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
        return self.TN, self.TS, self.TS_noisy



def run_scenarios(scenarios=None, single_scenario=None):
    TN_fixed = BipolarSeesawModel().generate_tn()

    # If running a single scenario
    if single_scenario:
        scenarios = [single_scenario]

    # Run models for each scenario
    models = [BipolarSeesawModel(tau=sc["tau"], alpha=sc["alpha"], TN=TN_fixed) for sc in scenarios]
    results = [model.run() for model in models]

    # Convert time to kyr BP
    T_max = models[0].T_max
    t = models[0].t
    kyrBP = (T_max - t) / 1000

    fig, axes = plt.subplots(nrows=len(scenarios) + 1, figsize=(10, 3 * (len(scenarios) + 1)), sharex=True)

    axes[0].plot(kyrBP, TN_fixed, 'k', label=r"$T_N$")
    axes[0].set_ylabel(r"$T_N$ Amplitude")
    axes[0].legend()
    axes[0].set_ylim(-3, 3)

    for i, (sc, (TN, TS, TS_noisy)) in enumerate(zip(scenarios, results)):
        axes[i + 1].plot(kyrBP, TS, 'royalblue', label=rf"$T_S$, $\tau={sc['tau']}$, $\alpha={sc['alpha']}$")
        axes[i + 1].plot(kyrBP, TS_noisy, 'mediumvioletred', alpha=0.4, label=r"$T_S^*$ (with red noise)")
        axes[i + 1].set_ylabel(r"$T_S$ Amplitude")
        axes[i + 1].legend()
        axes[i + 1].set_ylim(-3, 3)

    axes[-1].set_xlabel("Age (kyr BP)")
    plt.tight_layout()
    plt.show()

run_scenarios(single_scenario={"tau": 1500, "alpha": 0})

scenarios = [
    {"tau": 1000, "alpha": 0.5},
    {"tau": 1000, "alpha": 0.9},
    {"tau": 500, "alpha": 0.5},
    {"tau": 4000, "alpha": 0.5},
]

run_scenarios(scenarios=scenarios)