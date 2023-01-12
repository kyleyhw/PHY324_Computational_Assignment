import numpy as np
import matplotlib.pyplot as plt

class wave():
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, t):
        return np.sin(self.omega * t)

class Exercise_1():
    def __init__(self):
        self.lower_limit = 0
        self.upper_limit = 1
        self.range = self.upper_limit - self.lower_limit
        self.steps = 10000
        self.spacing = self.range/self.steps

        self.omega_1 = 12 * 2 * np.pi / self.range
        self.omega_2 = 18 * 2 * np.pi / self.range

        self.wave_1 = wave(omega=self.omega_1)
        self.wave_2 = wave(omega=self.omega_2)

        self.times = np.linspace(self.lower_limit, self.upper_limit, self.steps)
        self.position_space = self.wave_1(self.times) + self.wave_2(self.times)

        self.frequencies = np.fft.fftfreq(self.steps, self.spacing)
        self.frequency_space = np.abs(np.fft.fft(self.position_space))

    def plot(self, save=False):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(self.times, self.position_space)
        ax2.plot(self.frequencies, self.frequency_space)
        ax2.set_xlim(-30, 30)
        ax1.grid(visible=True, which='both')
        ax2.grid(visible=True, which='both')
        ax2.minorticks_on()
        plt.show()
        if save:
            plt.savefig('exercise_1_plot.png')
        plt.close()

