import numpy as np
from matplotlib import pyplot as plt

class LinearModel():
    def __init__(self, m, c):
        self.m = m
        self.c = c

    def __call__(self, x):
        return self.m * x + self.c

class Wave():
    def __init__(self, amplitude, omega):
        self.amplitude = amplitude
        self.omega = omega

    def __call__(self, t):
        return self.amplitude * np.sin(self.omega * t)

class TimeDependentFrequencyWave():
    def __init__(self, amplitude, omega_callable):
        self.amplitude = amplitude
        self.omega_callable = omega_callable

    def __call__(self, t):
        return self.amplitude * np.sin(self.omega_callable(t) * t)


class Exercise4():
    def __init__(self):
        self.min = 0
        self.max = 1
        self.range = self.max - self.min
        self.steps = 10000
        self.spacing = self.range/self.steps

        self.omega_slope = 100
        self.omega_intercept = (2*np.pi)/self.steps
        self.amplitude = 1

        self.omega = LinearModel(self.omega_slope, self.omega_intercept)

        wave = TimeDependentFrequencyWave(self.amplitude, self.omega)

        self.times = np.linspace(self.min, self.max, self.steps)
        self.position_space = wave(self.times)

        self.frequencies = np.fft.fftfreq(self.steps, self.spacing)
        self.amplitudes = np.fft.fft(self.position_space)

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(self.times, self.position_space)

        ax2.plot(self.frequencies, np.abs(self.amplitudes))
        ax2.set_xlim(-100, 100)

        plt.show()