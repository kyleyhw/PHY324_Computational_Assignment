import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 25})

class LinearModel():
    def __init__(self, m, c):
        self.m = m
        self.c = c

    def __call__(self, x):
        return self.m * x + self.c

class Wave():
    def __init__(self, amplitude, f):
        self.amplitude = amplitude
        self.f = f

    def __call__(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.f * t)

class TimeDependentFrequencyWave():
    def __init__(self, amplitude, f_callable):
        self.amplitude = amplitude
        self.f_callable = f_callable

    def __call__(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.f_callable(t) * t)


class Exercise4():
    def __init__(self):
        self.min = 0
        self.max = 1
        self.range = self.max - self.min
        self.steps = 10000
        self.spacing = self.range/self.steps

        self.omega_slope = 8
        self.omega_intercept = 10 # (2*np.pi)/self.steps
        self.amplitude = 1

        self.omega = LinearModel(self.omega_slope, self.omega_intercept)

        wave = TimeDependentFrequencyWave(self.amplitude, self.omega)

        self.times = np.linspace(self.min, self.max, self.steps)
        self.position_space = wave(self.times)

        self.frequencies = np.fft.fftfreq(self.steps, self.spacing) # in cycles/s
        self.amplitudes = np.fft.fft(self.position_space)

    def plot(self, save=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32, 9))

        ax1.grid(visible=True, which='both')
        ax2.grid(visible=True, which='both')
        ax2.minorticks_on()

        ax1.plot(self.times, self.position_space)
        ax1.set_ylabel('position $x$ / m', fontsize=30)
        ax1.set_xlabel('time $t$ / s', fontsize=30)
        ax1.set_title('Position-time plot for object in motion', fontsize=40)


        ax2.plot(self.frequencies, np.abs(self.amplitudes))
        ax2.set_ylabel('amplitude / m', fontsize=30)
        ax2.set_xlabel('frequency f / Hz', fontsize=30)
        ax2.set_title('FT of position-time for object in motion', fontsize=40)
        ax2.set_xlim(-40, 40)

        fig.subplots_adjust(hspace=1)

        if save:
            plt.savefig('exercise_4_plots.png')

        plt.show()