import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})

class Wave():
    def __init__(self, amplitude, f):
        self.amplitude = amplitude
        self.f = f

    def __call__(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.f * t)

class Superpose():
    def __init__(self, callables_array):
        self.callables_array = callables_array

    def __call__(self, x):
        return np.sum(np.array([func(x) for func in self.callables_array]), axis=0)

class Exercise1():
    def __init__(self):
        self.lower_limit = 0
        self.upper_limit = 1
        self.range = self.upper_limit - self.lower_limit
        self.steps = 10000
        self.spacing = self.range/self.steps

        self.parameters_array = np.array([(1., 12.), (2., 18.)]) # np array of parameters with format (amplitude, frequency (in cycles/s))

        self.waves_array = np.array([Wave(amplitude, omega) for (amplitude, omega) in self.parameters_array]) # np array of (callable) wave objects

        self.times = np.linspace(self.lower_limit, self.upper_limit, self.steps)
        superpose_waves = Superpose(self.waves_array)
        self.position_space = superpose_waves(self.times)

        self.frequencies = np.fft.fftfreq(self.steps, self.spacing) # in cycles/s
        self.frequency_space = np.fft.fft(self.position_space)

        self.ifft = np.real(np.fft.ifft(self.frequency_space))

    def plot(self, show=False, save=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32, 18), dpi=100)

        ax1.set_title('Position vs time for object in motion in 1D')
        ax1.plot(self.times, self.position_space)
        ax1.set_xlabel('time $t$ / s')
        ax1.set_ylabel('position $x$ / m')

        ax2.set_title('Amplitude vs frequency Fourier decomposition of position vs time for object in motion')
        ax2.plot(self.frequencies, np.abs(self.frequency_space))
        ax2.set_xlim(-30, 30)
        ax2.set_xlabel('frequency $f / Hz$')
        ax2.set_ylabel('amplitude $A$ / m')

        ax1.grid(visible=True, which='both')
        ax2.grid(visible=True, which='both')
        ax2.minorticks_on()

        if save:
            plt.savefig('exercise_1_plots.png')

        if show:
            plt.show()

    def ifft_plot(self, show=False, save=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32, 18), dpi=100)
        ax1.plot(self.times, self.position_space)
        ax1.set_title('Original position vs time data')
        ax1.set_xlabel('time $t$ / s')
        ax1.set_ylabel('position $x$ / m')

        ax2.plot(self.times, self.ifft)
        ax2.set_title('Inverse of FFT of original data')
        ax2.set_xlabel('time $t$ / s')
        ax2.set_ylabel('position $x$ / m')

        if save:
            plt.savefig('exercise_1_ifft_plots.png')

        if show:
            plt.show()