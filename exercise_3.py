import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 25})
import pickle as pkl

class Gaussian():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        exponent = -((x - self.mu) ** 2) / self.sigma
        return np.exp(exponent)

class GaussianSymmetricalAboutZero():
    def __init__(self, mu, sigma):
        self.Gaussian = Gaussian(mu, sigma)

    def __call__(self, x):
        return self.Gaussian(x) + self.Gaussian(-x)

class Superpose():
    def __init__(self, callables_array):
        self.callables_array = callables_array

    def __call__(self, x):
        return np.sum(np.array([func(x) for func in self.callables_array]), axis=0)

class filter():
    def __init__(self, x, noisy_data_array, filter_callable):
        self.x = x
        self.noisy_data_array = noisy_data_array
        self.filter_callable = filter_callable
        self.filter_points = self.filter_callable(self.x)
        self.filtered_data = self.noisy_data_array * self.filter_points

class Wave():
    def __init__(self, amplitude, f):
        self.amplitude = amplitude
        self.f = f

    def __call__(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.f * t)




class Exercise3():
    def __init__(self):

        with open('noisy_sine_wave', 'rb') as file:
            self.data_from_file = pkl.load(file)

        self.number_of_points = len(self.data_from_file)
        self.spacing = 1

        self.positions = self.data_from_file
        self.times = np.arange(len(self.positions))

        self.amplitudes = np.fft.fft(self.positions)
        self.frequencies = np.fft.fftfreq(self.number_of_points, self.spacing)

        self.gaussian_filter_parameters_array = np.array([(0.059, 0.00004), (0.077, 0.00004), (0.14, 0.0001)])
        self.symmetrical_gaussians = np.array([GaussianSymmetricalAboutZero(*params) for params in self.gaussian_filter_parameters_array])

        self.filter_function = Superpose(self.symmetrical_gaussians)

        self.filtered_data = filter(self.frequencies, self.amplitudes, self.filter_function).filtered_data

        self.cleaned_positions = np.real(np.fft.ifft(self.filtered_data)) # ifft is real-valued, imaginary part is identically zero; np.real converts data type

        self.best_guess_parameters_array = np.array([(23.44, 0.059), (50.07, 0.077), (94.84, 0.143)]) # sine wave parameters in form (amplitude, frequency)
        self.best_guess_waves = np.array([Wave(*params) for params in self.best_guess_parameters_array])
        self.best_guess_function = Superpose(self.best_guess_waves)


    def plot(self):
        fig, ( (ax1, ax2), (ax3, ax4), (ax5, ax6) ) = plt.subplots(3, 2, sharex='col', sharey=False)

        ax1.plot(self.times, self.positions, label='noisy data') # position-time plot of noisy data
        ax1.plot(self.times, self.cleaned_positions, color='orange', label='cleaned data') # position-time plot of cleaned data
        ax1.set_title('Position-time plot of data')
        ax1.set_xlim(800, 1200)

        ax3.plot(self.times, self.positions - self.cleaned_positions) # residuals
        ax3.set_title('Residuals between noisy and cleaned data')

        ax5.plot(self.times, self.best_guess_function(self.times)) # best guess
        ax5.set_title('Best guess wave')



        ax2.plot(self.frequencies, np.abs(self.amplitudes)) # amplitude-frequency plot
        ax2.set_title('Amplitude-frequency plot of FFT of noisy data')

        ax4.plot(self.frequencies, self.filter_function(self.frequencies)) # filter function plot
        ax4.set_title('Filter function (sum of Gaussians)')

        ax6.plot(self.frequencies, np.abs(self.filtered_data)) # filtered amplitude-frequencies plot
        ax6.set_title('Filtered amplitude-frequency plot of FFT of noisy data')


        plt.show()

    def plot_frequency_domain_for_report(self, show=False, save=False):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(32, 18), sharex='col')

        ax1.plot(self.frequencies, np.abs(self.amplitudes))  # amplitude-frequency plot
        ax1.set_title('Amplitude-frequency plot of FFT of noisy data')
        ax1.set_ylabel('amplitude / m')
        ax1.set_xlabel('frequency f / Hz')

        ax2.plot(self.frequencies, self.filter_function(self.frequencies))  # filter function plot
        ax2.set_title('Filter function F (sum of Gaussians)')
        ax2.set_ylabel('F(t)')
        ax2.set_xlabel('frequency f / Hz')

        ax3.plot(self.frequencies, np.abs(self.filtered_data))  # filtered amplitude-frequencies plot
        ax3.set_title('Filtered amplitude-frequency plot of FFT of noisy data')
        ax3.set_ylabel('amplitude / m')
        ax3.set_xlabel('frequency f / Hz')


        if save:
            plt.savefig('exercise_3_frequency_domain_plots.png')

        if show:
            plt.show()

    def plot_position_domain_for_report(self, show=False, save=False):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(32, 18), sharex='col')
        ax1.set_xlim(800, 1200)

        ax1.plot(self.times, self.positions) # position-time plot of noisy data
        ax1.set_title('Position-time plot of noisy data')
        ax1.set_ylabel('position $x$ / m')

        ax2.plot(self.times, self.cleaned_positions)
        ax2.set_title('Position-time plot of cleaned data')
        ax2.set_ylabel('position $x$ / m')

        ax3.plot(self.times, self.positions - self.cleaned_positions)
        ax3.set_title('Residuals between noisy data and cleaned data')
        ax3.set_ylabel('residual / m')
        ax3.set_xlabel('time $t$ / s')

        if save:
            plt.savefig('exercise_3_position_domain_plots.png')

        if show:
            plt.show()

    def plot_best_guess(self, show=False, save=False):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(32, 18), sharex='col')
        ax1.set_xlim(800, 1200)

        ax1.plot(self.times, self.positions) # position-time plot of noisy data
        ax1.set_ylabel('position $x$ / m')
        ax1.set_title('Position-time plot of noisy data')

        ax2.plot(self.times, self.cleaned_positions) # cleaned data
        ax2.set_ylabel('position $x$ / m')
        ax2.set_title('Position-time plot of cleaned data')

        ax3.plot(self.times, self.best_guess_function(self.times)) # best guess
        ax3.set_ylabel('Position $x$ / m')
        ax3.set_title('Best guess wave')
        ax3.set_xlabel('time $t$ / s')

        if save:
            plt.savefig('exercise_3_best_guess_plots.png')

        if show:
            plt.show()