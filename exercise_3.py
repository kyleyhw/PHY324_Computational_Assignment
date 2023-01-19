import numpy as np
from matplotlib import pyplot as plt
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
    def __init__(self, amplitude, omega):
        self.amplitude = amplitude
        self.omega = omega

    def __call__(self, t):
        return self.amplitude * np.sin(self.omega * t)




class Exercise3():
    def __init__(self):

        with open('noisy_sine_wave', 'rb') as file:
            self.data_from_file = pkl.load(file)

        self.number_of_points = len(self.data_from_file)
        self.spacing = 1

        self.positions = self.data_from_file
        self.times = np.arange(len(self.positions))

        self.amplitudes = np.abs(np.fft.fft(self.positions))
        self.frequencies = np.fft.fftfreq(self.number_of_points, self.spacing)

        self.gaussian_filter_parameters_array = np.array([(0.059, 0.00004), (0.077, 0.00004), (0.14, 0.00004)])
        self.symmetrical_gaussians = np.array([GaussianSymmetricalAboutZero(*params) for params in self.gaussian_filter_parameters_array])

        self.filter_function = Superpose(self.symmetrical_gaussians)

        self.filtered_data = filter(self.frequencies, self.amplitudes, self.filter_function).filtered_data

        self.cleaned_positions = np.fft.ifft(self.filtered_data)

        self.best_guess_parameters_array = np.array([(23.44, 0.059), (50.07, 0.077), (94.84, 0.143)]) * (1, 2*np.pi)# sine wave parameters in form (amplitude, frequency)
        self.best_guess_waves = np.array([Wave(*params) for params in self.best_guess_parameters_array])
        self.best_guess_function = Superpose(self.best_guess_waves)


    def plot(self):
        fig, ( (ax1, ax2), (ax3, ax4), (ax5, ax6) ) = plt.subplots(3, 2, sharex=False, sharey=False)

        ax1.plot(self.times, self.positions) # position-time plot
        ax1.plot(self.times, self.cleaned_positions, color='orange')
        ax1.set_xlim(0, 200)

        ax3.plot(self.times, self.positions - self.cleaned_positions) # residuals
        ax3.set_xlim(0, 200)

        ax5.plot(self.times, self.best_guess_function(self.times)) # best guess
        ax5.set_xlim(0, 200)



        ax2.plot(self.frequencies, self.amplitudes) # amplitude-frequency plot

        ax4.plot(self.frequencies, self.filter_function(self.frequencies)) # filter function plot

        ax6.plot(self.frequencies, self.filtered_data) # filtered amplitude-frequencies plot


        plt.show()