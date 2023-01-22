# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:53:25 2017

@author: Brian
"""

save=True # if True then we save images as files

from random import gauss
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np

N=200   # N is how many data points we will have in our sine wave

time=np.arange(N)

A1=5.   # wave amplitude
T1=17.  # wave period
y1=A1*np.sin(2.*np.pi*time/T1)

A2=9.
T2=13.
y2=A2*np.sin(2.*np.pi*time/T2)

y=y1#+y2

noise_amp=A1/2. 
# set the amplitude of the noise relative to sine's amp

"""
i=0
noise=[]
while i < N:
    noise.append(gauss(0,noise_amp))
    i+=1
"""
noise=[gauss(0,noise_amp) for _usused_variable in range(len(y))]
# this line, and the commented block above, do exactly the same thing

x=y+noise
#x=y
# y is our pure sine wave, x is y with noise added

z1=np.fft.fft(y)
z2=np.fft.fft(x)
# take the Fast Fourier Transforms of both x and y

freqs = np.fft.fftfreq(N, 1/N) * 2 * np.pi # in rad/s

fig, ( (ax1,ax2), (ax3,ax4) ) = plt.subplots(2,2,sharex='col',sharey='col', figsize=(32,18))
""" 
this setups up a 2x2 array of graphs, based on the first two arguments
of plt.subplots()

the sharex and sharey force the x- and y-axes to be the same for each 
column
"""

ax1.plot(time/N,y)
ax1.set_ylabel('position $x / m$')
ax1.set_title('Pure sine wave')
ax1.set_xlabel('time $t / s$')


ax2.plot(freqs, np.abs(z1))
ax2.set_ylabel('amplitude $A / m$')
ax2.set_title('FFT of pure sine wave')
ax2.set_xlabel('frequency f / Hz')

ax3.plot(time/N,x)
ax3.set_ylabel('position $x / m$')
ax3.set_title('Same wave with noise')
ax3.set_xlabel('time $t / s$')

ax4.plot(freqs, np.abs(z2))
ax4.set_ylabel('amplitude $A / m$')
ax4.set_xlabel('frequency / Hz')
ax4.set_title('FFT of noisy wave')
""" 
our graphs are now plotted

(ax1,ax2) is a list of figures which are the top row of figures

therefore ax1 is top-left and ax2 is top-right

we plot the position-time graphs rescaled by a factor of N so that
the FFT x-axis agrees with the frequency we could measure from the
position-time graph. by default, both graphs use "data-point number"
on their x-axes, so would go 0 to 200 since N=200.
"""

# fig.subplots_adjust(hspace=0)
# remove the horizontal space between the top and bottom row

ax3.set_ylim(-13,13)
ax4.set_ylim(0,480)


mydpi=300
# plt.tight_layout()

if (save):
    plt.savefig('exercise_2_position_time_plots.png', figsize = (32,18))
plt.show()
"""
plt.show() displays the graph on your computer

plt.savefig will save the graph as a .png file, useful for including
in your report so you don'times have to cut-and-paste
"""


M=len(z2)
freq=np.arange(M)  # frequency values, like time is the time values
width=1 # 8  # width=2*sigma**2 where sigma is the standard deviation
peak=N/T1 # 12.3    # ideal value is approximately N/T1

print(peak)

filter_function=(np.exp(-(freq-peak)**2/width)+np.exp(-(freq+peak-M)**2/width))
z_filtered=z2*filter_function
"""
we choose Gaussian filter functions, fairly wide, with
one peak per spike in our FFT graph

we eyeballed the FFT graph to figure out decent values of 
peak and width for our filter function

a larger width value is more forgiving if your peak value
is slightly off

making width a smaller value, and fixing the value of peak,
will give us a better final result
"""



fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col', figsize=(32,18))
# this gives us an array of 3 graphs, vertically aligned
ax1.plot(freqs, np.abs(z2))
ax2.plot(freqs, np.abs(filter_function))
ax3.plot(freqs, np.abs(z_filtered))
"""
note that in general, the fft is a complex function, hence we plot
the absolute value of it. in our case, the fft is real, but the
result is both positive and negative, and the absolute value is still
easier to understand

if we plotted (abs(fft))**2, that would be called the power spectra
"""

# fig.subplots_adjust(hspace=0)
ax1.set_ylim(0,480)
ax2.set_ylim(0,1.2)
ax3.set_ylim(0,480)
ax1.set_title('Noisy FFT')
ax1.set_ylabel('amplitude $A / m$')
ax1.set_xlabel('frequency / Hz')

ax2.set_title('Filter Function F')
ax2.set_ylabel('F($\omega$)')
ax2.set_xlabel('frequency / Hz')

ax3.set_title('Filtered FFT')
ax3.set_ylabel('amplitude $A / m$')
ax3.set_xlabel('frequency / Hz')

# plt.tight_layout()
""" 
the \n in our xlabel does not save to file well without the
tight_layout() command
"""

if(save):
    plt.savefig('exercise_2_frequency_domain_plots.png', figsize=(32,18))
plt.show()

cleaned=np.real(np.fft.ifft(z_filtered))
"""
ifft is the inverse FFT algorithm

it converts an fft graph back into a sinusoidal graph

we took the data, took the fft, used a filter function 
to eliminate most of the noise, then took the inverse fft
to get our "cleaned" version of the original data
"""

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col',figsize=(32,18))
ax1.plot(time/N,x)
ax2.plot(time/N,cleaned)
ax3.plot(time/N,y)
"""
we plot the real part of our cleaned data - but since the 
original data was real, the result of our tinkering should 
be real so we don'times lose anything by doing this

if you don'times explicitly plot the real part, python will 
do it anyway and give you a warning message about only
plotting the real part of a complex number. so really, 
it's just getting rid of a pesky warning message
"""

# fig.subplots_adjust(hspace=0)
ax1.set_ylim(-13,13)
ax1.set_ylabel('position $x / m$')
ax1.set_title('Original Data')
ax1.set_xlabel('time $t / s$')

ax2.set_title('Filtered Data')
ax2.set_ylabel('position $x / m$')
ax2.set_xlabel('time $t / s$')


ax3.set_title('Ideal wave')
ax3.set_ylabel('position $x / m$')
ax3.set_xlabel('time $t / s$')


if(save):
    plt.savefig('exercise_2_time_domain_plots.png', figsize=(32,18))
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1,sharex='col',sharey='col',figsize=(32,18))

ax1.plot(time / N, x - cleaned)
ax1.set_title('Residual between noisy wave and cleaned wave')
ax1.set_ylabel('noisy generated position - \n cleaned position')
ax1.set_xlabel('time $t / s$')

ax2.plot(time / N, y - cleaned)
ax2.set_title('Residual between ideal wave and cleaned wave')
ax2.set_ylabel('ideal position - \n cleaned position')
ax2.set_xlabel('time $t / s$')

if(save):
    plt.savefig('exercise_2_residual_plots.png', figsize=(32,18))
plt.show()