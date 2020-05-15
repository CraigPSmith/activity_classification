from signalProcessing import signalProcess as SP

#########
from readData import readRawData
raw = readRawData()

for key, items in raw.items():
    print(key)

    rData = {}
    for key1, items in raw[key].items():

        datafilt = {}
        for key2, items in raw[key][key1].items():
            d = raw[key][key1][key2]

#########



x = SP(d).filterLp(cutoff=2)
y = SP(x).downsample()
z = SP(y).movingWindow(period=20, operation='average')
zz = SP(x).movingWindow(period=20, operation='removeAverage')

len(z)





from matplotlib import pyplot as plt

plt.figure('filtered')
plt.plot(x.X_value,c='b')
plt.figure('downsampled')
plt.plot(y.X_value,c='b')
plt.plot(yy.X_value,c='r')
plt.figure('kte')
plt.plot(z,c='b')


import numpy as np
import matplotlib.pyplot as plt

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

# np.fft.fft
freqs = np.fft.fftfreq(time.size, 1/fs)
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(x))**2
plt.figure()
plt.plot(freqs[idx], ps[idx])
plt.title('Power spectrum (np.fft.fft)')


fs = 20
t = np.array(yy.time)
x = np.array(yy.X_value)
t = t[85:500] - t[85]
x = x[85:500]

plt.figure()
plt.plot(x)

# np.fft.fft
freqs = np.fft.fftfreq(t.size, 1/fs)
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(x))**2
plt.figure()
plt.plot(freqs[idx], ps[idx])
plt.title('Power spectrum (np.fft.fft)')

from scipy import signal

# signal.welch
f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()


fs = 20
t = np.array(yy.time)
x = np.array(yy.X_value)
t = t[2300:2600] - t[2300]
x = x[2300:2600]

plt.figure()
plt.plot(x)


from scipy import signal
from scipy import stats

# spectral density
f, Pxx_den = signal.periodogram(x, fs, return_onesided=True, scaling='spectrum')
plt.figure()
plt.plot(f, Pxx_den)

#normalise spectral density
Pxx_denN = Pxx_den/sum(Pxx_den)

#entropy of normalised spectral density
ent = stats.entropy(Pxx_denN)
print(ent)

