# if window and shift size same

import pandas as pd

def windows(data, window_size=0.5, shift_size=0.5):
    output = {}

    for key, items in data.items():
        print(key)

        rData = {}
        for key1, items in data[key].items():

            windowedData = {}
            for key2, items in data[key][key1].items():

                d = data[key][key1][key2]
                t = d['time']
                dur = t[len(t) - 1] - t[0]
                fs = len(t) / dur

                shift = int(shift_size * fs)
                sizeW = int(window_size * fs)
                nW = round((len(t) - sizeW) / shift)
                nW = int(nW)


                meanRemoved = pd.DataFrame()
                for ii in range(0, nW):
                    w = d[shift * ii:(shift * ii) + sizeW]
                    #w = w.reset_index(drop=True)
                    wMeanRemoved = w - w.mean(axis=0)

                    meanRemoved = meanRemoved.append(wMeanRemoved)
                    meanRemoved['time'] = t


                windowedData[key2] = meanRemoved

            rData[key1] = windowedData

        output[key] = rData

    return output

#####################################


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import stats


def specEntropy(x, fs=20):
    f, Pxx_den = signal.periodogram(x, fs, return_onesided=True, scaling='spectrum')
    # normalise spectral density
    Pxx_denN = Pxx_den / sum(Pxx_den)

    # entropy of normalised spectral density
    ent = stats.entropy(Pxx_denN)

    return ent

def specKurt(x, fs=20):
    f, Pxx_den = signal.periodogram(x, fs, return_onesided=True, scaling='spectrum')
    # normalise spectral density
    Pxx_denN = Pxx_den / sum(Pxx_den)

    # entropy of normalised spectral density

    kurt = stats.kurtosis(Pxx_denN)
    return kurt



#LOAD DATA
data = pd.DataFrame(dds)

#METHOD 1
# REMOVE MEAN WITH 1s WINDOW (20Hz)
rm1 = data.rolling(20,center=True).mean()

#METHOD2
#PAD START AND END OF ROLLING AVERAGE DATA BECAUSE HALF OF WINDOW IS CUT FROM START AND END.
windowSize = 20

pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
rm2=pd.DataFrame()

for ii in range(0, len(data) - (windowSize - 1)):
    d = pd.Series(data[ii:ii+windowSize].mean())
    dd=pd.DataFrame(d)
    rm2 = rm2.append(dd, ignore_index=True)

rm2new = pd.concat([pad1, rm2, pad2], ignore_index=True)

#REMOVE MEAN FROM ORIGINAL SIGNAL
x = data.iloc[:, 0]- rm2new.iloc[:,0]


plt.figure()
plt.plot(rm1,'r')
plt.plot(data,'b')


plt.figure()
plt.plot(x)
plt.plot(data)


windowSize = 40
pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
sd2=pd.DataFrame()
kd2=pd.DataFrame()


for ii in range(0, len(x) - (windowSize - 1)):
    sd = pd.Series(specEntropy(x[ii:ii+windowSize], fs=20))
    sd1 = pd.DataFrame(sd)
    sd2 = sd2.append(sd1, ignore_index=True)

    kd = pd.Series(specKurt(x[ii:ii + windowSize], fs=20))
    kd1 = pd.DataFrame(kd)
    kd2 = sd2.append(kd1, ignore_index=True)

sd2new = pd.concat([pad1, sd2, pad2], ignore_index=True)
kd2new = pd.concat([pad1, kd2, pad2], ignore_index=True)


windowSize = 100
pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
sd2mean=pd.DataFrame()
for ii in range(0, len(sd2) - (windowSize - 1)):
    d = pd.Series(sd2[ii:ii+windowSize].mean())
    dd=pd.DataFrame(d)
    sd2mean = sd2mean.append(dd, ignore_index=True)

sd2meannew = pd.concat([pad1, sd2mean, pad2], ignore_index=True)


def kte(signal):

    d2 = signal[1:-1].reset_index(drop=True)**2

    d3 = signal[:-2].reset_index(drop=True) * signal[2:].reset_index(drop=True)
    kte = d2-d3

    return kte

    from scipy.signal import butter, filtfilt

    nyq = 0.5 * 20
    normal_cutoff = 2 / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    yy = filtfilt(b, a, filtK)


sdKte = kte(sd2meannew)


windowSize = 200
pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
sdKtemean=pd.DataFrame()
for ii in range(0, len(sdKte) - (windowSize - 1)):
    d = pd.Series(sdKte[ii:ii+windowSize].mean())
    dd=pd.DataFrame(d)
    sdKtemean = sdKtemean.append(dd, ignore_index=True)

sdKtemeannew = pd.concat([pad1, sdKtemean, pad2], ignore_index=True)



windowSize = 100
pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
minMax=pd.DataFrame()
for ii in range(0, len(x) - (windowSize - 1)):
    d = pd.Series(np.max(x[ii:ii+windowSize])-np.min(x[ii:ii+windowSize]))
    dd=pd.DataFrame(d)
    minMax = minMax.append(dd, ignore_index=True)

minMaxnew = pd.concat([pad1, minMax, pad2], ignore_index=True)



windowSize = 100
pad1 = pd.DataFrame(np.zeros((int(windowSize/2-1), 1)))
pad2 = pd.DataFrame(np.zeros((int(windowSize/2), 1)))
meanAmp=pd.DataFrame()
for ii in range(0, len(x) - (windowSize - 1)):
    d = pd.Series((x[ii:ii+windowSize]**2).mean())
    dd=pd.DataFrame(d)
    meanAmp = minMax.append(dd, ignore_index=True)

meanAmpnew = pd.concat([pad1, meanAmp, pad2], ignore_index=True)

plt.figure()
plt.plot(x)
plt.plot(meanAmpnew,c='r')
plt.hlines(np.mean(meanAmpnew.iloc[200:-200])-np.std(meanAmpnew.iloc[200:-200]),0,3300)

plt.figure()
plt.plot(x)
plt.plot(minMaxnew,c='r')
plt.hlines(np.mean(minMaxnew.iloc[200:-200]),0,3300)

plt.figure()
plt.subplot(3,1,1)
plt.plot(rm2new.iloc[200:-200],c='r')
plt.plot(x.iloc[200:-200]**2,c='k')


plt.subplot(3,1,2)
plt.plot(sd2meannew.iloc[200:-200],c='r')
plt.hlines(np.mean(sd2meannew.iloc[200:-200])-np.std(sd2meannew.iloc[200:-200]), 200, 3000,colors='r')

plt.subplot(3,1,3)
#plt.plot(sdKte.iloc[100:-100],c='r')
plt.plot(np.absolute(sdKtemeannew.iloc[200:-200]),c='k')
plt.hlines(np.std(np.absolute(sdKtemeannew.iloc[200:-200]))*2, 200, 3000,colors='r')

plt.plot(sdKtemeannew.iloc[200:-200],c='k')
plt.hlines(np.std(sdKtemeannew.iloc[200:-200])*2, 200, 3000,colors='r')



plt.figure()

plt.subplot(2,1,1)
plt.plot(data)
plt.plot(rm2new,c='r')

plt.subplot(2,1,2)
plt.plot(sd2new,c='k')
plt.plot(sd2meannew,c='r')

from butterLowpassFilt import butterLowFilt

#data2 = sd2[100:]
dat = sd2new.loc[:,0]
filtent = butterLowFilt(dat, 1, 20, 4)


plt.figure()

plt.subplot(2,1,1)
plt.plot(x)

plt.subplot(2,1,2)
plt.plot(filtent)


########

import pylab
import numpy as np

def smoothListGaussian(list, degree=5):
    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weightGauss = []

    for i in range(window):
        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return smoothed

 def smoothList(list,strippedXs=False,degree=10):

     if strippedXs==True:
         return Xs[0:-(len(list)-(len(list)-degree+1))]

     smoothed=[0]*(len(list)-degree+1)

     for i in range(len(smoothed)):

         smoothed[i]=sum(list[i:i+degree])/float(degree)

     return smoothed

 def ent(list,degree=10):


     smoothed=[0]*(len(list)-degree+1)

     for i in range(len(smoothed)):

         smoothed[i]=specEntropy(list[i:i+degree])

     return smoothed


z = y.X_value
z = z.astype(float)

zz = np.array(z)

degreeSize = 20

smoothData = smoothListGaussian(zz, degree=degreeSize)
sd = np.array(smoothData)
sd = sd.reshape((1, len(sd)))



smoothList = smoothList(zz,strippedXs=False,degree=degreeSize)
sl = np.array(smoothList)
sl = sl.reshape((1, len(sl)))




padSD = np.full((1, degreeSize), 0)

padSL = np.full((1, int(degreeSize/2)), 0)

padE = np.full((1, int(1/2)), 0)

paddedSD = np.concatenate((padSD, sd, padSD), axis = 1)
paddedSL = np.concatenate((padSL, sl, padSL), axis = 1)

zeroed = zz - paddedSL[0,0:len(zz)]

from matplotlib import pyplot as plt

plt.figure()
plt.subplot(2,1,1)
plt.plot(zz, c='b')
plt.plot(paddedSL[0,:], c='k')
plt.plot(zeroed, c='r')



entropy = ent(zeroed, degree=10)
entData = np.array(entropy)
entData = entData.reshape((1, len(entData)))

paddedE = np.concatenate((padE, entData, padE), axis = 1)
plt.subplot(2,1,2)
plt.plot(paddedE[0,:])
