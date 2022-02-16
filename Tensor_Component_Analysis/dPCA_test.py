from numpy import *
from numpy.random import rand, randn, randint
import matplotlib.pyplot as plt
from dPCA import dPCA


# number of neurons, time-points and stimuli
N,T,S = 100,250,6

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zt = (arange(T)/float(T))
zs = (arange(S)/float(S))

# build trial-by trial data
trialR = noise*randn(n_samples,N,S,T)
trialR += randn(N)[None,:,None,None]*zt[None,None,None,:]
trialR += randn(N)[None,:,None,None]*zs[None,None,:,None]

# trial-average data
R = mean(trialR,0)

# center data
R -= mean(R.reshape((N,-1)),1)[:,None,None]

dpca = dPCA.dPCA(labels='st',regularizer='auto')
dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

time = arange(T)


figure_1 = plt.figure(figsize=(16, 7))
rows = 1
columns = 3

axis_1 = figure_1.add_subplot(rows, columns, 1)
axis_2 = figure_1.add_subplot(rows, columns, 2)
axis_3 = figure_1.add_subplot(rows, columns, 3)

axis_1.set_title('1st time component')
axis_2.set_title('1st stimulus component')
axis_3.set_title('1st mixing component')

for s in range(S):
    axis_1.plot(time, Z['t'][0, s])

for s in range(S):
    axis_2.plot(time, Z['s'][0, s])

for s in range(S):
    axis_3.plot(time, Z['st'][0, s])

plt.show()