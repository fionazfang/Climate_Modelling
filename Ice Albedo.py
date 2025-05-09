import numpy as np
import matplotlib.pyplot as plt

niters = 100
timeStep = 100                            # years
waterDepth = 4000                         # meters
epsilon = 1
sigma = 5.67E-8

LRange = [1200, 1600]                     # Watts/m2
L = LRange [1]                            # starts from the greatest value
albedo = 0.15

plotType = 'iterDown','iterUp','iter','L', 'lat', 'albedo'
plotType = input('')
y = []
x = []

while L > LRange [0]-1:
    for iter in range (0, niters):
        T = pow(((1- albedo) * L / (4 * epsilon * sigma)), 0.25)

        # albedo
        albedo = -0.01 * T + 2.8
        if 0.65 >= albedo and albedo >= 0.15:
            albedo = albedo
        elif 0.65 < albedo:
            albedo = 0.65
        else:
            albedo = 0.15

        # latitude
        lat = 1.5 * T - 322.5
        if 90 >= lat and lat >= 0:
            lat = lat
        elif 90 < lat:
            lat = 90
        else:
            lat = 0

        # plot
        if plotType == 'iter' or plotType == 'iterDown':
            x.append (iter)
            y.append (T)
    if plotType == 'iter' or plotType == 'iterDown':
        x.append (numpy.nan)
        y.append (numpy.nan)
    if plotType == 'L':
        x.append (L)
        y.append (T)
    if plotType == 'albedo':
        x.append (albedo)
        y.append (T)
    if plotType == 'lat':
        x.append (L)
        y.append (lat)
    L = L - 10

while L < LRange [1]+1:
    for iter in range (0, niters):
        T = pow(((1- albedo) * L / (4 * epsilon * sigma)), 0.25)

        # albedo
        albedo = -0.01 * T + 2.8
        if 0.65 >= albedo and albedo >= 0.15:
            albedo = albedo
        elif 0.65 < albedo:
            albedo = 0.65
        else:
            albedo = 0.15

        # latitude
        lat = 1.5 * T - 322.5
        if 90 >= lat and lat >= 0:
            lat = lat
        elif 90 < lat:
            lat = 90
        else:
            lat = 0

        # plot
        if plotType == 'iter' or plotType == 'iterUp':
            x.append (iter)
            y.append (T)
    if plotType == 'iter' or plotType == 'iterUp':
        x.append (numpy.nan)
        y.append (numpy.nan)
    if plotType == 'L':
        x.append (L)
        y.append (T)
    if plotType == 'albedo':
        x.append (albedo)
        y.append (T)
    if plotType == 'lat':
        x.append (L)
        y.append (lat)
    L = L + 10


plt.plot(x,y)
plt.show
