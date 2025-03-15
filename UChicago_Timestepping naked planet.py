import numpy as np
# import matplotlib.pyplot as plt

nStep = int(input(""))
timeStep = 100                            # years
waterDepth = 4000                         # meters
L = 1350                                  # Watts/m2
albedo = 0.3
epsilon = 1
sigma = 5.67E-8                           # W/m2 K4

heatCapacity = waterDepth * 4.2E6         # J/K m2
timeYears = [0]
TK = [0]                                  # Initial temperature in Kelvin

heatContent = heatCapacity * TK[0]        # Initialize heat content
ASR = (1 - albedo) * L / 4                # Absorbed Solar Radiation
OLR = 0

for i in range(0, nStep+1):
    timeYears.append (timeYears[-1] + timeStep)
    OLR = epsilon * sigma * pow(TK[-1], 4) # Outgoing Longwave Radiation
    # print (timeYears[-1], OLR)
    heatContent += (ASR - OLR) * timeStep * 3.14e7 # Update heat content
    TK.append (heatContent / heatCapacity) # Update temperature


print (TK[-2], OLR)