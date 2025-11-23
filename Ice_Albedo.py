import numpy as np
import matplotlib.pyplot as plt

# Settings
niters = 100
dt_years = 100
epsilon = 1.0
sigma = 5.67e-8

waterDepth = 4000          
L_range = [1200, 1600]      # W/m2
L = L_range[1]              # start from high end

albedo = 0.15
plotType = input("Plot type (iterDown, iterUp, iter, L, lat, albedo): ").strip()

x = []
y = []

# Helpers
def update_T(L, albedo):
    return ((1 - albedo) * L / (4 * epsilon * sigma)) ** 0.25

def update_albedo(T):
    alb = -0.01 * T + 2.8
    return np.clip(alb, 0.15, 0.65)

def update_latitude(T):
    lat = 1.5 * T - 322.5
    return np.clip(lat, 0, 90)


while L >= L_range[0]:

    for i in range(niters):
        T = update_T(L, albedo)
        albedo = update_albedo(T)
        lat = update_latitude(T)

        if plotType in ["iter", "iterDown"]:
            x.append(i)
            y.append(T)

    if plotType in ["iter", "iterDown"]:
        x.append(np.nan)
        y.append(np.nan)

    if plotType == "L":
        x.append(L)
        y.append(T)
    elif plotType == "albedo":
        x.append(albedo)
        y.append(T)
    elif plotType == "lat":
        x.append(L)
        y.append(lat)

    L -= 10


while L <= L_range[1]:

    for i in range(niters):
        T = update_T(L, albedo)
        albedo = update_albedo(T)
        lat = update_latitude(T)

        if plotType in ["iter", "iterUp"]:
            x.append(i)
            y.append(T)

    if plotType in ["iter", "iterUp"]:
        x.append(np.nan)
        y.append(np.nan)

    if plotType == "L":
        x.append(L)
        y.append(T)
    elif plotType == "albedo":
        x.append(albedo)
        y.append(T)
    elif plotType == "lat":
        x.append(L)
        y.append(lat)

    L += 10


plt.plot(x, y)
plt.xlabel(plotType)
plt.ylabel("Temperature (K)")
plt.title("Zero-Dimensional Climate Model")
plt.show()
