import math
import matplotlib

# -------------------------------
# VARIABLES Initialization 1
# -------------------------------

A = 204
B = 2.17
C = 3.87
Aice = 0.62
Acloud = 0.3
Tcrit = -10
SCfrac = 1
SC = 1370
iterations = 30
Zones = ["80-90", "70-80", "60-70", "50-60", "40-50", "30-40", "0-30", "0-20", "0-10"]
ZoneLat = [85, 75, 65, 55, 45, 35, 25, 15, 6]
ZoneSun = [0.5, 0.531, 0.624, 0.77, 0.892, 1.021, 1.12, 1.189, 1.219]
Init_T = [-16.9, -12.3, -5.1, 2.2, 8.8, 16.2, 22.9, 26.1, 26.4]




# -------------------------------
# VARIABLES Initialization 2
# -------------------------------

# Calculate initial albedo according to initial temperature
Init_a = []
for lat in range(9):
    if Init_T[lat] < Tcrit:
        Init_a.append(Aice)
    else:
        Init_a.append(Acloud)

# Calculate ratio correction / weighting
sumcos = 0
coz = []
for lat in range(9):
    coz_val = math.cos(ZoneLat[lat] * math.pi / 180)
    coz.append(coz_val)
    sumcos += coz_val

# Calculate incoming radiation
R_in = []
for lat in range(9):
    R_in.append(SC / 4 * SCfrac * ZoneSun[lat])




# -------------------------------
# FIRST STEP
# -------------------------------

# Calculate Mean Global Temperature
Tcos = []
for lat in range(9):
    Tcos.append(Init_T[lat] * coz[lat])

Mean_T = [0]
for lat in range(9):
    Mean_T[0] += Tcos[lat]
Mean_T[0] /= sumcos

# Temperature and Albedo for Each Latitude Zone
Temp = []
Albedo = []
for lat in range(9):
    temp_val = (R_in[lat] * (1 - Init_a[lat]) + C * Mean_T[0] - A) / (B + C)
    Temp.append(temp_val)
    if temp_val < Tcrit:
        Albedo.append(Aice)
    else:
        Albedo.append(Acloud)





# -------------------------------
# INTERMEDIATE STEPS
# -------------------------------

for step in range(1, iterations + 1): #这里是不是已经把最后一步涵盖了？

    # Calculate Mean Global Temperature
    for lat in range(9):
        Tcos[lat] = Temp[lat] * coz[lat]

    Mean_T.append(0)
    for lat in range(9):
        Mean_T[step] += Tcos[lat]
    Mean_T[step] /= sumcos

    # Temperature and Albedo for Each Latitude Zone
    for lat in range(9):
        Temp[lat] = (R_in[lat] * (1 - Albedo[lat]) + C * Mean_T[step] - A) / (B + C)
        if Temp[lat] < Tcrit:
            Albedo[lat] = Aice
        else:
            Albedo[lat] = Acloud



# FINAL STEP and OUTPUT
GMT = Mean_T[iterations]
Final_T = []
Final_a = []
R_out = []
for lat in range(9):
    final_t_val = (R_in[lat] * (1 - Init_a[lat]) + C * GMT - A) / (B + C)
    Final_T.append(final_t_val)
    if final_t_val < Tcrit:
        Final_a.append(Aice)
    else:
        Final_a.append(Acloud)
    R_out.append(A + B * final_t_val)
    print(f"{ZoneLat[lat]}, {Final_a[lat]}, {R_in[lat]}, {R_out[lat]}, {Final_T[lat]}")