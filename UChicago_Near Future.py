import numpy
import matplotlib.pyplot as plt
import math

# VARIABLES
timeStep    = 1      # years
eqCO2       = 280    # ppm
initCO2     = 290
expGrow     = 0.0225 # for BUSINESS AS USUAL
rampdown    = 0.01   # for WORLD WITHOUT US
RFaerosol_now   = -0.75  # Watts/m2
Temp_2x_CO2 = 3      # degrees/ppm      --> climate sensitivity 1: equilibrium T change for doubling CO2
RF_2x_CO2   = 4      # Watts/m2
Temp_per_RF = Temp_2x_CO2 / RF_2x_CO2
#print (Temp_per_RF) # degrees*m2/Watts --> climate sensitivity 2: equilibrium T change per W/m2 RF
Tresponse   = 20     # what does this mean?

# BUSINESS AS USUAL
timeYears    = [1900]
bauCO2       = [initCO2]
rfCO2        = [0]
eqTemp       = [0]
transTemp    = [0]
timeTemp     = [0]
mask         = [0]
rfMask       = []
rfTotal      = []

# WORLD WITHOUT US
wwuCO2       = [0]
rf_wwu       = [0]
eqTemp_wwu   = [0]
transTemp_wwu= [0]

# INPUT FOR CODE CHECK
# nYear = int(input("").strip())

# BUSINESS AS USUAL
while timeYears[-1] < 2100:
    timeYears.append (timeYears[-1] + timeStep)
    bauCO2.append (eqCO2 + (bauCO2[-1] - eqCO2) * (1 + expGrow * timeStep))
    rfCO2.append (RF_2x_CO2 * math.log(bauCO2[-1] / eqCO2) / math.log(2))
    if len(bauCO2) > 1:
        mask.append ((bauCO2[-1] - bauCO2[-2]) / timeStep)
# print(timeYears)

# iYear = timeYears.index(nYear)

i2015 = timeYears.index(2015) # find the value of masking in 2015 using the index
aerosolCo = RFaerosol_now / ((bauCO2[i2015] - bauCO2[i2015-1]) / timeStep)
#aerosolCo = -0.3
#print (aerosolCo)
#print (bauCO2[i2015])
#print (bauCO2[i2015-1])

transTemp = [0]
for i in range (0, len(timeYears)-1):  # the number of years from 1900 - 2100 can be calculated using this syntax
    rfMask.append (max(mask[i-1]*aerosolCo, RFaerosol_now))
    rfTotal.append (rfCO2[i-1] + rfMask[i-1])
    eqTemp.append (rfTotal[i] * Temp_per_RF)         # bug
    transTemp.append (transTemp[-1] + (eqTemp[i] - transTemp[-1]) / Tresponse * timeStep) # bug

# WORLD WITHOUT US
wwuCO2       = [0]
rf_wwu       = [0]
eqTemp_wwu   = [0]
transTemp_wwu= [0]

for i in range (1, i2015):
    wwuCO2.append (bauCO2[i])
    rf_wwu.append (rfTotal[i])
    eqTemp_wwu.append (eqTemp[i])
    transTemp_wwu.append (transTemp[i])


for i in range (i2015, len(timeYears)):
    wwuCO2.append (wwuCO2[-1] + (340 - wwuCO2[-1]) * (rampdown * timeStep))
    rf_wwu.append (RF_2x_CO2 * math.log( wwuCO2[i] / eqCO2 ) / math.log(2))
    eqTemp_wwu.append (rf_wwu[i] * Temp_per_RF)
    transTemp_wwu.append (transTemp_wwu[-1] + (eqTemp_wwu[i] - transTemp_wwu[-1]) * timeStep / Tresponse)



# OUTPUT FOR CODE CHECK
# print(bauCO2[iYear], rfCO2[iYear], eqTemp[iYear],transTemp[iYear])


# TIME - CO2 diagram
#plt.plot (timeYears, bauCO2, '-', color = 'lightblue', label = 'Business as usual')
#plt.plot (timeYears, wwuCO2, '-', color = 'violet', label = 'World without us')
#plt.xlabel ('Years')
#plt.ylabel ('CO2 (ppm)')
#plt.legend ()
#plt.show ()

# TIME - Temperature diagram
plt.plot (timeYears, transTemp, '-', color = 'lightblue', label = 'Business as usual')
plt.plot (timeYears, transTemp_wwu, '-', color = 'violet', label = 'World without us')
plt.xlabel ('Years')
plt.ylabel ('Temperature (degrees)')
plt.legend ()
plt.show ()
