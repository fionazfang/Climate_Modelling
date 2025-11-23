import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import math

plotOutput = True

# Grid and Variable Initialization

ncol = 7  # grid size (number of cells)
nrow = ncol

#nSlices, iRowOut, iColOut = input("").split()
#nSlices, iRowOut, iColOut = [int(nSlices), int(iRowOut), int(iColOut)]

ntAnim  = 1  # number of time steps for each frame
nSlices = 100



horizontalWrap = True  # determines whether the flow wraps around, connecting
# the left and right-hand sides of the grid, or whether
# there's a wall there.
interpolateRotation = True
rotationScheme = "PlusMinus"  # "WithLatitude", "PlusMinus", "Uniform"

# Note: the rotation rate gradient is more intense than the real world,
# so that the model can equilibrate quickly.

windScheme = ""  # "Curled", "Uniform"
initialPerturbation = "Tower"  # "Tower", "NSGradient", "EWGradient"
textOutput = True
plotOutput = True
arrowScale = 30

dT = 200            # seconds
G = 9.8e-4          # m/s2
HBackground = 4000  # meters

dX = 10.E3          # meters -> a small ocean on a small, low-G planet

dxDegrees = dX / 110.e3
flowConst = G       # 1/s2
dragConst = 1.E-6   # about 10 days decay time
meanLatitude = 30   # degrees

latitude = []
rotConst = []
windU = []
for irow in range(0, nrow):
    if rotationScheme == "WithLatitude":
        latitude.append(meanLatitude + (irow - nrow / 2) * dxDegrees)
        rotConst.append(-7.e-5 * math.sin(math.radians(latitude[-1])))  # s-1
    elif rotationScheme == "PlusMinus":
        rotConst.append(-3.5e-5 * (1. - 0.8 * (irow - (nrow - 1) / 2) / nrow))  # rot 50% +-
    elif rotationScheme == "Uniform":
        rotConst.append(-3.5e-5)
    else:
        rotConst.append(0)
    if windScheme == "Curled":
        windU.append(1e-8 * math.sin((irow + 0.5) / nrow * 2 * 3.14))
    elif windScheme == "Uniform":
        windU.append(1.e-8)
    else:
        windU.append(0)
itGlobal = 0

U = numpy.zeros((nrow, ncol + 1))
V = numpy.zeros((nrow + 1, ncol))
H = numpy.zeros((nrow, ncol + 1))  # for ghost cells...
dUdT = numpy.zeros((nrow, ncol))
dVdT = numpy.zeros((nrow, ncol))
dHdT = numpy.zeros((nrow, ncol))
dHdX = numpy.zeros((nrow, ncol + 1))
dHdY = numpy.zeros((nrow, ncol))
dUdX = numpy.zeros((nrow, ncol))
dVdY = numpy.zeros((nrow, ncol))
rotV = numpy.zeros((nrow, ncol))  # interpolated to u locations
rotU = numpy.zeros((nrow, ncol))  # to v

midCell = int(ncol / 2)
if initialPerturbation == "Tower":
    H[midCell, midCell] = 1
elif initialPerturbation == "NSGradient":
    H[0:midCell, :] = 0.1
elif initialPerturbation == "EWGradient":
    H[:, 0:midCell] = 0.1


def animStep():
    global stepDump, itGlobal

    # Time Loop
    for it in range(0, ntAnim):

        # Longitudinal Derivatives (dHdX, dUdX)
        for irow in range(0, nrow):
            for icol in range(0, ncol + 1):
                if horizontalWrap:
                    if icol == 0 or icol == ncol:
                        dHdX[irow, icol] = (H[irow, icol] - H[irow, -1]) / dX
                    else:
                        dHdX[irow, icol] = (H[irow, icol] - H[irow, icol - 1]) / dX
                else:
                    if icol == 0 or icol == ncol + 1:  # Boundary conditions
                        dHdX[irow, icol] = 0
                    else:
                        dHdX[irow, icol] = (H[irow, icol] - H[irow, icol - 1]) / dX

        for irow in range(0, nrow):
            for icol in range(0, ncol):
                if horizontalWrap:
                    if icol == ncol - 1:
                        dUdX[irow, icol] = (U[irow, icol] - U[irow, -1]) / dX
                    else:
                        dUdX[irow, icol] = (U[irow, icol + 1] - U[irow, icol]) / dX
                else:
                    if icol == 0 or icol == ncol:  # Boundary conditions
                        dUdX[irow, icol] = 0
                    else:
                        dUdX[irow, icol] = (U[irow, icol + 1] - U[irow, icol]) / dX

        # Latitudinal Derivatives (dHdY, dVdY)
        for irow, icol in numpy.ndindex(nrow, ncol):
            if irow == 0:
                dHdY[irow, icol] = (H[irow, icol] - 0) / dX
            else:
                dHdY[irow, icol] = (H[irow, icol] - H[irow - 1, icol]) / dX

        for irow, icol in numpy.ndindex(nrow, ncol):
            if irow == 0:  # Boundary conditions
                dVdY[irow, icol] = (V[irow + 1, icol] - 0) / dX
            elif irow == nrow -1:
                dVdY[irow, icol] = (0 - V[irow, icol]) / dX
            else:
                dVdY[irow, icol] = (V[irow + 1, icol] - V[irow, icol]) / dX

        # Rotational Terms
        for irow, icol in numpy.ndindex(nrow, ncol):
            rotV[irow, icol] = rotConst[irow] * V[irow, icol]
            rotU[irow, icol] = -rotConst[irow] * U[irow, icol]

        # Encode Latitudinal Derivatives Here
        for irow, icol in numpy.ndindex(nrow, ncol):
            dUdT[irow, icol] = rotV[irow, icol] - flowConst * dHdX[irow, icol] - dragConst * U[irow, icol] + windU[irow]
            dVdT[irow, icol] = - rotU[irow, icol] - flowConst * dHdY[irow, icol] - dragConst * V[irow, icol]
            dHdT[irow, icol] = - (dUdX[irow, icol] + dVdY[irow, icol]) * HBackground 

        # Step Forward One Time Step
        for irow, icol in numpy.ndindex(nrow, ncol):
            U[irow, icol] += dUdT[irow, icol] * dT
            V[irow, icol] += dVdT[irow, icol] * dT
            H[irow, icol] += dHdT[irow, icol] * dT

        # Update the Boundary and Ghost Cells
        if horizontalWrap == True:
            for irow in range(nrow):
                U[irow, ncol] = U[irow, 0]
                H[irow, ncol] = H[irow, 0]
        else:
            U[:, 0] = 0
            U[:, ncol - 1] = 0

        #print("dUdX:\n", dUdX)
        #print("dVdY:\n", dVdY)
        #print("dHdX:\n", dHdX)
        #print("dHdY:\n", dHdY)
        #print("rotU:\n", rotU)
        #print("rotV:\n", rotV)
        #print("dUdT:\n", dUdT)
        #print("dVdT:\n", dVdT)
        #print("dHdT:\n", dHdT)
        #print("U:\n", U)
        #print("V:\n", V)
        #print("H:\n", H)


    itGlobal = itGlobal + ntAnim


for i_anim_step in range(0, nSlices):
    animStep()

#print(H[iRowOut, iColOut], dHdT[iRowOut, iColOut], U[iRowOut, iColOut], V[iRowOut, iColOut], rotU[iRowOut, iColOut])


def firstFrame():
    global fig, ax, hPlot
    fig, ax = plt.subplots()
    ax.set_title("H")
    hh = H[:,0:ncol]
    loc = tkr.IndexLocator(base=1, offset=1)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    grid = ax.grid(which='major', axis='both', linestyle='-')
    hPlot = ax.imshow(hh, interpolation='nearest', clim=(-0.5,0.5))
    plotArrows()
    plt.show(block=False)

def plotArrows():
    global quiv, quiv2
    xx = []
    yy = []
    uu = []
    vv = []
    for irow in range( 0, nrow ):
        for icol in range( 0, ncol ):
            xx.append(icol - 0.5)
            yy.append(irow )
            uu.append( U[irow,icol] * arrowScale )
            vv.append( 0 )
    quiv = ax.quiver( xx, yy, uu, vv, color='white', scale=1)
    for irow in range( 0, nrow ):
        for icol in range( 0, ncol ):
            xx.append(icol)
            yy.append(irow - 0.5)
            uu.append( 0 )
            vv.append( -V[irow,icol] * arrowScale )
    quiv2 = ax.quiver( xx, yy, uu, vv, color='white', scale=1)

def updateFrame():
    global fig, ax, hPlot, quiv, quiv2
    hh = H[:,0:ncol]
    hPlot.set_array(hh)
    quiv.remove()
    quiv2.remove()
    plotArrows()
    plt.pause(0.00001)
    fig.canvas.draw()
    print("Time: ", math.floor( itGlobal * dT / 86400.*10)/10, "days")

def textDump():
    print("time step ", itGlobal)
    print("H", H)
    print("dHdX" )
    print( dHdX)
    print("dHdY" )
    print( dHdY)
    print("U" )
    print( U)
    print("dUdX" )
    print( dUdX)
    print("rotV" )
    print( rotV)
    print("V" )
    print( V)
    print("dVdY" )
    print( dVdY)
    print("rotU" )
    print( rotU)
    print("dHdT" )
    print( dHdT)
    print("dUdT" )
    print( dUdT)
    print("dVdT" )
    print( dVdT)

if textOutput is True:
    textDump()
if plotOutput is True:
    firstFrame()
for i_anim_step in range(0,nSlices):
    animStep()
    if textOutput is True:
        textDump()
    if plotOutput is True:
        updateFrame()