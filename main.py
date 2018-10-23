from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
import math as mt
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special

def integralTable(func,arr):
    result = integrate.quad(func, -np.pi, np.pi)
    arr.append(result[0])
    return arr

def plotTable(frame,fig,loc):
    ax = fig.add_subplot(4,1,loc)
    ax.table(cellText=frame.values, colLabels=frame.columns, loc='center')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return frame

def buildCosSquaredFrame(fig,r_begin,r_end):
    dataFrame = pd.DataFrame(columns=list('12345'))
    for m in range(r_begin,r_end):
        arr = []
        for n in range(r_begin,r_end):
            cos_squared = lambda x: ((np.cos(m*x)*np.cos(n*x))/np.pi)
            arr = integralTable(cos_squared,arr)
        innerFrame = pd.DataFrame([arr], columns=list('12345'))
        row = innerFrame.xs(0)
        row.name = str(m)
        dataFrame = dataFrame.append(row)
    
    return plotTable(dataFrame,fig,1)


def buildFourierOddFrame(fig,r_begin,r_end):
    dataFrame = pd.DataFrame()
    arr = []
    for m in range(r_begin,r_end):
        fourier_sin = lambda x: ((np.sin(m*x)*(x**2))/np.sqrt(np.pi))
        arr = integralTable(fourier_sin,arr)    
    innerFrame = pd.DataFrame([arr])
    row = innerFrame.xs(0)
    row.name = str(m)
    dataFrame = dataFrame.append(row)
    plotTable(dataFrame,fig,2)
    return row

def buildFourierEvenFrame(fig,r_begin,r_end):
    dataFrame = pd.DataFrame()
    arr = []
    for m in range(r_begin,r_end):
        fourier_cos = lambda x: ((np.cos(m*x)*(x**2))/np.sqrt(np.pi))
        arr = integralTable(fourier_cos,arr)
    innerFrame = pd.DataFrame([arr])
    row = innerFrame.xs(0)
    row.name = str(m)
    dataFrame = dataFrame.append(row)
    plotTable(dataFrame,fig,3)
    return row


def make_plot():
    with PdfPages('cos_fourier.pdf') as pdf:

        # X-Squared Function
        x2 = np.arange((-np.pi), np.pi,0.01)
        y2 = (x2)**2



        # Creating PDF document with plots

        # Build First Table for CosSquared
        mainfig = plt.figure(figsize=(12,16))
        buildCosSquaredFrame(mainfig,1,6)

        odd_co = buildFourierOddFrame(mainfig,1,20)
        even_co = buildFourierEvenFrame(mainfig,1,20)
        print odd_co
        print even_co
        # Cosine Function
        x = np.arange((-np.pi), np.pi, 0.01) #start, stop, step
        y = []
        for el in x:
            tmp=0
            for j in range(len(even_co)):
                tmp += -(even_co[j]*np.cos(j*el)) + (odd_co[j]*np.sin(j*el))
            y.append(tmp - 4.84)
        print len(x), len(y)
        print y[len(y)/2]
        #Graph our fourier series versus actual function
        mainfig.add_subplot(4,1,4)
        plot(x,y)
        plot(x2,y2)
        plt.close()
        show(block=False)
        pdf.savefig(mainfig)
        print('continue computation')

# Now display plot in a window
make_plot()
