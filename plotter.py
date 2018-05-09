import numpy as np
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        self.updatedX=[]
        self.measuredX=[]
        self.updatedY=[]
        self.measuredY=[]
        self.times=[]

    def add_updated(self, measured, updated, dt):
        self.updatedX.append(updated[0])
        self.updatedY.append(updated[1])
        self.measuredX.append(measured[0])
        self.measuredY.append(measured[1])
        if len(self.times)>0:
            t=self.times[-1]+dt
        else:
            t=dt
        self.times.append(t)

    def plot(self, adaptive=False, color='red'):
       # plt.clf()
       # plt.close()
        fig1=plt.figure('figure')
        titletxt = "Smoothing with adaptive covariances" if adaptive else "Smoothing with fixed covariances"
        p1=fig1.add_subplot(211)
        plt.title(titletxt)
        #plt.xlabel("time (ms)")
        plt.ylabel("coordinate X (pixel)")

        ux, =plt.plot(self.times, self.updatedX, '0.0', label='Updated')

        if color == 'red':
            mx, =plt.plot(self.times, self.measuredX, 'r*', label='Measured')
        elif color=='green':
            mx, = plt.plot(self.times, self.measuredX, 'g*', label='Measured')
        else:
            mx, = plt.plot(self.times, self.measuredX, 'y*', label='Measured')
        plt.legend(handles=[ux, mx])
        #plt.setp(p1.get_xticklabels(), visible=False)
        p2= fig1.add_subplot(212, sharex=p1)  # creates 2nd subplot with yellow background
        plt.xlabel("time (ms)")

        plt.ylabel("coordinate Y (pixel)")
        uy, =plt.plot(self.times, self.updatedY, '0.0', label='Updated')

        if color == 'red':
            my, =plt.plot(self.times, self.measuredY, 'r*',  label='Measured')
        elif color == 'green':
            my, = plt.plot(self.times, self.measuredY, 'g*', label='Measured')
        else:
            my, = plt.plot(self.times, self.measuredY, 'y*', label='Measured')
        plt.legend(handles=[uy, my])
        plt.show()

    def reset(self):
        self.updatedX = []
        self.measuredX = []
        self.updatedY = []
        self.measuredY = []
        self.times = []
