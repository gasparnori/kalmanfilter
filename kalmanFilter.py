import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import sys, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from filter.kf import Filter
import math
import random

num_variables = 6
missing_num = 5

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

class Feature():
    def __init__(self, color, offset):
        self.color=color
        self.offset=offset
        self.plotter = Plotter()
        self.reset()

    def reset(self):
        #to draw
        #self.points_final = []
        #self.points_online = []
        self.points_measured = []

        self.kf=Filter(missing_num, num_variables)

    def createNoisyPoint(self, x, y, randomNoise):
        mu, sigma = 0, 10  # mean and standard deviation
        if randomNoise:
            randCheck = random.randint(0, 100) % 100
            if randCheck == 0:
                datax = random.randint(0, 300)
                datay = random.randint(0, 300)
            else:
                datax = int(x + self.offset + np.random.normal(mu, sigma, 1))
                datay = int(y + self.offset + np.random.normal(mu, sigma, 1))
        else:
            datax = int(x + self.offset + np.random.normal(mu, sigma, 1))
            datay = int(y + self.offset + np.random.normal(mu, sigma, 1))
        return [datax, datay]

    def addPoints(self, x, y, t, adapt, predict, randomNoise):

        if x is None:
            self.kf.measured_state=None
            p = self.kf.iterate_filter(t, adapt, predict)
            if p is not None:
                return (p[0], p[1], t)
            else:
                return None
        else:
            [datax, datay]=self.createNoisyPoint(x,y,randomNoise)
            self.points_measured.append([datax, datay])
            self.kf.add_measurement(datax, datay, t)
            p=self.kf.iterate_filter(t, adapt, predict)
            if p is not None:
                return (p[0], p[1], t)
            else:
                return None

class Object():
    def __init__(self, red, green):
        self.red=red
        self.green=green
        self.plotter=Plotter()
        self.reset()

    def reset(self):
        self.points_measured = []
        self.kf = Filter(missing_num, num_variables)

    def add_points(self, red, green, t, adapt, predict):
        # if at least one of the the data is missing
        if (red[0]==0 and red [1]==0) or (green[0]==0 and green[1]==0):
            p = self.kf.iterate_filter(t, adapt, predict)
            #p=None
            if p is not None:
                return (p[0], p[1], t)
            else:
                return None

        else:
            datax=(red[0][0,0]+green[0][0,0])/2.0
            datay=(red[1][0,0]+green[1][0,0])/2.0

            self.points_measured.append([datax, datay])

            self.kf.add_measurement(datax, datay, t)
            p = self.kf.iterate_filter(t, adapt, predict)
           # p=None
            if p is not None:
                return (p[0], p[1], t)
            else:
                return None

class Window(QWidget):

    def __init__(self):
        # every 100 times, 5-10 coordinates are missing
        self.missing_counter=100
        self.timer = QElapsedTimer()
        self.RLED=Feature('red', offset=5.0)
        self.GLED=Feature('red', offset=-5.0)
        self.obj=Object(self.RLED, self.GLED)

        super(Window, self).__init__()
        #self.layout = QVBoxLayout(self)
        self.layout =  QGridLayout(self)
        #self.smoothbtn = QPushButton('Smooth trajectory', self)
        self.adaptbtn=QCheckBox('Adaptive Filtering', self)
        self.predictbtn=QCheckBox('Predict missing values', self)
        self.randombtn = QCheckBox('Add random noise', self)
       # self.fixedbtn=QRadioButton('Fixed parameters', self)
        self.resetbtn= QPushButton('Reset', self)
        self.plotbtn = QPushButton('Plot results', self)
        self.drawing= QWidget(self)

        #self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.adaptbtn, 0, 0)
        self.layout.addWidget(self.resetbtn, 0, 1)
        self.layout.addWidget(self.predictbtn, 1, 0)
        self.layout.addWidget(self.randombtn, 2, 0)
        #self.layout.addWidget(self.fixedbtn, 0, 1)
        self.layout.addWidget(self.plotbtn, 1, 1)
        self.layout.addWidget(self.drawing, 2, 1)
        self.initUI()

    def resetGUI(self, state):
        print "restart"
        self.RLED.reset()
        self.GLED.reset()
        self.obj.reset()

        self.timer.start()
        self.update()

    def initUI(self):
        self.setGeometry(300, 300, 600, 600)

        self.layout.setAlignment(Qt.AlignTop)
        #self.smoothbtn.clicked.connect(self.smoothing)
        self.resetbtn.clicked.connect(self.resetGUI)
        #self.plotbtn.clicked.connect(lambda: self.plotter.plot(self.adaptbtn.isChecked()))
        self.plotbtn.clicked.connect(self.Plotting)

        self.setWindowTitle('Points')
        self.show()

    def Plotting(self):
        self.RLED.plotter.plot(self.adaptbtn.isChecked(), 'red')
        self.GLED.plotter.plot(self.adaptbtn.isChecked(), 'green')
        self.obj.plotter.plot(self.adaptbtn.isChecked(), 'yellow')

    def plotMeasured(self, qp, m, u, color):
        if color=='red':
            qp.setBrush(Qt.red)
        elif color=='green':
            qp.setBrush(Qt.green)
        else:
            qp.setBrush(Qt.yellow)
        numPoints = u.shape[1]
        if len(m) > 0:
            if len(m) > numPoints:
                [qp.drawEllipse(m[i][0], m[i][1], 5, 5)
                 for i in range(len(m) - numPoints, len(m))]
            else:
                [qp.drawEllipse(m[i][0], m[i][1], 5, 5)
                 for i in range(0, len(m))]

    def plotUpdated(self, qp, m, u, color):
        numPoints=u.shape[1]
        startpoint = 1 if (len(m) > numPoints) else (numPoints - len(m) + 1)
        if color=='red':
            qp.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        elif color=='green':
            qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        else:
            qp.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
        [qp.drawLine(int(u[0, i]), int(u[1, i]), int(u[0, (i - 1)]),
                     int(u[1, (i - 1)])) for i in range(startpoint, (numPoints - 1))]

    def paintEvent(self, e):
        #print "e", e
        qp = QPainter(self)

        self.plotMeasured(qp, self.RLED.points_measured, self.RLED.kf.updated_state, 'red')
        self.plotMeasured(qp, self.GLED.points_measured, self.GLED.kf.updated_state, 'green')
        self.plotMeasured(qp, self.obj.points_measured, self.obj.kf.updated_state, 'yellow')

        self.plotUpdated(qp, self.RLED.points_measured, self.RLED.kf.updated_state, 'red')
        self.plotUpdated(qp, self.GLED.points_measured, self.GLED.kf.updated_state, 'green')
        self.plotUpdated(qp, self.obj.points_measured, self.obj.kf.updated_state, 'yellow')

        qp.end()

    def addPoints(self, x,y):
        t = self.timer.elapsed()
        self.timer.restart()
        updateRED=self.RLED.addPoints(x,y, t, self.adaptbtn.isChecked(), self.predictbtn.isChecked(), self.randombtn.isChecked())
        updateGREEN=self.GLED.addPoints(x, y, t, self.adaptbtn.isChecked(), self.predictbtn.isChecked(), self.randombtn.isChecked())
        #adds the measured coordinates
        updateobj=self.obj.add_points(updateRED[0],updateGREEN[0], t, self.adaptbtn.isChecked(), self.predictbtn.isChecked())
        if updateRED is not None:
            self.RLED.plotter.add_updated(updateRED[0], updateRED[1], updateRED[2])
        if updateGREEN is not None:
            self.GLED.plotter.add_updated(updateRED[0], updateRED[1], updateRED[2])
        if updateobj is not None:
            self.obj.plotter.add_updated(updateobj[0],updateobj[1],updateobj[2])

    def mousePressEvent(self, mouse_event):
        #print "start"
        self.RLED.reset()
        self.GLED.reset()
        self.obj.reset()

        self.timer.start()
        if num_variables==4:
            initial_state=np.array([[mouse_event.x()], [mouse_event.y()], [0.001], [0.001]])
        else:
            initial_state = np.array([[mouse_event.x()], [mouse_event.y()], [0.001], [0.001], [0.00], [0.0]])
        self.RLED.points_measured.append([mouse_event.x(), mouse_event.y()])
        self.GLED.points_measured.append([mouse_event.x(), mouse_event.y()])
        self.obj.points_measured.append([mouse_event.x(), mouse_event.y()])

       # self.RLED.updated_state[:, -1]=initial_state[:,0]
        self.RLED.kf.startFilter(initial_state[:,0])
        self.GLED.kf.startFilter(initial_state[:,0])
        self.obj.kf.startFilter(initial_state[:,0])
        self.update()

    def mouseMoveEvent(self, mouse_event):
        # print "mouse moved", mouse_event.x(), mouse_event.y()
        if self.predictbtn.isChecked():
            if self.missing_counter>missing_num:
                self.missing_counter=self.missing_counter-1
                self.addPoints(mouse_event.x(), mouse_event.y())
            elif self.missing_counter>0:
                self.missing_counter = self.missing_counter - 1
                self.addPoints(None, None)
            else:
                self.missing_counter=100
                self.addPoints(None, None)
        else:
            self.addPoints(mouse_event.x(), mouse_event.y())


        self.update()

    def mouseReleaseEvent(self, mouse_event):
        print "stop"
        print "smooth line"

def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
