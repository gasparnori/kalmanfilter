import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import sys, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math
import random
num_variables=6
missing_num=5
forget_R= 0.3 #forgetting factor
forget_Q=0.3

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

        ux, =plt.plot(self.times, self.updatedX, label='Updated')

        if color == 'red':
            mx, =plt.plot(self.times, self.measuredX, 'r', label='Measured')
        elif color=='green':
            mx, = plt.plot(self.times, self.measuredX, 'g', label='Measured')
        else:
            mx, = plt.plot(self.times, self.measuredX, 'y', label='Measured')
        plt.legend(handles=[ux, mx])
        #plt.setp(p1.get_xticklabels(), visible=False)
        p2= fig1.add_subplot(212, sharex=p1)  # creates 2nd subplot with yellow background
        plt.xlabel("time (ms)")

        plt.ylabel("coordinate Y (pixel)")
        uy, =plt.plot(self.times, self.updatedY, label='Updated')

        if color == 'red':
            my, =plt.plot(self.times, self.measuredY, 'r',  label='Measured')
        elif color == 'green':
            my, = plt.plot(self.times, self.measuredY, 'g', label='Measured')
        else:
            my, = plt.plot(self.times, self.measuredY, 'y', label='Measured')
        plt.legend(handles=[uy, my])
        plt.show()

    def reset(self):
        self.updatedX = []
        self.measuredX = []
        self.updatedY = []
        self.measuredY = []
        self.times = []

class Filter():
    def __init__(self):
        self.init()
    def init(self):
        deltaT=5
        if num_variables == 4:
            # Fk: transition matrix for only position and velocity
            self.Fk = np.array([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
        if num_variables == 6:
            # Fk: transition matrix for acceleration, velocity and position
            self.Fk = np.matrix(((1, 0, deltaT, 0, 0.5 * deltaT * deltaT, 0),
                                 (0, 1, 0, deltaT, 0, 0.5 * deltaT * deltaT),
                                 (0, 0, 1, 0, deltaT, 0),
                                 (0, 0, 0, 1, 0, deltaT),
                                 (0, 0, 0, 0, 1, 0),
                                 (0, 0, 0, 0, 0, 1)))
        # Hk:observation matrix
        self.Hk = np.eye(num_variables, num_variables)  # not going to change
        # Pk: transition covariance
        self.Pk = np.zeros(shape=(num_variables, num_variables))  # np.eye(num_variables, num_variables)
        # Rk: observation covariance
        # self.measurement_covariance = np.eye(4, 4)
        self.Rk = np.eye(num_variables, num_variables) * 20  # estimate of measurement variance, change to see effect
        # Q
        if num_variables == 4:
            self.Qk = np.matrix([[0.3, 0, 0, 0],
                                 [0, 0.3, 0, 0],
                                 [0, 0, 0.001, 0],
                                 [0, 0, 0, 0.001]])
        if num_variables == 6:
            self.Qk = np.matrix([[0.7, 0, 0, 0, 0, 0],
                                 [0, 0.7, 0, 0, 0, 0],
                                 [0, 0, 0.01, 0, 0, 0],
                                 [0, 0, 0, 0.01, 0, 0],
                                 [0, 0, 0, 0, 0.0, 0],
                                 [0, 0, 0, 0, 0, 0.0]])

        self.Kgain = np.eye(num_variables, num_variables)

    def iterate_filter(self, dt, u, m, adaptive=False, fixed=True):
        # print dt


       # m = np.asmatrix(m_state)
       # u = np.asmatrix(u_state).transpose()
        HkT=self.Hk.transpose()
        if num_variables == 4:
            self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            self.Fk = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                                [0, 1, 0, dt, 0, 0.5*dt*dt],
                                [0, 0, 1, 0, dt, 0],
                                [0, 0, 0, 1, 0,dt],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])

        if m is None:
            pred = self.Fk * u
            cov = self.Fk * self.Pk * self.Fk.transpose() + self.Qk  # Qk-1: Q always a step behind
            #  print "predicted:"
            #  print "velocity: ", pred[2, 0], pred[3, 0], "acceleration: ", pred[4,0], pred[5,0], "last measure: ", u[2, 0], u[3, 0], "\n\n"
            updateval = pred
            self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
            self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov
        else:
            if adaptive:
                pred = self.Fk * u
                cov = self.Fk * self.Pk * self.Fk.transpose() + self.Qk #Qk-1: Q always a step behind

                innovation = m - self.Hk * pred
                residual = m - (self.Hk * (pred + self.Kgain * innovation))

                self.Rk = forget_R * self.Rk + (1 - forget_R) * (residual * residual.T+ (self.Hk * cov * HkT))
                self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
                self.Qk = forget_Q * self.Qk + (1 - forget_Q) * self.Kgain * innovation * innovation.T * self.Kgain.T

                #sets the update values again
                updateval = pred + self.Kgain * innovation
                self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov
            else:
                pred = self.Fk*u
                cov = self.Fk*self.Pk*self.Fk.transpose()+self.Qk
                #before update
                diff = m - self.Hk * pred
                self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
                updateval = pred + self.Kgain * diff
                self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov

            #self.plotter.add_updated([m[0,0], m[1,0]], [updateval[0,0], updateval[1,0]], dt)
        #print updateval
        return updateval

class Feature():
    def __init__(self, color, offset):
        self.color=color
        self.offset=offset
        self.plotter = Plotter()
        self.reset()

    def reset(self):
        #to draw
        self.points_final = []
        self.points_online = []
        self.points_measured = []
        # for the filter
        self.measured_state = np.zeros(shape=(num_variables, 1))
        # a FIFO
        self.updated_state = np.zeros(shape=(num_variables, 100))
        self.kf=Filter()

    def addPoints(self, x,y, t, adapt, predict):
        mu, sigma = 0, 10  # mean and standard deviation

        if x is None:
            p = self.kf.iterate_filter(t,
                                    np.asmatrix(self.updated_state[:, -1]).transpose(),
                                    None,
                                    adapt,
                                    predict)

            self.updated_state[:, :-1]=self.updated_state[:, 1:]
            self.updated_state[:, -1]=p.A1
            self.measured_state = self.updated_state[:, -1]
            self.measured_state.shape = (num_variables, 1)
            return ([0,0], [p[0, 0], p[1, 0]], t)
        else:
            if self.measured_state is None:
                self.measured_state=self.updated_state[:, -1]
                self.measured_state.shape=(num_variables,1)

            datax = int(x +self.offset+ np.random.normal(mu, sigma, 1))
            datay = int(y +self.offset+ np.random.normal(mu, sigma, 1))

            if t>0:
                vx= (datax-self.measured_state[0,0])/t      #px/usec
                vy= (datay-self.measured_state[1,0])/t      #px/usec
                if num_variables == 6:
                    ax= (vx-self.measured_state[2,0])/t         #px/usec^2
                    ay= (vy-self.measured_state[3,0])/t         #px/usec^2
                    print "ax, ay calculation:", ax, ay
            else:
                vx= (self.measured_state[2,0])
                vy= (self.measured_state[3,0])
                if num_variables == 6:
                    ax= (self.measured_state[4,0])
                    ay= (self.measured_state[5,0])

            if num_variables == 4:
                self.measured_state = np.array([[datax], [datay], [vx], [vy]])
            else:
                self.measured_state=np.array([[datax], [datay], [vx], [vy], [ax], [ay]])
           # print self.measured_state
            self.points_measured.append([datax, datay])
            p=self.kf.iterate_filter(t,
                              np.asmatrix(self.updated_state[:, -1]).transpose(),
                              np.asmatrix(self.measured_state),
                              adapt,
                              predict)

            self.updated_state[:, :-1] = self.updated_state[:, 1:]
            self.updated_state[:, -1] = p.A1
            print self.updated_state[:, -1]
            #self.plotter.add_updated([0, 0], [updateval[0, 0], updateval[1, 0]], dt)
            return ([self.measured_state[0,0], self.measured_state[1,0]], [p[0,0], p[1,0]], t)

class Object():
    def __init__(self, red, green):
        self.red=red
        self.green=green
        self.plotter=Plotter()
        self.reset()

    def reset(self):
        #to draw
        self.points_final = []
        self.points_online = []
        self.points_measured = []
        # for the filter
        self.measured_state = np.zeros(shape=(num_variables, 1))
        # a FIFO
        self.updated_state = np.zeros(shape=(num_variables, 100))
        self.kf=Filter()

    def add_points(self, red, green, t, adapt, predict):
        # if at least one of the the data is missing
        if (red[0]==0 and red [1]==0) or (green[0]==0 and green[1]==0):
            p = self.kf.iterate_filter(t,
                                    np.asmatrix(self.updated_state[:, -1]).transpose(),
                                    None,
                                    adapt,
                                    predict)

            self.updated_state[:, :-1]=self.updated_state[:, 1:]
            self.updated_state[:, -1]=p.A1
            self.measured_state = self.updated_state[:, -1]
            self.measured_state.shape = (num_variables, 1)
            return ([0,0], [p[0, 0], p[1, 0]], t)
        else:
            datax=(red[0]+green[0])/2.0
            datay=(red[1]+green[1])/2.0
            if t > 0:
                vx = (datax - self.measured_state[0, 0]) / t  # px/usec
                vy = (datay - self.measured_state[1, 0]) / t  # px/usec
                if num_variables == 6:
                    ax = (vx - self.measured_state[2, 0]) / t  # px/usec^2
                    ay = (vy - self.measured_state[3, 0]) / t  # px/usec^2
                    print "ax, ay calculation:", ax, ay
            else:
                vx = (self.measured_state[2, 0])
                vy = (self.measured_state[3, 0])
                if num_variables == 6:
                    ax = (self.measured_state[4, 0])
                    ay = (self.measured_state[5, 0])

            if num_variables == 4:
                self.measured_state = np.array([[datax], [datay], [vx], [vy]])
            else:
                self.measured_state = np.array([[datax], [datay], [vx], [vy], [ax], [ay]])
                # print self.measured_state
            self.points_measured.append([datax, datay])
            p = self.kf.iterate_filter(t,
                                       np.asmatrix(self.updated_state[:, -1]).transpose(),
                                       np.asmatrix(self.measured_state),
                                       adapt,
                                       predict)

            self.updated_state[:, :-1] = self.updated_state[:, 1:]
            self.updated_state[:, -1] = p.A1
            print self.updated_state[:, -1]
            # self.plotter.add_updated([0, 0], [updateval[0, 0], updateval[1, 0]], dt)
            return ([self.measured_state[0, 0], self.measured_state[1, 0]], [p[0, 0], p[1, 0]], t)



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
       # self.fixedbtn=QRadioButton('Fixed parameters', self)
        self.resetbtn= QPushButton('Reset', self)
        self.plotbtn = QPushButton('Plot results', self)
        self.drawing= QWidget(self)

        #self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.adaptbtn, 0, 0)
        self.layout.addWidget(self.resetbtn, 0, 1)
        self.layout.addWidget(self.predictbtn, 1, 0)
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

        self.plotMeasured(qp, self.RLED.points_measured, self.RLED.updated_state, 'red')
        self.plotMeasured(qp, self.GLED.points_measured, self.GLED.updated_state, 'green')
        self.plotMeasured(qp, self.obj.points_measured, self.obj.updated_state, 'yellow')

        self.plotUpdated(qp, self.RLED.points_measured, self.RLED.updated_state, 'red')
        self.plotUpdated(qp, self.GLED.points_measured, self.GLED.updated_state, 'green')
        self.plotUpdated(qp, self.obj.points_measured, self.obj.updated_state, 'yellow')

        qp.end()


    def addPoints(self, x,y):
        t = self.timer.elapsed()
        self.timer.restart()
        updateRED=self.RLED.addPoints(x,y, t, self.adaptbtn.isChecked(), self.predictbtn.isChecked())
        updateGREEN=self.GLED.addPoints(x, y, t, self.adaptbtn.isChecked(), self.predictbtn.isChecked())
        #adds the measured coordinates
        udapteobj=self.obj.add_points(updateRED[0],updateGREEN[0], t, self.adaptbtn.isChecked(), self.predictbtn.isChecked())

        self.RLED.plotter.add_updated(updateRED[0], updateRED[1], updateRED[2])
        self.GLED.plotter.add_updated(updateRED[0], updateRED[1], updateRED[2])
        self.obj.plotter.add_updated(udapteobj[0],udapteobj[1],udapteobj[2])


    def mousePressEvent(self, mouse_event):
        print "start"
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

        self.RLED.updated_state[:, -1]=initial_state[:,0]
        self.GLED.updated_state[:, -1] = initial_state[:, 0]
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
