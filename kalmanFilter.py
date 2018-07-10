import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import sys, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math
import random
num_variables=4
forget= 0.001 #forgetting factor
confidenceInterval=20
confIntcoeff=60
max_x=640
max_y=360

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

    def plot(self, adaptive=False):
        plt.clf()
        plt.close()
        titletxt = "Smoothing with adaptive covariances" if adaptive else "Smoothing with fixed covariances"
        p1=plt.subplot(211)
        plt.title(titletxt)
        #plt.xlabel("time (ms)")
        plt.ylabel("coordinate X (pixel)")
        ux, =plt.plot(self.times, self.updatedX, 'g', label='Updated')
        mx, =plt.plot(self.times, self.measuredX, 'r', label='Measured')
        plt.legend(handles=[ux, mx])
        #plt.setp(p1.get_xticklabels(), visible=False)
        p2= plt.subplot(212, sharex=p1)  # creates 2nd subplot with yellow background
        plt.xlabel("time (ms)")
        plt.ylabel("coordinate Y (pixel)")
        uy, =plt.plot(self.times, self.updatedY, 'g', label='Updated')
        my, =plt.plot(self.times, self.measuredY, 'r',  label='Measured')
        plt.legend(handles=[uy, my])
        plt.show()

    def reset(self):
        self.updatedX = []
        self.measuredX = []
        self.updatedY = []
        self.measuredY = []
        self.times = []


class Window(QWidget):
    maxPredictions=100
    def __init__(self):
        # every 100 times, 5-10 coordinates are missing
        self.missing_counter=100
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
        self.plotter=Plotter()

        #self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.adaptbtn, 0, 0)
        self.layout.addWidget(self.resetbtn, 0, 1)
        self.layout.addWidget(self.predictbtn, 1, 0)
        #self.layout.addWidget(self.fixedbtn, 0, 1)
        self.layout.addWidget(self.plotbtn, 1, 1)
        self.layout.addWidget(self.drawing, 2, 1)
        self.resetFilter()

        self.initUI()

    def resetFilter(self):
        self.plotter.reset()
        self.timer = QElapsedTimer()
        #to draw
        self.points_final = []
        self.points_online = []
        self.points_measured = []

        #for the filter
        self.measured_state = np.zeros(shape=(2, 1))
       # self.predicted_state = []
        #a FIFO
        self.updated_state = np.zeros(shape=(num_variables, 100))

        deltaT=0.5
        #Fk: transition matrix for only position and velocity
        #self.Fk =np.array( [[1, 0, deltaT, 0],[0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])

        #Fk: transition matrix for acceleration, velocity and position
        # self.Fk = np.matrix(((1, 0, deltaT, 0, 0.5*deltaT*deltaT, 0),
        #                     (0, 1, 0, deltaT, 0, 0.5*deltaT*deltaT),
        #                     (0, 0, 1, 0, deltaT, 0),
        #                     (0, 0, 0, 1, 0, deltaT),
        #                     (0, 0, 0, 0, 1,0),
        #                     (0, 0, 0, 0, 0, 1)))

        # Hk:observation matrix
        self.Hk = np.eye(2, num_variables)  # not going to change
        # Pk: transition covariance
        self.Pk = np.zeros(shape=(num_variables, num_variables))  # np.eye(num_variables, num_variables)
        # Rk: observation covariance
        # self.measurement_covariance = np.eye(4, 4)

        self.Rk = np.eye(2, 2) * 10  # estimate of measurement variance, change to see effect
        # Q
        self.Qk = np.matrix([[0.5, 0, 0, 0],#, 0, 0],
                    [0, 0.5, 0, 0],#, 0, 0],
                    [0, 0, 0.001, 0],#, 0, 0],
                    [0, 0, 0, 0.001]])#, 0, 0],
                   # [0, 0, 0, 0, 0.0001, 0],
                   # [0, 0, 0, 0, 0, 0.0001]])
        self.Kgain=np.eye(num_variables, 2)
        self.kf = None

        self.onlinefilter_means = []
        self.onlinefilter_covariance = []

    def resetGUI(self, state):
        print "restart"
        self.resetFilter()
        self.timer.start()
        self.update()

    def initUI(self):
        self.setGeometry(300, 300, 600, 600)

        self.layout.setAlignment(Qt.AlignTop)
        #self.smoothbtn.clicked.connect(self.smoothing)
        self.resetbtn.clicked.connect(self.resetGUI)
        self.plotbtn.clicked.connect(lambda: self.plotter.plot(self.adaptbtn.isChecked()))

        self.setWindowTitle('Points')
        self.show()

    def paintEvent(self, e):
        #print "e", e
        numPoints=self.updated_state.shape[1]
        qp = QPainter(self)
        qp.setBrush(Qt.red)
        if len(self.points_measured)>0:
            if len(self.points_measured)>numPoints:
                [qp.drawEllipse(self.points_measured[i][0], self.points_measured[i][1], 5, 5)
                    for i in range(len(self.points_measured)-numPoints, len(self.points_measured))]
            else:
                [qp.drawEllipse(self.points_measured[i][0], self.points_measured[i][1], 5, 5)
                 for i in range(0, len(self.points_measured))]

        # qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        # if len(self.predicted_state)>1:
        #     [qp.drawLine(self.predicted_state[i][0], self.predicted_state[i][1], self.predicted_state[i-1][0],
        #                  self.predicted_state[i-1][1]) for i in range(1, len(self.predicted_state)-1)]

        qp.setPen(QPen(Qt.blue, 2, Qt.SolidLine))

        #print self.updated_state.shape
        #print len(self.updated_state)

       # if self.updated_state)>2:
        startpoint=1 if len(self.points_measured)>numPoints else (numPoints-len(self.points_measured)+1)
        #print startpoint
        [qp.drawLine(int(self.updated_state[0,i]), int(self.updated_state[1,i]), int(self.updated_state[0,(i - 1)]),
                int(self.updated_state[1,(i - 1)])) for i in range(startpoint, (numPoints - 1))]
        qp.end()

    def addPoints(self, x,y):
        mu, sigma = 0, 10  # mean and standard deviation
        t = self.timer.elapsed()
        # print t
        self.timer.restart()

        if x is None:
            p = self.iterateRobustFilter(t,
                                    np.asmatrix(self.updated_state[:, -1]).transpose(),
                                    None,
                                    self.adaptbtn.isChecked(),
                                    self.predictbtn.isChecked())
            #self.updated_state[:, :-1]=self.updated_state[:, 1:]
            #self.updated_state[:, -1]=p.A1
            self.measured_state = self.updated_state[0:2, -1]
            self.measured_state.shape = (2, 1)
            return
        else:
            if self.measured_state is None:
                self.measured_state = self.updated_state[0:2, -1]
                self.measured_state.shape = (2, 1)
            #print self.measured_state
            datax = int(x + np.random.normal(mu, sigma, 1))
            datay = int(y + np.random.normal(mu, sigma, 1))

            # if t>0:
            #     vx= (datax-self.measured_state[0,0])/t
            #     vy= (datay-self.measured_state[1,0])/t
            #     ax= (vx-self.measured_state[2,0])/t
            #     ay= (vy-self.measured_state[3,0])/t
            #     print "ax, ay calculation:", ax, ay
            # else:
            #     vx= (self.measured_state[2,0])
            #     vy= (self.measured_state[3,0])
            #     ax= (self.measured_state[4,0])
            #     ay= (self.measured_state[5,0])

            self.measured_state=np.array([[datax], [datay]])#, [vx], [vy]])#, [ax], [ay]])
           # print self.measured_state
            self.points_measured.append([datax, datay])
            p=self.iterateRobustFilter(t,
                              np.asmatrix(self.updated_state[:, -1]).transpose(),
                              np.asmatrix(self.measured_state),
                              self.adaptbtn.isChecked(),
                              self.predictbtn.isChecked())

            #self.updated_state[:, :-1] = self.updated_state[:, 1:]
            #self.updated_state[:, -1] = p.A1
            #print self.updated_state[:, -1]


    def mousePressEvent(self, mouse_event):
        print "start"
        self.resetFilter()

        self.timer.start()
        initial_state=np.array([[mouse_event.x()], [mouse_event.y()], [0.0001], [0.0001]])#, [0.0001], [0.0001]])
        self.points_measured.append([mouse_event.x(), mouse_event.y()])
        #self.measured_state[:, 0] = initial_state[:, 0]
        #print self.measured_state
        self.updated_state[:, -1]=initial_state[:,0]
       # self.predicted_state.append(self.initial_state)

    def mouseMoveEvent(self, mouse_event):
        # print "mouse moved", mouse_event.x(), mouse_event.y()
        if self.predictbtn.isChecked():
            if self.missing_counter>10:
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

    def expWeight(self, x):
        if abs(x) < confidenceInterval:
            y = 1
        else:
            y = 0.5 * np.exp(1 - (abs(x) / confIntcoeff))
        return (y)

    def iterateRobustFilter(self,dt, u, m, adaptive=False, guessing_enabled=True):
       # u = np.asmatrix(self.updated_state[:, -1]).transpose()         #state matrix (4x1) or (6x1)
        if num_variables == 4:
            self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])         # system dynamics (4x4 or 6x6)
        else:
            self.Fk = np.array([[1, 0, dt, 0, 0.5 * dt * dt, 0],
                                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                                [0, 0, 1, 0, dt, 0],
                                [0, 0, 0, 1, 0, dt],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
        # #prediction step
        pred = self.Fk * u         #A priori state matrix (4x1) or (6x1)
        cov = self.Fk * self.Pk * self.Fk.transpose() + self.Qk         #A priori covariance matrix

        retVal=None #default return value

        #update step if there was a measurement
        if m is not None:
            self.predictionCounter=0
          #  m = self.add_measurement(dt, coordinates)

            #our addition to the weight

            dx = np.linalg.norm(m-pred[0:2])
            mWeight = self.expWeight(dx)

            diff = (m - self.Hk * pred)  # difference between prediction and measurement * mWeight

            # setting the matrices
            S = np.linalg.inv(self.Hk * cov * self.Hk.T + self.Rk)
            self.Kgain = cov * self.Hk.T * S# *mWeight # Kalman gain
            self.Pk = (np.eye(num_variables) - (self.Kgain * self.Hk)) * cov  # A posteriori covariance matrix

            # update
            updateval = pred + self.Kgain * diff

            # print self.calibrating
            #if self.calibrating:
            #    self.QCalib.append(self.Kgain * diff * diff.T * self.Kgain.T)

            # #adapting Rk and Qk
            if adaptive:
                self.Qk = forget * self.Qk + \
                          (1 - forget) * self.Kgain * diff * diff.transpose() * self.Kgain.transpose()
                residual = m - (self.Hk * updateval)
                self.Rk = forget * self.Rk + (1 - forget) * (
                            residual * residual.T + (self.Hk * self.Pk * self.Hk.T))

            self.updated_state[:, :-1] = self.updated_state[:, 1:]
            self.updated_state[:, -1] = updateval.A1

            xval = int(round(updateval[0, 0])) if round(updateval[0, 0]) > 0 and round(
                updateval[0, 0]) < max_x else None
            yval = int(round(updateval[1, 0])) if round(updateval[1, 0]) > 0 and round(
                updateval[1, 0]) < max_y else None

            if xval is not None and yval is not None:
                # print 'Measured: ', (m[0,0], m[1,0]), 'Predicted: ', (pred[0,0],  pred[1,0]), 'updated: ', (xval, yval)
                retVal=(xval, yval)
        else:
            #if not self.calibrating:
            if guessing_enabled and self.predictionCounter<self.maxPredictions:
                #update the list with the prediction
                self.updated_state[:, :-1] = self.updated_state[:, 1:]
                self.updated_state[:, -1] = pred.A1
                self.predictionCounter=self.predictionCounter+1

                xpred=int(round(pred[0,0])) if round(pred[0,0])>0 and round(pred[0,0])<max_x else None
                ypred=int(round(pred[1,0])) if round(pred[1,0])>0 and round(pred[1,0])<max_y else None

                if xpred is not None and ypred is not None:
                    #self.log.debug("predicting missing value")
                   # print 'Missing value nr. ', self.predictionCounter, '. Predicted: ', (pred[0,0],  pred[1,0]), 'updated: ', (xpred, ypred)
                    retVal=(xpred, ypred)
        return retVal




    #the filter itself
    def iterate_filter(self, dt, u, m, adaptive=False, fixed=True):
        # print dt


       # m = np.asmatrix(m_state)
       # u = np.asmatrix(u_state).transpose()

        HkT=self.Hk.transpose()
        self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.Fk = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
        #                     [0, 1, 0, dt, 0, 0.5*dt*dt],
        #                     [0, 0, 1, 0, dt, 0],
        #                     [0, 0, 0, 1, 0,dt],
        #                     [0, 0, 0, 0, 1, 0],
        #                     [0, 0, 0, 0, 0, 1]])

        # #prediction step

        pred = self.Fk*u
        cov = self.Fk*self.Pk*self.Fk.transpose()+self.Qk
        #if self.missing_counter<10:
        #    print pred, u, dt, self.Fk
       # print "acceleration: ", pred[4,0], pred[5,0], "last measure: ", u[4, 0], u[5, 0]
        #check if there is a data:
        if m is None:
          #  print "predicted:"
          #  print "velocity: ", pred[2, 0], pred[3, 0], "acceleration: ", pred[4,0], pred[5,0], "last measure: ", u[2, 0], u[3, 0], "\n\n"
            updateval=pred
            self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
            self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov
            self.plotter.add_updated([0, 0], [updateval[0, 0], updateval[1, 0]], dt)

        else:
            #before update
            diff = m - self.Hk * pred
            self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
            # #adapting Rk and Qk
            if adaptive:
                self.Qk = forget * self.Qk + (1 - forget) * self.Kgain * diff * diff.transpose() * self.Kgain.transpose()

            #update
            updateval = pred + self.Kgain * diff
            residual=m-(self.Hk*updateval)
            self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov

            self.plotter.add_updated([m[0,0], m[1,0]], [updateval[0,0], updateval[1,0]], dt)
            #c=np.cov(residual)
            if adaptive:
                self.Rk = forget * self.Rk + (1 - forget) * (residual * residual.T + (self.Hk * self.Pk * HkT))

       # print "\n m: \n", m, "\n u: \n", u, "\n updateval: \n", updateval
        return updateval
        #self.predicted_state.append(pred)

    def mouseReleaseEvent(self, mouse_event):
        print "stop"
        print "smooth line"
        #self.predict_all_trajectories()
        #print self.points


def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

mu, sigma = 0, 2 # mean and standard deviation
datax=np.random.normal(mu, sigma, 1000)
print datax