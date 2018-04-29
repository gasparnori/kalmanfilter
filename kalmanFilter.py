import matplotlib.pyplot as plt
import numpy as np

import sys, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
#from pykalman import KalmanFilter
import math
num_variables=6
forget= 0.3 #forgetting factor

class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()
        #self.layout = QVBoxLayout(self)
        self.layout =  QGridLayout(self)
        self.smoothbtn = QPushButton('Smooth trajectory', self)
        self.predictbtn= QPushButton('Online prediction', self)
        self.drawing= QWidget(self)

        self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.predictbtn, 0, 1)
        self.layout.addWidget(self.drawing, 1, 1)
        self.resetFilter()


        self.initUI()

    def resetFilter(self):
        self.timer = QElapsedTimer()
        #to draw
        self.points_final = []
        self.points_online = []
        self.points_measured = []

        #for the filter
        self.measured_state = []
        self.predicted_state = []
        self.updated_state = []

        deltaT=0.5
        #Fk: transition matrix for only position and velocity
        #self.Fk =np.array( [[1, 0, deltaT, 0],[0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])

        #Fk: transition matrix for acceleration, velocity and position
        self.Fk = np.array([[1, 0, deltaT, 0, 0.5*deltaT*deltaT, 0],
                            [0, 1, 0, deltaT, 0, 0.5*deltaT*deltaT],
                            [0, 0, 1, 0, deltaT, 0],
                            [0, 0, 0, 1, 0, deltaT],
                            [0, 0, 0, 0, 1,0],
                            [0, 0, 0, 0, 0, 1]])
        #Hk:observation matrix
        self.Hk =np.eye(num_variables, num_variables) #not going to change
        #Pk: transition covariance
        self.Pk=np.zeros((num_variables, num_variables))#np.eye(num_variables, num_variables)
        #Rk: observation covariance
        #self.measurement_covariance = np.eye(4, 4)
        self.Rk =np.eye(num_variables, num_variables)* 10  # estimate of measurement variance, change to see effect
        # Q
        self.Q = [[0.2, 0, 0, 0, 0, 0],
                    [0, 0.2, 0, 0, 0, 0],
                    [0, 0, 0.0005, 0, 0, 0],
                    [0, 0, 0, 0.0005, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
        self.kf = None

        self.onlinefilter_means = []
        self.onlinefilter_covariance = []
    def smoothing(self):

        self.predictbtn.setChecked(False)


    def onlinepred(self, state):
        if state:
            print "online predicton"


    def initUI(self):
        self.setGeometry(300, 300, 600, 600)

        self.layout.setAlignment(Qt.AlignTop)
        self.smoothbtn.clicked.connect(self.smoothing)
        self.predictbtn.clicked.connect(self.onlinepred)
        self.predictbtn.setCheckable(True)
        self.setWindowTitle('Points')
        self.show()

    def paintEvent(self, e):
        #print "e", e
        qp = QPainter(self)
        qp.setBrush(Qt.red)
        if len(self.points_measured)>0:
            [qp.drawEllipse(self.points_measured[i][0], self.points_measured[i][1], 5, 5)
                for i in range(0, len(self.points_measured))]
        # qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        # if len(self.predicted_state)>1:
        #     [qp.drawLine(self.predicted_state[i][0], self.predicted_state[i][1], self.predicted_state[i-1][0],
        #                  self.predicted_state[i-1][1]) for i in range(1, len(self.predicted_state)-1)]

        qp.setPen(QPen(Qt.blue, 2, Qt.SolidLine))

        #print self.updated_state.shape
        #print len(self.updated_state)

        if len(self.updated_state)>2:

            [qp.drawLine(int(self.updated_state[i][0]), int(self.updated_state[i][1]), int(self.updated_state[i - 1][0]),
                         int(self.updated_state[i - 1][1])) for i in range(1, len(self.updated_state) - 1)]
        qp.end()

    def addPoints(self, x,y):
        mu, sigma = 0, 10  # mean and standard deviation
        datax = int(x + np.random.normal(mu, sigma, 1))
        datay = int(y + np.random.normal(mu, sigma, 1))
        t = self.timer.elapsed()
        # print t
        self.timer.restart()
        if t>0:
            vx= (datax-self.points_measured[-1][0])/t
            vy= (datay-self.points_measured[-1][1])/t
            ax= (vx-self.measured_state[2])/t
            ay= (vy-self.measured_state[3])/t
        else:
            vx= (self.measured_state[2])
            vy= (self.measured_state[3])
            ax= (self.measured_state[4])
            ay= (self.measured_state[5])

        self.measured_state=[datax, datay, vx, vy, ax, ay]
        self.points_measured.append([datax, datay])

        if self.predictbtn.isChecked():
            self.iterate_filter(t)


    def mousePressEvent(self, mouse_event):
        print "start"
        self.resetFilter()

        self.timer.start()
        self.initial_state=[mouse_event.x(), mouse_event.y(), 0, 0, 0, 0]
        self.points_measured.append([mouse_event.x(), mouse_event.y()])
        self.measured_state=self.initial_state
        self.updated_state.append(self.initial_state)
        self.predicted_state.append(self.initial_state)


    def mouseMoveEvent(self, mouse_event):
        # print "mouse moved", mouse_event.x(), mouse_event.y()
        self.addPoints(mouse_event.x(), mouse_event.y())

        self.update()

    #the filter itself
    def iterate_filter(self, dt):
        # print dt
       #dt=0.5
        #if len(self.points_measured)>100:
            #print np.cov(self.points_measured[-101:-1])
        HkT=self.Hk.transpose()
        self.Fk = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                            [0, 1, 0, dt, 0, 0.5*dt*dt],
                            [0, 0, 1, 0, dt, 0],
                            [0, 0, 0, 1, 0,dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        # self.Fk = np.array([[1, 0, dt, 0, 0, 0],
        #                      [0, 1, 0, dt, 0, 0],
        #                      [0, 0, 1, 0, dt, 0],
        #                      [0, 0, 0, 1, 0,dt],
        #                      [0, 0, 0, 0, 1, 0],
        #                      [0, 0, 0, 0, 0, 1]])
        pred = np.matmul(self.Fk, self.updated_state[-1])
        cov = np.matmul(np.matmul(self.Fk, self.Pk),self.Fk.transpose())+self.Q

        residual = self.measured_state - np.matmul(self.Hk, pred)
        #print residual
        S=np.matmul(residual, residual.transpose())
        #self.Rk = forget*self.Rk + (1-forget)*(S + np.matmul(np.matmul(self.Hk, cov), HkT))

        temp=np.matmul(np.matmul(self.Hk,cov),HkT) + self.Rk
        Kgain =np.matmul(np.matmul(cov, HkT), np.linalg.inv(temp))
        # print "Kgain", Kgain

        updateval = pred + np.matmul(Kgain, residual)

        self.Pk = np.matmul(np.eye(num_variables, num_variables) - np.matmul(Kgain, self.Hk), cov)

        self.updated_state.append(updateval)
        self.predicted_state.append(pred)

        # print "measured: ", self.measured_state[-1][0], self.measured_state[-1][1]
        # print "pred: ", self.predicted_state[-1][0], self.predicted_state[-1][1]
        # print "updated",  self.updated_state[-1][0],  self.updated_state[-1][1]

        # print "covariance: ", cov

    # def iterate_filter2(self, dt):
    #     #print dt
    #
    #     self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    #     pred =np.matmul(self.Fk, self.updated_state[-1])
    #     cov = self.Fk*self.Pk*self.Fk.transpose()
    #     #Rk=np.eye(4,4)*self.R
    #
    #     #Kgain=np.matmul(np.matmul(self.Pk, self.Hk.transpose()), temp2)
    #     Kgain = cov*self.Hk.transpose()*np.linalg.inv(self.Hk*cov*self.Hk.transpose()+self.Rk)
    #     #print "Kgain", Kgain
    #     updateval=pred+np.matmul(Kgain, self.measured_state[-1])-np.matmul(self.Hk, pred)
    #     self.Pk= cov - np.matmul(np.matmul(Kgain, self.Hk), cov)
    #
    #     self.updated_state.append(updateval)
    #     self.predicted_state.append(pred)
    #
    #     #print "measured: ", self.measured_state[-1][0], self.measured_state[-1][1]
    #     #print "pred: ", self.predicted_state[-1][0], self.predicted_state[-1][1]
    #     #print "updated",  self.updated_state[-1][0],  self.updated_state[-1][1]
    #
    #
    #     #print "covariance: ", cov


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