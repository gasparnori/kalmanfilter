import matplotlib.pyplot as plt
import numpy as np

import sys, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from pykalman import KalmanFilter
import math

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
        #Fk: transition matrix
        self.Fk =np.array( [[1, 0, deltaT, 0],[0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
        #Hk:observation matrix
        self.Hk =np.eye(4, 4) #not going to change
        #Pk: transition covariance
        self.Pk=np.eye(4, 4)
        #Rk: observation covariance
        #self.measurement_covariance = np.eye(4, 4)
        self.R = 5  # estimate of measurement variance, change to see effect
        # Q
        self.Q = 0.1  # process variance
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
        qp.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        if len(self.predicted_state)>1:
            [qp.drawLine(self.predicted_state[i][0], self.predicted_state[i][1], self.predicted_state[i-1][0],
                         self.predicted_state[i-1][1]) for i in range(1, len(self.predicted_state)-1)]

        qp.setPen(QPen(Qt.blue, 2, Qt.SolidLine))

        if len(self.updated_state)>2:
            [qp.drawLine(self.updated_state[i][0], self.updated_state[i][1], self.updated_state[i - 1][0],
                         self.updated_state[i - 1][1]) for i in range(1, len(self.updated_state) - 1)]
        qp.end()

    def addPoints(self, x,y):
        mu, sigma = 0, 10  # mean and standard deviation
        datax = int(x + np.random.normal(mu, sigma, 1))
        datay = int(y + np.random.normal(mu, sigma, 1))
        t = self.timer.elapsed()
        # print t
        self.timer.restart()

        vx= (datax-self.points_measured[-1][0])/t
        vy= (datay-self.points_measured[-1][1])/t
        self.measured_state.append([datax, datay, vx, vy])
        self.points_measured.append([datax, datay])

        if self.predictbtn.isChecked():
            self.iterate_filter(t)


    def mousePressEvent(self, mouse_event):
        print "start"
        self.resetFilter()

        self.timer.start()
        self.initial_state=[mouse_event.x(), mouse_event.y(), 0, 0]
        self.points_measured.append([mouse_event.x(), mouse_event.y()])
        self.measured_state.append(self.initial_state)
        self.updated_state.append(self.initial_state)
        self.predicted_state.append(self.initial_state)


    def mouseMoveEvent(self, mouse_event):
        # print "mouse moved", mouse_event.x(), mouse_event.y()
        self.addPoints(mouse_event.x(), mouse_event.y())

        self.update()

    #the filter itself
    def iterate_filter(self, dt):
        #print dt
        #self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        pred =np.matmul(self.Fk, self.updated_state[-1])
        cov = np.matmul(np.matmul(self.Fk, self.Pk), self.Fk.transpose()) + self.Q
        Rk=np.eye(4,4)*self.R

        #Kgain=np.matmul(np.matmul(self.Pk, self.Hk.transpose()), temp2)
        Kgain = np.matmul(self.Pk, np.linalg.inv(self.Pk+self.R))
        print "Kgain", Kgain
        updateval=pred+np.matmul(Kgain, (self.measured_state[-1]-pred))
        self.Pk=cov+np.matmul(Kgain, self.Pk)

        self.updated_state.append(updateval)
        self.predicted_state.append(pred)

        print "measured: ", self.measured_state[-1][0], self.measured_state[-1][1]
        print "pred: ", self.predicted_state[-1][0], self.predicted_state[-1][1]
        print "updated",  self.updated_state[-1][0],  self.updated_state[-1][1]


        #print "covariance: ", cov


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