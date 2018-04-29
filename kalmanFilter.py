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
        #self.smoothbtn = QPushButton('Smooth trajectory', self)
        self.adaptbtn=QCheckBox('Adaptive Filtering', self)
        self.predictbtn= QPushButton('Start prediction', self)
        self.drawing= QWidget(self)

        #self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.adaptbtn, 0, 0)
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
        self.measured_state = np.zeros(shape=(num_variables, 1))
       # self.predicted_state = []
        #a FIFO
        self.updated_state = np.zeros(shape=(num_variables, 100))

        deltaT=0.5
        #Fk: transition matrix for only position and velocity
        #self.Fk =np.array( [[1, 0, deltaT, 0],[0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])

        #Fk: transition matrix for acceleration, velocity and position
        self.Fk = np.matrix(((1, 0, deltaT, 0, 0.5*deltaT*deltaT, 0),
                            (0, 1, 0, deltaT, 0, 0.5*deltaT*deltaT),
                            (0, 0, 1, 0, deltaT, 0),
                            (0, 0, 0, 1, 0, deltaT),
                            (0, 0, 0, 0, 1,0),
                            (0, 0, 0, 0, 0, 1)))
        #Hk:observation matrix
        self.Hk =np.eye(num_variables, num_variables) #not going to change
        #Pk: transition covariance
        self.Pk=np.zeros(shape=(num_variables, num_variables))#np.eye(num_variables, num_variables)
        #Rk: observation covariance
        #self.measurement_covariance = np.eye(4, 4)
        self.Rk =np.eye(num_variables, num_variables)* 10  # estimate of measurement variance, change to see effect
        # Q
        self.Qk = np.matrix([[0.5, 0, 0, 0, 0, 0],
                    [0, 0.5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
        self.Kgain=np.eye(num_variables, num_variables)
        self.kf = None

        self.onlinefilter_means = []
        self.onlinefilter_covariance = []
 #   def smoothing(self):

 #       self.predictbtn.setChecked(False)


    def onlinepred(self, state):
        if state:
            print "online predicton"


    def initUI(self):
        self.setGeometry(300, 300, 600, 600)

        self.layout.setAlignment(Qt.AlignTop)
        #self.smoothbtn.clicked.connect(self.smoothing)
        self.predictbtn.clicked.connect(self.onlinepred)
        self.predictbtn.setCheckable(True)
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
        datax = int(x + np.random.normal(mu, sigma, 1))
        datay = int(y + np.random.normal(mu, sigma, 1))
        t = self.timer.elapsed()
        # print t
        self.timer.restart()
        if t>0:
            vx= (datax-self.points_measured[-1][0])/t
            vy= (datay-self.points_measured[-1][1])/t
            ax= (vx-self.measured_state[2,0])/t
            ay= (vy-self.measured_state[3,0])/t
        else:
            vx= (self.measured_state[2,0])
            vy= (self.measured_state[3,0])
            ax= (self.measured_state[4,0])
            ay= (self.measured_state[5,0])

        self.measured_state=np.array([[datax], [datay], [vx], [vy], [ax], [ay]])
        self.points_measured.append([datax, datay])

        if self.predictbtn.isChecked():
            #important to change it to the matrix form required

            p=self.iterate_filter(t,
                                  np.asmatrix(self.updated_state[:, -1]).transpose(),
                                  np.asmatrix(self.measured_state),
                                  self.adaptbtn.isChecked())
            self.updated_state[:, :-1]=self.updated_state[:, 1:]

            self.updated_state[:, -1]=p.A1

           # print "blabla"


    def mousePressEvent(self, mouse_event):
        print "start"
        self.resetFilter()

        self.timer.start()
        initial_state=np.array([[mouse_event.x()], [mouse_event.y()], [0], [0], [0], [0]])
        self.points_measured.append([mouse_event.x(), mouse_event.y()])
        self.measured_state[:, 0] = initial_state[:, 0]
        #print self.measured_state
        self.updated_state[:, -1]=initial_state[:,0]
       # self.predicted_state.append(self.initial_state)


    def mouseMoveEvent(self, mouse_event):
        # print "mouse moved", mouse_event.x(), mouse_event.y()
        self.addPoints(mouse_event.x(), mouse_event.y())

        self.update()

    #the filter itself
    def iterate_filter(self, dt, u, m, adaptive=False):
        # print dt


       # m = np.asmatrix(m_state)
       # u = np.asmatrix(u_state).transpose()
        HkT=self.Hk.transpose()

        self.Fk = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                            [0, 1, 0, dt, 0, 0.5*dt*dt],
                            [0, 0, 1, 0, dt, 0],
                            [0, 0, 0, 1, 0,dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        # #prediction step
        pred = self.Fk*u
        cov = self.Fk*self.Pk*self.Fk.transpose()+self.Qk

        #before update
        diff = m - self.Hk * pred
        self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
        # #adapting Rk and Qk

        #update
        updateval = pred + self.Kgain * diff
        residual=m-(self.Hk*updateval)

        if adaptive:
            self.Rk = forget * self.Rk + (1 - forget) * (residual * residual.transpose() + self.Hk * cov * HkT)
            self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
            self.Qk = forget * self.Qk + (1 - forget) * self.Kgain * diff * diff.transpose() * self.Kgain.transpose()

        self.Pk = (np.eye(num_variables, num_variables) - (self.Kgain * self.Hk)) * cov


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