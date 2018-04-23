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
        self.ObservationCoeff=QLabel('Observation coefficient (blue line) value: ', self)
        self.ObservationCoeffVal = QSlider(Qt.Horizontal, self)
        self.ObservationCoefftickMin = QLabel('0.5', self)
        self.ObservationCoefftickMax = QLabel('20', self)

        self.layout.addWidget(self.smoothbtn, 0, 0)
        self.layout.addWidget(self.predictbtn, 0, 1)
        self.layout.addWidget(self.ObservationCoeff, 1, 0)
        self.layout.addWidget(self.ObservationCoeffVal, 1, 1)
        self.layout.addWidget(self.ObservationCoefftickMin, 2, 1)
        self.layout.addWidget(self.ObservationCoefftickMax, 2, 3)
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

        #mask: adding missing values
        self.mask_list=np.sort(random.sample(range(1, 2000), 20))
        self.active_mask=0

        deltaT=0.5
        #Fk: transition matrix
        self.Fk =np.array( [[1, 0, deltaT, 0],[0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
        #Hk:observation matrix
        self.Hk =np.eye(4, 4) #not going to change
        #Pk: transition covariance
        self.Pk=np.eye(4, 4)
        self.Pk2 = np.eye(4, 4)
        #Rk: observation covariance
        self.Rk=np.eye(4,4)*20
        #self.measurement_covariance = np.eye(4, 4)
        #self.R = 5  # estimate of measurement variance, change to see effect
        # Q
        #self.Q = 0.1  # process variance

        self.kf = KalmanFilter(transition_matrices=self.Fk,
                               observation_matrices=self.Hk,
                               transition_covariance=self.Pk,
                               observation_covariance=self.Rk,
                               random_state=0)

        #self.kf2=KalmanFilter(transition_matrices=self.Fk,
        #                       observation_matrices=self.Hk)

    def smoothing(self):
        self.predictbtn.setChecked(False)
        self.smooth_trajectories()

    def onlinepred(self, state):
        if state:
            print "online predicton"

    def initUI(self):
        self.setGeometry(300, 300, 600, 600)

        self.layout.setAlignment(Qt.AlignTop)
        self.smoothbtn.clicked.connect(self.smoothing)
        self.predictbtn.clicked.connect(self.onlinepred)
        self.ObservationCoeffVal.valueChanged.connect(self.valuechange)
        self.ObservationCoeffVal.setMinimum(0.5)
        self.ObservationCoeffVal.setMaximum(20)
        self.ObservationCoeffVal.setValue(15)
        self.ObservationCoeffVal.setTickPosition(QSlider.TicksBelow)
        self.ObservationCoeffVal.setTickInterval(0.5)
        self.predictbtn.setCheckable(True)
        self.setWindowTitle('Points')
        self.show()

    def valuechange(self):
        s='Current Observation coefficient value: '+str(self.ObservationCoeffVal.value())
        self.ObservationCoeff.setText(s)
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

            qp.setBrush(Qt.green)
            if self.active_mask>0:
                [qp.drawEllipse(self.updated_state[m][0], self.updated_state[m][1], 15, 15) for m in self.mask_list[0:self.active_mask-1]]
        qp.end()

    def addPoints(self, x,y):
        mu, sigma = 0, 10  # mean and standard deviation
        datax = int(x + np.random.normal(mu, sigma, 1))
        datay = int(y + np.random.normal(mu, sigma, 1))
        t = self.timer.elapsed()
        # print t
        self.timer.restart()
        if t>0:
            vx= (datax-self.updated_state[-1][0])/t
            vy= (datay-self.updated_state[-1][1])/t
        else:
            vx=1
            vy=1

        #an index to be masked
        if len(self.measured_state)!=self.mask_list[self.active_mask]:
            self.measured_state.append([datax, datay, vx, vy])
            self.points_measured.append([datax, datay])
            if self.predictbtn.isChecked() and len(self.measured_state) > 5:
                self.trajectories_online(t)
                #self.estimated_trajectory()


        else:
            if self.predictbtn.isChecked() and len(self.measured_state) > 5:
                print self.mask_list[self.active_mask]
                self.measured_state.append(self.measured_state[-1])
                self.trajectories_online(t)
                #self.estimated_trajectory()
                self.active_mask=self.active_mask+1



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


    def mouseReleaseEvent(self, mouse_event):
        print "stop"
        print "smooth line"
        #self.predict_all_trajectories()
        #print self.points

    def smooth_trajectories(self):
        kf = KalmanFilter(transition_matrices=self.Fk, observation_matrices=self.Hk)
        measurements = np.asarray(self.measured_state)
        kf = kf.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
        [self.updated_state.append(u) for u in smoothed_state_means]
        self.update()

    def trajectories_online(self, dt):
        if len(self.measured_state)==5:
            measurements = np.asarray(self.measured_state)
            #self.kf2 = self.kf2.em(measurements, n_iter=5)
        self.kf.observation_covariance=np.eye(4,4)*self.ObservationCoeffVal.value()
        #self.kf.transition_matrices=np.array( [[1, 0, dt, 0],[0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        online_means, self.Pk = self.kf.filter_update(self.updated_state[-1], self.Pk, self.measured_state[-1])
        self.updated_state.append(online_means)
       # online_means, self.Pk2 = self.kf2.filter_update(self.predicted_state[-1], self.Pk2, self.measured_state[-1])
       # self.predicted_state.append(online_means)
        self.update()

    #if the next measurement is missing
    def estimated_trajectory(self):
        print "estimating"
       # self.measured_state.append(self.measured_state[-1])
        if len(self.measured_state)>40:
            measurements=self.updated_state[-40:-2]
            measurements.append(self.measured_state[-1])
            measurements = np.asarray(measurements)
        else:
            measurements = self.updated_state[:]
            measurements.append(self.measured_state[-1])
            measurements = np.asarray(measurements)
        (smoothed_state_means, smoothed_state_covariances) = self.kf.smooth(measurements)
        #print "measurements, smoothed states: ", self.measured_state[-1],\
        #    int(smoothed_state_means[-1][0]),\
        #    int(smoothed_state_means[-1][1])
        self.updated_state.append(smoothed_state_means[-1])
        self.update()

def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

mu, sigma = 0, 2 # mean and standard deviation
datax=np.random.normal(mu, sigma, 1000)
print datax