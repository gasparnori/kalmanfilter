import numpy as np

class Filter():

    forget_R = 0.3  # forgetting factor
    forget_Q = 0.3

    def __init__(self, missing_num, num_variables):
        self.init(missing_num, num_variables)

    def init(self,  missing_num, num_variables):
        deltaT=5
        self.active=False

        self.calibrationCounter=0
        self.calibNum=10
        self.Rcalib=np.zeros(shape=(num_variables, self.calibNum)) #saves 100 points in the same location

        #self.calibrationDiff=np.zeros(shape=(num_variables, 100))# saves the last 100 measurements for initialization

        self.missing_num=missing_num #number of missing variables to change
        self.num_variables=num_variables #number of variables

        self.missing_counter=0
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
        self.Pk = np.eye(num_variables, num_variables)*0.1  # np.eye(num_variables, num_variables)
        # Rk: observation covariance
        # self.measurement_covariance = np.eye(4, 4)
        self.Rk = np.eye(num_variables, num_variables) * 10  # estimate of measurement variance, change to see effect
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

        # for the filter
        self.measured_state = np.zeros(shape=(num_variables, 1))
        # a FIFO
        self.updated_state = np.zeros(shape=(num_variables, 100))

    def calibrateR(self, x, y, t):
        # measurements in a fixed position: should determine the sensor error
        if self.calibrationCounter<self.calibNum:
            print "calibrating R...", self.calibNum-self.calibrationCounter
            if self.calibrationCounter>0:
                vx = (x - self.Rcalib[0, self.calibrationCounter-1]) / t  # px/usec
                vy = (y - self.Rcalib[1, self.calibrationCounter-1]) / t  # px/usec
            else:
                vx=0
                vy=0
            if self.num_variables == 4:
                self.Rcalib[:, self.calibrationCounter]=[x,y,vx,vy]

            if self.num_variables == 6:
                if self.calibrationCounter > 1:
                    ax = (vx - self.Rcalib[2, self.calibrationCounter-1]) / t  # px/usec^2
                    ay = (vy - self.Rcalib[3, self.calibrationCounter-1]) / t  # px/usec^2
                else:
                    ax=0
                    ay=0
                self.Rcalib[:, self.calibrationCounter] = [x, y, vx, vy, ax, ay]
            self.calibrationCounter = self.calibrationCounter + 1
            return True
        else:
            print "now calculate..."
            self.Rk=np.cov(self.Rcalib[:, 2:])
            self.calibrationCounter=0
            print self.Rk
            return False


    def calibrateP(self):
        print "calibratingQ"

    def resetFilter(self):
        self.init(self.missing_num, self.num_variables)

    def startFilter(self, initial_state):
        self.resetFilter()
        self.initCounter=0
        self.active=True
        self.updated_state[:, -1] = initial_state


    def add_updated(self, p):
        self.updated_state[:, :-1] = self.updated_state[:, 1:]
        self.updated_state[:, -1] = p.A1
        self.measured_state = self.updated_state[:, -1]
        self.measured_state.shape = (self.num_variables, 1)

    def add_measurement(self, x, y, t):
        if t > 0:
            #if we need to do a velocity calculation after a missing point
            if self.measured_state is None:
                self.measured_state = self.updated_state[:, -1]
                self.measured_state.shape = (self.num_variables, 1)

            vx = (x - self.measured_state[0, 0]) / t  # px/usec
            vy = (y - self.measured_state[1, 0]) / t  # px/usec
            if self.num_variables == 6:
                ax = (vx - self.measured_state[2, 0]) / t  # px/usec^2
                ay = (vy - self.measured_state[3, 0]) / t  # px/usec^2
            # print "ax, ay calculation:", ax, ay
        else:
            vx = (self.measured_state[2, 0])
            vy = (self.measured_state[3, 0])
            if self.num_variables == 6:
                ax = (self.measured_state[4, 0])
                ay = (self.measured_state[5, 0])

        if self.num_variables == 4:
            self.measured_state = np.array([[x], [y], [vx], [vy]])
        else:
            self.measured_state = np.array([[x], [y], [vx], [vy], [ax], [ay]])

    def iterate_filter(self, dt, adaptive=False, predicting=True):
        # print dt
        if self.measured_state is None:
            m=None
        else:
            m = np.asmatrix(self.measured_state)
        u = np.asmatrix(self.updated_state[:, -1]).transpose()
        HkT=self.Hk.transpose()
        if self.num_variables == 4:
            self.Fk = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            self.Fk = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                                [0, 1, 0, dt, 0, 0.5*dt*dt],
                                [0, 0, 1, 0, dt, 0],
                                [0, 0, 0, 1, 0,dt],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])

        pred = self.Fk * u
        cov  = self.Fk * self.Pk * self.Fk.transpose() + self.Qk  # Qk-1: Q always a step behind

        noData = True
        if m is not None and pred is not None:
            #if m[0] is not None:
           # if abs((m-pred)[0,0])<100 and abs((m-pred)[1,0])<100:
                noData=False

        if noData is False:
            self.missing_counter=0
            innovation = m - self.Hk * pred
            residual = m - (self.Hk * (pred + self.Kgain * innovation))
            if adaptive:
                self.Rk = self.forget_R * self.Rk + (1 - self.forget_R) * (residual * residual.T)#+ (self.Hk * cov * HkT))
            self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
            self.Pk = (np.eye(self.num_variables, self.num_variables) - (self.Kgain * self.Hk)) * cov
            if adaptive:
                self.Qk = self.forget_Q * self.Qk + (1 - self.forget_Q) * self.Kgain * innovation * innovation.T * self.Kgain.T

            updateval = pred + self.Kgain * innovation
            #print "Data", updateval.shape
            self.add_updated(updateval)
            return (m[0:2].A1, updateval[0:2].A1)

        else:
            if self.missing_counter<self.missing_num:
                self.missing_counter=self.missing_counter+1
                self.Kgain = cov * HkT * np.linalg.inv(self.Hk * cov * HkT + self.Rk)
                self.Pk = (np.eye(self.num_variables, self.num_variables) - (self.Kgain * self.Hk)) * cov
                updateval = pred
               #  "noData", updateval.shape
                self.add_updated(updateval)
                return ([0,0],  updateval[0:2].A1)
            #too many values were missing already: need to reset or recalibrate
            else:
                self.missing_counter=0
                updateval=None
                self.resetFilter()
                print "reset"
                return ([0,0],[0,0])
