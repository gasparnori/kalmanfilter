import numpy as np

class conf_ellipse:
    def __init__(self, cov, interval='95%'):
        if interval=='95%':
            s=5.991
        else:
            s=4.605

        print np.linalg.eig(cov)


class init_covariance:
    def __init__(self, measurement):
        self.x=measurement[:]

