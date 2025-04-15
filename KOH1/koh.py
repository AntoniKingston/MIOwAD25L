import numpy as np
import random
from tqdm import tqdm
class KohonenNetwork():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.weights = None

        #functions with respect to x**2
        def gaussian(x):
            return np.exp(-x/2)
        self.gaussian = gaussian
        def sec_gaussian_der(x):
            return gaussian(x)*(1-x)
        self.sec_gaussian_der = sec_gaussian_der
    #setting parameters
    def set_sigma(self, s):
        def gaussian(x):
            return np.exp(-x/(s*2))
        self.gaussian = gaussian
        def sec_gaussian_der(x):
            return 1/s*gaussian(x)*(1-x/s)
        self.sec_gaussian_der = sec_gaussian_der

    def set_lambda(self, l):
        self.l = l

    def coords_with_min_dist(self, vec):
        weights = self.weights
        min_dist = float('inf')
        min_coords = np.zeros(2, dtype=int)
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                if np.sum((weights[i][j] - vec)**2) < min_dist:
                    min_dist = np.sum((weights[i][j] - vec)**2)
                    min_coords = np.array([i, j])
        return min_coords

    def dist_sq(coords1, coords2):
        return np.sum((coords1 - coords2)**2)

    def set_neighborhood(self, name):
        if name == "gaussian":
            self.neighborhood = self.gaussian
        if name == "sec_gaussian_der":
            self.neighborhood = self.sec_gaussian_der


    def learn(self, lr_set, iterations, eta=0.1, random_init=True):
        vec_len = len(lr_set[0])
        if random_init:
            self.weights = [[np.random.randn(vec_len) for j in range(self.m)] for i in range(self.n)]
        step_size = 1
        multiplier = np.exp(-1/self.l)
        for _ in tqdm(range(iterations)):
            vec = random.choice(lr_set)
            if len(vec) != vec_len:
                raise ValueError("Vector length is not constant across learning set")
            coord_closest = self.coords_with_min_dist(vec)
            coord_closest = np.array(coord_closest)
            for k in range(self.n):
                for l in range(self.m):
                    curr_coords = np.array([k,l])
                    upd_vec = self.weights[k][l]
                    theta = self.neighborhood(KohonenNetwork.dist_sq(coord_closest,  curr_coords))
                    self.weights[k][l] = self.weights[k][l] + eta * step_size * theta *(vec - upd_vec)
            step_size *= multiplier

