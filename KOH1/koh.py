import numpy as np
import random
class KohonenNetwork():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.weights = None
        def gaussi
        self.gaussian =

    def coords_with_min_dist(self, vec):
        weights = self.weights
        min_dist = float('inf')
        min_coords = (0,0)
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                if (weights[i][j] - vec)**2 < min_dist:
                    min_dist = (weights[i][j] - vec)**2
                    min_coords = (i, j)
        return min_coords

    def dist_sq(self, coords1, coords2):
        return (coords1 - coords2)**2

    def set_sigma(self):

    def learn(self, lr_set, iterations, random_init=True):
        vec_len = len(lr_set[0])
        if random_init:
            self.weights = [[np.random.randn(vec_len) for j in range(self.m)] for i in range(self.n)]
        for it in range(iterations):
            vec = random.choice(lr_set)
            if len(vec) != vec_len:
                raise ValueError("Vector length is not constant across learning set")
            i, j = self.coords_with_min_dist(vec)