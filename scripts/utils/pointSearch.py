import numpy as np
import scipy
from abc import ABC, abstractmethod


class PointSearch(ABC):
    def __init__(self, points: np.array):
        self.points = points

    @abstractmethod
    def findNeighbors(self, centerIndex: int, epsilon: float) -> np.array:
        pass


class DistanceMatrixSearch(PointSearch):
    def __init__(self, points: np.array):
        super().__init__(points)
        self.distanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.points, 'euclidean'))

    def findNeighbors(self, centerIndex, epsilon):
        return np.where(self.distanceMatrix[centerIndex] < epsilon)[0]


class KDTreeSearch(PointSearch):
    def __init__(self, points: np.array):
        super().__init__(points)
        self.tree = scipy.spatial.KDTree(points)

    def findNeighbors(self, centerIndex, epsilon):
        return self.tree.query_ball_point(self.points[centerIndex], epsilon)
