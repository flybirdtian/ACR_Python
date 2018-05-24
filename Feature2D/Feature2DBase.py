import abc
import numpy as np

class Feature2DBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    @abc.abstractmethod
    def detectAndMatch(cls, image1, image2, mask1=None, mask2=None) -> (np.ndarray, np.ndarray) :
        raise NotImplementedError


