
import abc
import numpy as np
import typing


class CameraCalibration(object):
    def __init__(self, leftK=np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]]),
                 rightK=np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])):
        self.__leftK = np.float32(leftK)
        self.__rightK = np.float32(rightK)

    def getK(self):
        return self.__leftK.copy()

    def getLeftK(self):
        return self.__leftK.copy()

    def getRightK(self):
        return self.__rightK.copy()


class CameraBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cameraCalibration = CameraCalibration()) -> None:
        self.cameraCalibration = cameraCalibration

    def getCameraCalibration(self) -> CameraCalibration:
        return self.cameraCalibration

    @abc.abstractmethod
    def open(self) -> bool:
        return False

    @abc.abstractmethod
    def close(self) -> bool:
        return False

    @abc.abstractmethod
    def getImage(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        rgb_image = np.ndarray(shape=(640, 480), dtype=np.uint8)
        depth_image = np.ndarray(shape=(640, 480), dtype=np.uint8)
        return rgb_image, depth_image

    @abc.abstractmethod
    def getParameters(self) -> dict:
        return {}

    @abc.abstractmethod
    def setParameters(self, params: dict) -> bool:
        return False

    @abc.abstractmethod
    def modifyParamByGUI(self):
        return
