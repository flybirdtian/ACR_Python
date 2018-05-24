from Camera.CameraBase import CameraCalibration, CameraBase
from UnrealCVBase.UnrealCVEnv import UnrealCVEnv
from pytypes import override
import typing
import numpy as np


class UnrealCVCamera(CameraBase):

    def __init__(self, unreal_env: UnrealCVEnv, cameraCalib=CameraCalibration()):
        super(UnrealCVCamera, self).__init__()

        self.unrealCvEnv = unreal_env
        self.cameraCalibration = cameraCalib

    def getCameraCalibration(self) -> CameraCalibration:
        return self.cameraCalibration

    @override
    def open(self) -> bool:
        self.unrealCvEnv.connect()
        return self.unrealCvEnv.isConnected()

    @override
    def close(self) -> bool:
        self.unrealCvEnv.disconnect()
        return not self.unrealCvEnv.isConnected()

    @override
    def getImage(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.unrealCvEnv.grab_rgb_and_depth()

    @override
    def getParameters(self) -> dict:
        return {}

    @override
    def setParameters(self, params: dict) -> bool:
        return False

    @override
    def modifyParamByGUI(self):
        return