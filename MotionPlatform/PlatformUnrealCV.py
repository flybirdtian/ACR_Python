import numpy as np
from MotionPlatform.PlatformBase import PlatformBase
from Utils.POSE3 import Pose3
from UnrealCVBase.UnrealCVEnv import UnrealCVEnv


class PlatformUnrealCV(PlatformBase):
    def __init__(self, unreal_env: UnrealCVEnv, X: Pose3=Pose3()):
        super(PlatformBase, self).__init__()
        self.unrealCvEnv = unreal_env
        self.X = X  # hand eye relative pose

    def open(self) -> bool:
        self.unrealCvEnv.connect()
        return self.unrealCvEnv.isConnected()

    def close(self) -> bool:
        self.unrealCvEnv.disconnect()
        return not self.unrealCvEnv.isConnected()

    def rotate(self, pitch, yaw, roll) -> bool:
        self.unrealCvEnv.set_rotation(pitch=pitch, yaw=yaw, roll=roll)
        return True

    def translate(self, x, y, z)->bool:
        self.unrealCvEnv.set_position(x=x, y=y, z=z)
        return True

    def move(self, pitch, yaw, roll, tx, ty, tz) -> bool:
        rs = self.rotate(pitch, yaw, roll)
        ts = self.translate(tx, ty, tz)
        return rs and ts

    def movePose(self, movingPose: Pose3):
        self.unrealCvEnv.move_relative_pose_eye(movingPose)
        return True

    def autoHorizon(self) -> bool:
        raise NotImplementedError

    def stopAutoHorizon(self) -> bool:
        raise NotImplementedError

    def goHome(self) -> bool:
        self.unrealCvEnv.resetPose()
        return True
