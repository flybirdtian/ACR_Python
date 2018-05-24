
from Camera.CameraBase import *
from MotionPlatform.PlatformBase import PlatformBase
from RelativePose.FivePointsAlgorithm import FivePointsAlgorithm_Nghia
from Utils.POSE3 import Pose3
import numpy as np
import os
import cv2
import json


class ACRBisection(object):
    def __init__(self, camera, platform):
        self.camera = camera  # type: CameraBase
        self.platform = platform  # type: PlatformBase
        self.relativePoseAlgorithm = FivePointsAlgorithm_Nghia()

        # damp the computed rotation and translation
        self.dampRatio = 0.8

        # stop condition parameters
        self.stopAngle = 0.1
        self.stop_S = 0.1  # stop translation size s
        self.stopAFD = 0.3
        self.maxStep = 30  # max step number

        self.init_S = 15   # initial translation size s

        # current condition and data
        self.step = 0
        self.curPose = Pose3()
        self.curAFD = 0
        self.cur_S = 15

        # reference image
        self.refImage = np.ndarray([1, 1, 3], np.uint8)
        # dir to storage image
        self.data_dir = ""

        self.lastPose = Pose3()

    def initSettings(self, data_dir, refImage: np.ndarray):
        self.step = 0
        self.curPose = Pose3.from6D([100, 100, 100, 100, 100, 100])  # set initial pose to be a large value
        self.curAFD = 100
        self.cur_S = self.init_S

        self.data_dir = data_dir
        self.refImage = refImage

    def openAll(self):
        self.camera.open()
        self.platform.open()

    @classmethod
    def rtsize(cls, pose: Pose3):
        t, axis, angle = pose.to_t_aixsAngle()
        ts = np.linalg.norm(t, 2)
        return ts, angle

    def stopCondition(self):
        t, angle = self.rtsize(self.curPose)
        if (self.cur_S < self.stop_S and angle < self.stopAngle) \
                or self.curAFD < self.stopAFD or self.step >= self.maxStep:
            return True
        return False

    def computeAFD(cls, match_points_ref, match_points_cur):
        error = match_points_ref - match_points_cur
        error2 = np.linalg.norm(error, 2, axis=1)
        AFD = np.sum(error2) / len(error2)
        return AFD

    @classmethod
    def writeInfo(cls, directory, stepNum, cur_image, cur_image_depth, cur_pose, cur_AFD, cur_S):
        if not os.path.exists(directory):
            os.mkdir(directory)

        img_path = os.path.join(directory, "rgb_{}.png".format(stepNum))
        img_depth_path = os.path.join(directory, "depth_{}.png".format(stepNum))
        cv2.imwrite(img_path, cur_image)
        cv2.imwrite(img_depth_path, cur_image_depth)

        pose_file = os.path.join(directory, "pose_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            t, axis, angle = cur_pose.to_t_aixsAngle()
            pose_dic = {"curPose": cur_pose.toSE3().tolist(), "cur_AFD": cur_AFD, "cur_S": cur_S,
                        "cur_angle": angle, "cur_t": t.tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)

    @classmethod
    def writeHandInfo(cls, directory, stepNum, rots, trans):
        pose_file = os.path.join(directory, "handInfo_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            pose_dic = {"trans": trans.tolist(), "rots": rots.tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)


    def normalized(cls, x):
        return x / np.linalg.norm(x)

    def poseWithScale(cls, pose: Pose3, s: float):
        R, t = pose.toRt()
        t_norm = cls.normalized(t)
        ts = t_norm * s

        return Pose3.fromRt(R, ts)

    def dumpPose(self, pose):
        motion = pose.toCenter6D()
        rots = np.array(motion[3:]) * self.dampRatio   # sequence: rz, ry, rx
        trans = np.array(motion[:3]) * self.dampRatio  # sequence: tx, ty, tz
        return Pose3.fromCenter6D(np.append(trans, rots))

    def relocation(self):
        while True:
            image, image_depth = self.camera.getImage()
            pose, match_points_ref, match_points_cur = \
                self.relativePoseAlgorithm.getPose(self.refImage, image, self.camera.cameraCalibration.getK())

            # Bisection
            R, t = pose.toRt()
            lastR, lastT = self.lastPose.toRt()
            if np.dot(t, lastT) < -1e-6:
                self.cur_S /= 2.0

            self.lastPose = pose.copy()
            self.curPose = self.poseWithScale(pose, self.cur_S)
            self.curAFD = self.computeAFD(match_points_ref, match_points_cur)
            self.writeInfo(self.data_dir, self.step, image, image_depth, pose, self.curAFD, self.cur_S)

            if self.stopCondition():
                break

            # moving platform
            dumpMotion = self.dumpPose(self.curPose.inverse())
            self.platform.movePose(dumpMotion)
            self.step = self.step + 1


def test():
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3().from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa

    #X = Pose3.fromCenter6D([0, 0, 0, 1, 2, 3])
    X = Pose3.fromCenter6D([0, 0, 0, 0, 0, 0])
    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()

    ref_image, ref_image_depth = myACR.camera.getImage()

    pose = Pose3.fromCenter6D([10, 0, 8, 1.2, 0, -1.3])
    platform.movePose(movingPose=pose)

    directory = "D:/temp/acr"
    if not os.path.exists(directory):
        os.mkdir(directory)

    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)

    myACR.initSettings(data_dir=directory, refImage=ref_image)
    # input('Press to continue...')
    myACR.relocation()


if __name__ == "__main__":
    test()

