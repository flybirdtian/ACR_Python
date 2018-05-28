
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
        self.stop_S = 0.5  # stop translation size s
        self.stopAFD = 0.3
        self.maxStep = 30  # max step number

        self.init_S = 30   # initial translation size s
        self.init_angle_bound = 5  # initial angle bound

        # current condition and data
        self.step = 0
        self.curPose = Pose3()
        self.curAFD = 100
        self.cur_S = self.init_S
        self.cur_angle_bound = self.init_angle_bound

        # reference image
        self.refImage = np.ndarray([1, 1, 3], np.uint8)
        self.curImage = np.ndarray([1, 1, 3], np.uint8)
        # dir to storage image
        self.data_dir = ""

        self.lastPose = Pose3()

    def initSettings(self, data_dir, refImage: np.ndarray):
        self.step = 0
        self.curPose = Pose3.from6D([100, 100, 100, 100, 100, 100])  # set initial pose to be a large value
        self.curAFD = 100
        self.cur_S = self.init_S
        self.cur_angle_bound = self.init_angle_bound

        self.data_dir = data_dir
        self.refImage = refImage

    def openAll(self):
        self.camera.open()
        self.platform.open()

    def stopCondition(self):
        t, axis, angle = self.curPose.to_t_aixsAngle()
        if (self.cur_S < self.stop_S and angle < self.stopAngle) or self.step >= self.maxStep:
            # or self.curAFD < self.stopAFD
            return True
        return False

    def computeAFD(cls, match_points_ref, match_points_cur):
        error = match_points_ref - match_points_cur
        error2 = np.linalg.norm(error, 2, axis=1)
        AFD = np.sum(error2) / len(error2)
        return AFD


    def writeInfo(self, motion):
        directory = self.data_dir
        stepNum = self.step

        if not os.path.exists(directory):
            os.mkdir(directory)

        img_path = os.path.join(directory, "rgb_{}.png".format(stepNum))
        cv2.imwrite(img_path, self.curImage)

        pose_file = os.path.join(directory, "info_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            t, axis, angle = self.curPose.to_t_aixsAngle()
            t = np.linalg.norm(t)
            t_m, axis_m, angle_m = motion.to_t_aixsAngle()
            t_m = np.linalg.norm(t_m)

            pose_dic = {"step": self.step, "stopAngle":self.stopAngle, "stop_S": self.stop_S, "stopAFD": self.stopAFD,
                        "maxStep": self.maxStep, "AngleBound": self.cur_angle_bound,
                        "cur_AFD": self.curAFD, "cur_angle": angle, "cur_S": self.cur_S,
                        "motion_angle": angle_m, "motion_t": t_m,
                        "cur_Pose": self.curPose.toSE3().tolist(), "Motion_Pose": motion.toSE3().tolist()}
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

    def boundAngle(self, pose):
        t, axis, angle = pose.to_t_aixsAngle()
        if angle > self.cur_angle_bound:
            angle = self.cur_angle_bound

        return Pose3.from_t_axisAngle(t, axis, angle)

    def relocation(self):
        while True:
            image, image_depth = self.camera.getImage()
            pose, match_points_ref, match_points_cur = \
                self.relativePoseAlgorithm.getPose(self.refImage, image, self.camera.cameraCalibration.getK())

            self.curPose = pose.copy()
            self.curAFD = self.computeAFD(match_points_ref, match_points_cur)
            self.curImage = image

            # Bisection
            t, axis, angle = pose.to_t_aixsAngle()
            lastT, lastAxis, lastAngle = self.lastPose.to_t_aixsAngle()
            if np.dot(t, lastT) < -1e-6:
                self.cur_S /= 2.0

            if np.dot(axis, lastAxis) < -1e-6:
                self.cur_angle_bound /= 2.0
                if self.cur_angle_bound < self.stopAngle:
                    self.cur_angle_bound = self.stopAngle

            if self.stopCondition():
                break

            # moving platform
            eye_pose_guess = self.poseWithScale(pose, self.cur_S)
            # dumpMotion = self.dumpPose(eye_pose_guess.inverse())
            motion = eye_pose_guess.inverse()
            motion_bounded = self.boundAngle(motion)
            self.platform.movePose(motion_bounded)

            self.writeInfo(motion_bounded)

            self.lastPose = self.curPose.copy()
            self.step = self.step + 1

        motion = Pose3()
        self.writeInfo(motion)
        return self.step - 1  # last step, no motion happend


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

def test2():
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3.from6D(np.array([0, 1300, 1000, 0, 0, 0]))  # center of the room

    X = Pose3.fromCenter6D([0, 0, 0, 0, 0, 0])
    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()
    pose = Pose3.fromCenter6D([-50, 30, 60, 0.7, -0.5, -0.2])

    for i in range(0, 10):
        directory = r"C:\Users\tianf\Research\temp\data\bigScene\test2\{}".format(i+1)
        if not os.path.exists(directory):
            os.mkdir(directory)

        # reset pose
        myACR.platform.goHome()
        ref_image, ref_image_depth = myACR.camera.getImage()

        img_path = os.path.join(directory, "rgb_ref.png")
        cv2.imwrite(img_path, ref_image)

        myACR.initSettings(data_dir=directory, refImage=ref_image)
        platform.movePose(movingPose=pose)
        myACR.relocation()


if __name__ == "__main__":
    test2()

