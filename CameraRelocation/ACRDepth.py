
from Camera.CameraBase import CameraBase
from MotionPlatform.PlatformBase import PlatformBase
from RelativePose import ComputePnP as pnp
from Utils.POSE3 import Pose3
import numpy as np
import os
import cv2
import json


class ACRD(object):
    def __init__(self, camera, platform):
        self.camera = camera  # type: CameraBase
        self.pnp = pnp.PnP_CV()
        self.platform = platform  # type: PlatformBase

        # damp the computed rotation and translation
        self.dampRatio = 1

        # stop condition parameters
        self.stopAngle = 0.1
        self.stopTrans = 0.2
        self.stopAFD = 0.3
        self.maxStep = 20

        # current condition and data
        self.step = 0
        self.curPose = Pose3()
        self.curAFD = 0

        # reference image
        self.refImage = np.ndarray([1, 1, 3], np.uint8)
        self.refImage_depth = np.ndarray([1, 1, 3], np.uint16)
        # dir to storage image
        self.data_dir = ""

    def initSettings(self, data_dir, refImage: np.ndarray, refImage_depth: np.ndarray):
        self.step = 0
        self.curPose = Pose3.from6D([100, 100, 100, 100, 100, 100])  # set initial pose to be a large value
        self.curAFD = 100

        self.data_dir = data_dir
        self.refImage = refImage
        self.refImage_depth = refImage_depth

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
        if (t < self.stopTrans and angle < self.stopAngle) or self.curAFD < self.stopAFD or self.step >= self.maxStep:
            return True
        return False

    def computeAFD(cls, match_points_ref, match_points_cur):
        error = match_points_ref - match_points_cur
        error2 = np.linalg.norm(error,2, axis=1)
        AFD = np.sum(error2) / len(error2)
        return AFD

    def dumpPose(self, pose):
        motion = pose.toCenter6D()
        rots = np.array(motion[3:]) * self.dampRatio   # sequence: rz, ry, rx
        trans = np.array(motion[:3]) * self.dampRatio  # sequence: tx, ty, tz
        return Pose3.fromCenter6D(np.append(trans, rots))

    @classmethod
    def writeInfo(cls, directory, stepNum, cur_image, cur_image_depth, cur_pose, cur_AFD, cur_image_warp, cur_image_depth_warp):
        if not os.path.exists(directory):
            os.mkdir(directory)

        img_path = os.path.join(directory, "rgb_{}.png".format(stepNum))
        img_depth_path = os.path.join(directory, "depth_{}.png".format(stepNum))
        cv2.imwrite(img_path, cur_image)
        cv2.imwrite(img_depth_path, cur_image_depth)

        pose_file = os.path.join(directory, "pose_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            pose_dic = {"curPose": cur_pose.toSE3().tolist(), "cur_AFD": cur_AFD}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)

        img_warp_path = os.path.join(directory, "rgb_warp_{}.png".format(stepNum))
        img_depth_warp_path = os.path.join(directory, "depth_warp_{}.png".format(stepNum))
        cv2.imwrite(img_warp_path, cur_image_warp)
        cv2.imwrite(img_depth_warp_path, cur_image_depth_warp)

    @classmethod
    def writeHandInfo(cls, directory, stepNum, rots, trans):
        pose_file = os.path.join(directory, "handInfo_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            pose_dic = {"trans": trans.tolist(), "rots": rots.tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)


    def relocation(self):
        while True:
            image, image_depth = self.camera.getImage()
            pose, match_points_ref, match_points_cur = self.pnp.getPose(self.refImage, image, image_depth,
                                                                        self.camera.cameraCalibration.getK())

            R, t = pose.toRt()
            tmpPose = Pose3.fromRt(R, t)
            image_warp, image_depth_warp, warped_mask = cv2.rgbd.warpFrame(image, np.float32(image_depth),
                                                                           None, tmpPose.inverse().toSE3(),
                                                                           self.camera.cameraCalibration.getK(), None)
            image_depth_warp = np.uint16(image_depth_warp)

            self.curPose = pose.copy()  # type: Pose3
            self.curAFD = self.computeAFD(match_points_ref, match_points_cur)
            self.writeInfo(self.data_dir, self.step, image, image_depth, pose, self.curAFD, image_warp, image_depth_warp)

            if self.stopCondition():
                break

            # moving platform
            movingPose = self.curPose  # type: Pose3
            dumpMotion = self.dumpPose(movingPose)
            self.platform.movePose(dumpMotion.inverse())
            # rots = np.array(motion[3:]) * self.dampRatio  # sequence: rz, ry, rx
            # trans = np.array(motion[:3]) * self.dampRatio * -1  # sequence: tx, ty, tz
            # self.platform.rotate(rots[2], rots[1], rots[0])
            # self.platform.translate(trans[0], trans[1], trans[2])

            # self.writeHandInfo(self.data_dir, self.step, rots, trans)

            self.step = self.step + 1


