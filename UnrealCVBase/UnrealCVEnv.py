import unrealcv
import cv2
import numpy as np
import math
import transforms3d
from Utils.POSE3 import Pose3
import os

class UnrealCVEnv(object):

    def __init__(self, K_arr=np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]]),
                 init_pose: Pose3=Pose3()):
        self.camera_K = np.float32(K_arr)
        self.init_pose = init_pose
        self.client = unrealcv.Client(('127.0.0.1', 9000))
        self.cam_id = 0
        self.cam = dict(position=[0, 0, 0], rotation=[0, 0, 0])  # center 6D representation

        import tempfile
        self.temp_dir = tempfile.mkdtemp()

    def resetPose(self):
        self.set_pose_center6D(*self.init_pose.toCenter6D())

    def isConnected(self) -> bool:
        return self.client.isconnected()

    def disconnect(self):
        self.client.disconnect()
        return self.client.isconnected()

    def connect(self):
        print(self.client.isconnected())
        # raw_input('connected')
        while self.client.isconnected() is False:
            print('Cannot connect to UnrealCV, Please try again')
            self.client.connect()
            print(self.client.isconnected())

        self.set_pose_center6D(*self.init_pose.toCenter6D())

        return self.isConnected()

    """
    Image retrieval
    """

    def read_image(self, viewmode='lit'):
        # temp file name
        file_name = "unrealcv_tmp_image_rgb_" + viewmode + ".png"
        file_path = os.path.join(self.temp_dir, file_name)

        cmd = 'vget /camera/{cam_id}/{viewmode} {save_file}'
        cmd_format = cmd.format(
            cam_id=self.cam_id, viewmode=viewmode, save_file=file_path)
        self.client.request(cmd_format)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

        return image

    @staticmethod
    def _DepthConversion(pointDepth, f):
        """
        Copy from [depth_conversion.py](https://gist.github.com/edz-o/84d63fec2fc2d70721e775337c07e9c9)
        """
        H = pointDepth.shape[0]
        W = pointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
        PlaneDepth = pointDepth / (1 + (DistanceFromCenter / f) ** 2) ** 0.5
        return PlaneDepth

    def read_image_depth(self):

        cmd = 'vget /camera/{cam_id}/depth npy'
        cmd_format = cmd.format(cam_id=self.cam_id)
        res = self.client.request(cmd_format)
        image = _read_npy(res)

        print('Depth image shape:{}, dtype:{}'.format(image.shape,
                                                      np.dtype(image[0][0])))

        image *= 1000.0  # to simulate Kinect format

        image = self._DepthConversion(image.copy(), self.camera_K[0][0])

        image = image.astype(np.uint16)

        return image

    def read_image_depth_convert(self):

        image = self.read_image("depth")  # not the npy way
        image = np.float32(image[:, :, 1])
        print('Depth image shape:{}, dtype:{}'.format(image.shape,
                                                      np.dtype(image[0][0])))

        image *= (1000.0 / 48.5)  # to simulate Kinect format, and depth png scaled back
        image = self._DepthConversion(image.copy(), self.camera_K[0][0])
        image = image.astype(np.uint16)

        return image

    def grab_rgb_and_depth(self, use_convert=False):
        im_rgb = self.read_image("lit")
        if not use_convert:
            im_depth = self.read_image_depth()
        else:
            im_depth = self.read_image_depth_convert()

        return im_rgb, im_depth

    def get_pose_center6D(self):
        pos = self.get_position() + self.get_rotation()
        return np.float32(pos)

    def get_pose(self):
        pose = self.get_position() + self.get_rotation()
        return Pose3.fromCenter6D(pose)

    def get_position(self):
        return self.cam['position']

    def get_rotation(self):
        return self.cam['rotation']

    def set_position(self, x, y, z):
        self.cam['position'] = [float(x), float(y), float(z)]
        cmd = 'vset /camera/{cam_id}/location {x:.3f} {y:.3f} {z:.3f}'
        cmd = cmd.format(
            cam_id=self.cam_id, x=z / 10.0, y=x / 10.0, z=-y / 10.0)
        self.client.request(cmd)

        print("CMD:{}".format(cmd))

    @staticmethod
    def _zyxEulDegree2zxyEulDegree(z_degree, y_degree, x_degree):
        z_rad = math.radians(z_degree)
        y_rad = math.radians(y_degree)
        x_rad = math.radians(x_degree)

        R = transforms3d.euler.euler2mat(z_rad, y_rad, x_rad, 'rzyx')

        zxy_rad = transforms3d.euler.mat2euler(R, 'rzxy')
        zxy_degree = list(map(math.degrees, zxy_rad))

        return np.array(zxy_degree)

    def set_rotation(self, roll, yaw, pitch):
        """
        the input angle sequence is zyx, but for unreal, it is yxz, besides, the angle should minus -1
        :param roll:
        :param yaw:
        :param pitch:
        :return:
        """
        self.cam['rotation'] = [float(roll), float(yaw), float(pitch)]

        roll, pitch, yaw = self._zyxEulDegree2zxyEulDegree(roll, yaw, pitch)

        cmd = 'vset /camera/{cam_id}/rotation {pitch:.3f} {yaw:.3f} {roll:.3f}'
        cmd = cmd.format(cam_id=self.cam_id, pitch=-pitch, yaw=-yaw, roll=-roll)
        print("CMD:{}".format(cmd))
        self.client.request(cmd)

    def set_pose_center6D(self, x, y, z, roll, yaw, pitch):
        self.set_position(x, y, z)
        self.set_rotation(roll, yaw, pitch)

    def set_pose(self, pose: Pose3):
        x, y, z, roll, yaw, pitch = pose.toCenter6D()
        self.set_pose_center6D(x, y, z, roll, yaw, pitch)

    def move_relative_pose_eye(self, RP):
        print("In move_relative_pose_eye, the relative pose is:")
        RP.debug()

        P_CUR = self.get_pose()
        P_TAR = RP.compose(P_CUR)

        tar_pose = P_TAR.toCenter6D()
        print("Target Pose in move eye:{}".format(tar_pose))
        self.set_pose_center6D(*tar_pose)
        return True

    def move_relative_pose_hand(self, RP):
        print("MOVE relative pose with hand in ACRBaseUnreal")

        X_6D = [10, 10, 10, 8.0416, 9.2526, 8.0416]
        XH2E = Pose3.from6D(X_6D)

        Eye = self.get_pose()

        Hand = XH2E.inverse().compose(Eye)
        Hand2 = RP.compose(Hand)
        Eye2 = XH2E.compose(Hand2)

        relative_pose3_for_eye = XH2E.compose(RP).compose(XH2E.inverse())
        Eye2_ = relative_pose3_for_eye.compose(Eye)

        assert np.allclose(Eye2.rotation(), Eye2_.rotation())
        assert np.allclose(Eye2.center(), Eye2_.center())

        tar_pose_move_hand = Eye2.toCenter6D()

        self.set_pose_center6D(*tar_pose_move_hand)
        return True


def _read_npy(res):
    try:  # python2
        from StringIO import StringIO
        return np.load(StringIO(res))
    except ImportError:
        from io import BytesIO
        return np.load(BytesIO(res))


if __name__ == "__main__":
    initPose = Pose3().from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa

    ins = UnrealCVEnv(init_pose=initPose.toCenter6D())
    ins.connect()
    img_rgb, img_depth = ins.grab_rgb_and_depth()
    cv2.imwrite("RGB_before.png", img_rgb)
    cv2.imwrite("depth_before.png", img_depth)
    #
    # cv2.imshow("RGB_before", img_rgb)
    # cv2.imshow("depth_before", img_depth)

    ins.move_relative_pose_eye(Pose3.fromCenter6D([10, 10, 10, 0.3, 0.2, 0.1]))
    img_rgb, img_depth = ins.grab_rgb_and_depth()
    cv2.imwrite("RGB_after.png", img_rgb)
    cv2.imwrite("depth_after.png", img_depth)

    # cv2.imshow("RGB_after", img_rgb)
    # cv2.imshow("depth_after", img_depth)
