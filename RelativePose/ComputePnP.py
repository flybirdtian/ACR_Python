from Utils.POSE3 import Pose3
import subprocess
import os
import json
from Feature2D.SIFTFeature import *

class PnPBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self) -> None:

        self.bin_path = "C:/Code/miaodx/ACR_experiments/openMVG_Pose_Estimation//build/multiview_robust_pnp/Release/openMVG_sample_multiview_robustPnP.exe"
        self.__KPath = r"C:/Code/bird/AFGCD-master/data/ZED_K.txt"

    @classmethod
    def __writeImage(cls, refImg, curImg, curImg_depth):
        cwd = os.getcwd()
        tempDir = os.path.join(cwd, 'pnp_temp')
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)

        refImg_file = os.path.join(tempDir, 'ref.png')
        curImg_file = os.path.join(tempDir, 'cur.png')
        curImg_depth_file = os.path.join(tempDir, 'cur_depth.png')

        import cv2
        cv2.imwrite(refImg_file, refImg)
        cv2.imwrite(curImg_file, curImg)
        cv2.imwrite(curImg_depth_file, curImg_depth)

        return refImg_file, curImg_file, curImg_depth_file

    @classmethod
    def __writeK(cls, K):
        cwd = os.getcwd()
        tempDir = os.path.join(cwd, 'pnp_temp')
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)

        K_path = os.path.join(tempDir, 'K.txt')
        with open(K_path, 'w') as f:
            K_str_for_file = ' '.join(str(k[0]) for k in K.reshape(9, 1))
            f.write(K_str_for_file)

        return  K_path

    def getPose(self, refImg : np.ndarray, curImg: np.ndarray, curImg_depth:np.ndarray, K: np.ndarray) -> (Pose3, np.ndarray, np.ndarray):
        refImg_path, curImg_path, curImg_depth_path = self.__writeImage(refImg, curImg, curImg_depth)
        K_path = self.__writeK(K)

        baseName, ext = os.path.splitext(curImg_path)
        log_path = baseName + '_PnP.json'
        if os.path.isfile(log_path):
            os.remove(log_path)

        p3 = subprocess.Popen(
            [self.bin_path, "--camera_K_file", K_path, "-a", curImg_path, "-b",
             refImg_path, "-l", curImg_depth_path, "--H_filter", "1", "-o", log_path], shell=True,
            stdout=open(baseName + '_PnP.txt', 'w'))
        p3.wait()

        if not os.path.isfile(log_path):
            print ("Something wrong with the estimeation, return zeros")
            return None

        pose = self.loadPose(log_path)
        pose = pose.inverse()

        p_ref, p_cur = self.loadMatchPoints(log_path)

        return pose, p_ref, p_cur

    @classmethod
    def loadPose(cls, json_path):
        with open(json_path, 'r') as f:
            info = json.load(f)
            R = np.array(info['R']).reshape(3, 3)
            t = np.array(info['t'])
            # TODO: moidify PnP, and remove t = t * 1000
            t = t * 1000
            pose = Pose3.fromRt(R,t)

            return pose

    @classmethod
    def loadMatchPoints(cls, json_path):

        with open(json_path, 'r') as f:
            info = json.load(f)
            cur = np.array(info["xL"])
            ref = np.array(info["xR"])

            num = len(cur) // 2
            p_cur = cur.reshape(num,2)
            p_ref = ref.reshape(num,2)

            return p_ref, p_cur


class PnP_CV(PnPBase):
    def __init__(self):
        super(PnP_CV, self).__init__()
        self.bin_path = r"C:/Code/miaodx/ACR_experiments/openMVG_Pose_Estimation/build/multiview_robust_pnp/Release/openMVG_sample_multiview_robustPnP.exe"


    def get3dPoint(cls, imgpt, depth, K):
        imgpt_homo = np.matrix([imgpt[0], imgpt[1], 1]).T
        K_i = np.matrix(K).I
        pt3d = (K_i * imgpt_homo * depth).ravel()
        return pt3d

    def get3dPoints(cls, ref_pts, cur_pts, curImg_depth, K):
        # get valid data
        valid_ref_pts = []
        valid_cur_pts = []
        valid_cur_depth = []
        for i in range(0, len(ref_pts)):
            x = int(cur_pts[i, 0])
            y = int(cur_pts[i, 1])
            if curImg_depth[y, x] == 0: # invalid depth
                continue
            valid_ref_pts.append(ref_pts[i])
            valid_cur_pts.append(cur_pts[i])
            valid_cur_depth.append(curImg_depth[y, x]) # note: first y:height , then x:width

        # conver 2d image points to 3d points
        valid_cur_pts3d = []
        for i in range(0, len(valid_cur_pts)):
            pt3d = cls.get3dPoint(valid_cur_pts[i], valid_cur_depth[i], K)
            valid_cur_pts3d.append(pt3d)

        return np.float32(valid_ref_pts), np.float32(valid_cur_pts), np.float32(valid_cur_pts3d)

    def getPose(self, refImg: np.ndarray, curImg: np.ndarray, curImg_depth: np.ndarray, K: np.ndarray):
        import cv2
        ref_pts, cur_pts = SIFTFeature.detectAndMatch(image1=refImg,image2=curImg)
        ref_pts_2d, cur_pts_2d, cur_pts_3d = self.get3dPoints(ref_pts, cur_pts, curImg_depth, K)
        retval, rvec, tvec, inlier = cv2.solvePnPRansac(cur_pts_3d, ref_pts_2d, K, None, flags=cv2.SOLVEPNP_EPNP)

        if retval == True:
            R, jacobian = cv2.Rodrigues(rvec)
            pose = Pose3.fromRt(R, tvec)
            # pose must be inversed
            pose = pose.inverse()
            return pose, ref_pts_2d, cur_pts_2d
        else:
            return Pose3(), ref_pts_2d, cur_pts_2d


class PnP_MVG(PnPBase):
    def __init__(self) -> None:
        super(PnP_MVG, self).__init__()
        self.bin_path = r"C:/Code/miaodx/ACR_experiments/openMVG_Pose_Estimation/build/multiview_robust_pnp/Release/openMVG_sample_multiview_robustPnP.exe"

def testPnP_CV():
    print("Test PnP in OpenCV:")
    pnp = PnP_CV()
    import cv2
    dir = "D:/Research/OurPapers/2018/PAMI2018_Camera6dRelocation/Review_1st/data/5points_error/2/"
    refImg = cv2.imread(os.path.join(dir, '1_ref.png'))
    curImg = cv2.imread(os.path.join(dir, '1_cur.png'))
    curImg_depth = cv2.imread(os.path.join(dir, '1_cur_depth.png'), cv2.IMREAD_UNCHANGED)

    leftK = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    pose, p_ref, p_cur = pnp.getPose(refImg, curImg, curImg_depth, leftK)


    pose_gt_se3 = [[0.9999999132538266, -0.00039281740036523473, -0.0001385165311314052, -5.476009513799725],
    [0.00038960289756899543, 0.9997470051953627, -0.02248941556567579, -16.008759218854912],
    [0.0001473157209268832, 0.022489359648363134, 0.9997470715139329, -15.42580073522588],
    [0.0, 0.0, 0.0, 1.0]]
    pose_gt = Pose3.fromSE3(pose_gt_se3)

    print("Pose computed using PnP:")
    pose.display()
    print("ground-truth Pose:")
    pose_gt.display()


def testPnP_MVG():
    print("Test PnP in OpenMVG:")
    pnp = PnP_MVG()
    import cv2
    dir = "C:/Code/bird/AFGCD-master/data/unrealcv/sofa_1"
    refImg = cv2.imread(os.path.join(dir, '1_ref.png'))
    curImg = cv2.imread(os.path.join(dir, '1_cur.png'))
    curImg_depth = cv2.imread(os.path.join(dir, '1_cur_depth.png'), cv2.IMREAD_UNCHANGED)

    leftK = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    pose, p_ref, p_cur = pnp.getPose(refImg, curImg, curImg_depth, leftK)

    pose_gt_se3 = [[0.9999999132538266, -0.00039281740036523473, -0.0001385165311314052, -5.476009513799725],
    [0.00038960289756899543, 0.9997470051953627, -0.02248941556567579, -16.008759218854912],
    [0.0001473157209268832, 0.022489359648363134, 0.9997470715139329, -15.42580073522588],
    [0.0, 0.0, 0.0, 1.0]]
    pose_gt = Pose3.fromSE3(pose_gt_se3)

    print("Pose computed using PnP:")
    pose.display()
    print("ground-truth Pose:")
    pose_gt.display()

if __name__ == "__main__":
    # testPnP_MVG()
    testPnP_CV()

