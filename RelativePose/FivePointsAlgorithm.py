from Utils.POSE3 import Pose3
from Feature2D.SIFTFeature import *
import os


class FivePointsAlgorithmBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self) -> None:
        pass

    def getPose(self, refImg : np.ndarray, curImg: np.ndarray, K: np.ndarray) -> (Pose3, np.ndarray, np.ndarray):
        pass


class FivePointsAlgorithm_CV(FivePointsAlgorithmBase):
    def __init__(self):
        super(FivePointsAlgorithm_CV, self).__init__()

    def getPose(self, refImg : np.ndarray, curImg: np.ndarray, K: np.ndarray)-> (Pose3, np.ndarray, np.ndarray):
        import cv2
        ref_pts, cur_pts = SIFTFeature.detectAndMatch(image1=refImg,image2=curImg)
        E, mask = cv2.findEssentialMat(ref_pts, cur_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, ref_pts, cur_pts, mask=mask)
        pose = Pose3.fromRt(R, t)
        return pose, ref_pts, cur_pts

class FivePointsAlgorithm_Nghia(FivePointsAlgorithmBase):
    def __init__(self):
        super(FivePointsAlgorithm_Nghia, self).__init__()

        import RelativePose5Point
        self._fivePointsAlg = RelativePose5Point.RelativePose5Point()

    def getPose(self, refImg: np.ndarray, curImg: np.ndarray, K: np.ndarray) -> (Pose3, np.ndarray, np.ndarray):
        ref_pts, cur_pts = SIFTFeature.detectAndMatch(image1=refImg, image2=curImg)  # type: np.ndarray, np.ndarray

        ref_pts_list = list(ref_pts.flatten())
        cur_pts_list = list(cur_pts.flatten())
        K_list = list(K.flatten())

        print(ref_pts[:5])
        print(ref_pts_list[:10])

        R, t = self._fivePointsAlg.calcRP(ref_pts_list, cur_pts_list, K_list)
        R = np.array(R).reshape(3, 3)
        t = np.array(t)

        pose_ref_to_cur = Pose3.fromRt(R, t)
        return pose_ref_to_cur, ref_pts, cur_pts


def testFivePoints_CV():
    print("Test FivePoints Algorithm in OpenCV:")
    fivepoints = FivePointsAlgorithm_CV()
    import cv2
    dir = "D:/Research/OurPapers/2018/PAMI2018_Camera6dRelocation/Review_1st/data/5points_error/2/"
    refImg = cv2.imread(os.path.join(dir, '2_ref.png'))
    curImg = cv2.imread(os.path.join(dir, '2_cur.png'))

    leftK = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    pose, p_ref, p_cur = fivepoints.getPose(refImg, curImg, leftK)

    # pose_gt_se3 = [[0.9999999132538266, -0.00039281740036523473, -0.0001385165311314052, -5.476009513799725],
    # [0.00038960289756899543, 0.9997470051953627, -0.02248941556567579, -16.008759218854912],
    # [0.0001473157209268832, 0.022489359648363134, 0.9997470715139329, -15.42580073522588],
    # [0.0, 0.0, 0.0, 1.0]]
    # pose_gt = Pose3.fromSE3(pose_gt_se3)

    import json
    posefile = dir + "1_pose.json"
    with open(posefile, 'r') as f:
        info = json.load(f)
        pose_gt_se3 = np.matrix(info["A_GT"])
        pose_gt = Pose3.fromSE3(pose_gt_se3)

    print("Pose computed using fivePoints from OpenCV:")
    pose.display()
    print("ground-truth Pose:")
    pose_gt.display()


if __name__ == "__main__":
    testFivePoints_CV()
