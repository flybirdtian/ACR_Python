from RelativePose.FivePointsAlgorithm import *
import os
import cv2
import json


def readAndComputeRelativePose(directory):
    K = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    five_point = FivePointsAlgorithm_Nghia()
    num = 500
    for i in range(0, num):
        cur_name = os.path.join(directory, "{}_cur.png".format(i+1))
        ref_name = os.path.join(directory, "{}_ref.png".format(i+1))
        cur_image = cv2.imread(cur_name)
        ref_image = cv2.imread(ref_name)
        pose_ref_to_cur, ref_pts, cur_pts = five_point.getPose(ref_image, cur_image, K=K)
        pose_ref_to_cur.toSE3().tolist()

        posefile = os.path.join(directory, "{}_pose_5points_our.json".format(i+1))
        with open(posefile, 'w') as f:
            pose_dic = {"A": pose_ref_to_cur.toSE3().tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)

if __name__ == "__main__":
    leftK = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    directory = "C:/Users/tianf/Phd/Research/our_papers/ACR2PAMI/Review_1/data/5points_error/10/"
    readAndComputeRelativePose(directory)

