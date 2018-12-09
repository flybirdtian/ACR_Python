from RelativePose.FivePointsAlgorithm import FivePointsAlgorithm_CV
import os
import numpy as np
import math
from Utils.POSE3 import Pose3

def cal5PointsError():
    dir = "D:/Research/OurPapers/2018/PAMI2018_Camera6dRelocation/Review_1st/data/5points_error/4/"
    num = 500
    poseList = []
    poseGTList = []
    PRefList = []
    PCurList = []

    import cv2
    import json

    K = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    fivepoints = FivePointsAlgorithm_CV()

    print("Compute Relative Pose:")
    for i in range(0, num):
        print("Compute Relative Pose {}".format(i+1))
        posefile = dir + "{}_pose.json".format(i+1)
        with open(posefile, 'r') as f:
            info = json.load(f)
            pose_gt_se3 = np.matrix(info["A_GT"])
            pose_gt = Pose3.fromSE3(pose_gt_se3)
            poseGTList.append(pose_gt)

        refImg = cv2.imread(os.path.join(dir, '{}_ref.png'.format(i+1)))
        curImg = cv2.imread(os.path.join(dir, '{}_cur.png'.format(i+1)))
        pose, p_ref, p_cur = fivepoints.getPose(refImg, curImg, K)
        poseList.append(pose)
        PRefList.append(p_ref)
        PCurList.append(p_cur)

    writePose_5Points(dir, poseList)
    writeMatchPoints(dir, PRefList, PCurList)

    angle_list, t_length_list, angle_error_list, t_acos_list = computePoseError(poseGTList, poseList)
    writeError(dir, angle_list, t_length_list, angle_error_list, t_acos_list)
    plotError(angle_list, t_length_list, angle_error_list, t_acos_list)


def writeError(dir, angle_list, t_length_list, angle_error_list, t_acos_list):
    import json
    errorfile = dir + "errors.json"
    with open(errorfile, 'w') as f:
        error_dic = {"angle_list": angle_list, "t_length_list": t_length_list,
                    "angle_error_list": angle_error_list, "t_acos_list": t_acos_list }
        error_json = json.dumps(error_dic, indent=2, separators=(',', ': '))
        f.write(error_json)


def readError(dir):
    import json
    errorfile = dir + "errors.json"
    with open(errorfile, 'r') as f:
        info = json.load(f)
        angle_list = list(info["angle_list"])
        t_length_list = list(info["t_length_list"])
        angle_error_list = list(info["angle_error_list"])
        t_acos_list = list(info["t_acos_list"])
    return angle_list, t_length_list, angle_error_list, t_acos_list


def writePose_5Points(dir, poseList):
    import json
    num = len(poseList)
    for i in range(0, num):
        pose = poseList[i] # type: Pose3
        posefile = dir + "{}_pose_5points.json".format(i + 1)
        with open(posefile, 'w') as f:
            pose_dic = {"A": pose.toSE3().tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)

def readPose_5Points(dir, num):
    poseList = []
    import json
    for i in range(0, num):
        posefile =  dir + "{}_pose_5points_our.json".format(i+1)
        with open(posefile, 'r') as f:
            info = json.load(f)
            se3 = np.matrix(info["A"])
            pose = Pose3.fromSE3(se3)
            poseList.append(pose)
    return poseList

def readPose_gt(dir, num):
    poseGTList = []
    import json
    for i in range(0, num):
        print("Compute Relative Pose {}".format(i+1))
        posefile = dir + "{}_pose.json".format(i+1)
        with open(posefile, 'r') as f:
            info = json.load(f)
            pose_gt_se3 = np.matrix(info["A_GT"])
            pose_gt = Pose3.fromSE3(pose_gt_se3)
            poseGTList.append(pose_gt)
    return poseGTList

def readPose_5Points_t_divide_10(dir, num):
    poseList = []
    import json
    for i in range(0, num):
        posefile =  dir + "{}_pose_5points_our.json".format(i+1)
        with open(posefile, 'r') as f:
            info = json.load(f)
            se3 = np.matrix(info["A"])
            se3[0:3, 3] = se3[0:3, 3]
            # se3[0:3, 3] = se3[0:3, 3] / 10
            pose = Pose3.fromSE3(se3)
            poseList.append(pose)
    return poseList

def readPose_gt_t_divide_10(dir, num):
    poseGTList = []
    import json
    for i in range(0, num):
        print("Compute Relative Pose {}".format(i+1))
        posefile = dir + "{}_pose.json".format(i+1)
        with open(posefile, 'r') as f:
            info = json.load(f)
            pose_gt_se3 = np.matrix(info["A_GT"])
            pose_gt_se3[0:3, 3] = pose_gt_se3[0:3, 3]
            # pose_gt_se3[0:3, 3] = pose_gt_se3[0:3, 3] / 10
            pose_gt = Pose3.fromSE3(pose_gt_se3)
            poseGTList.append(pose_gt)
    return poseGTList

def readPoseAndComputeError_fnorm(dir, num):
    poseList = readPose_5Points_t_divide_10(dir, num)
    poseGTList = readPose_gt_t_divide_10(dir, num)
    angle_list, t_length_list, angle_error_list, t_acos_list = computePoseError(poseGTList, poseList)
    fnorm_list = computeFnorm(poseGTList)
    fnorm_glist, angle_glist, error_glist, error_avg_list, error_std_list = aggregateError_fnorm(fnorm_list, angle_list, angle_error_list)

    plotError2(fnorm_glist, error_avg_list, error_std_list)
    # plotError(angle_list, t_length_list, angle_error_list, t_acos_list)

def readRelativePoseAndComputeError(dir, num):
    poseList = readPose_5Points(dir, num)
    poseGTList = readPose_gt(dir, num)
    angle_list, t_length_list, angle_error_list, t_acos_list = computePoseError(poseGTList, poseList)
    # writeError(dir, angle_list, t_length_list, angle_error_list, t_acos_list)
    angle_glist, error_glist, error_avg_list, error_std_list = aggregateError(angle_list, angle_error_list)

    plotError2(angle_glist, error_avg_list, error_std_list)
    plotError2(list(list(range(50, 0, -1))), error_avg_list, error_std_list)
    plotError(angle_list, t_length_list, angle_error_list, t_acos_list)

def aggregateError_fnorm(fnorm_list, angle_list, angle_error_list):
    angle_glist = []
    error_glist = []
    fnorm_glist = []
    error_avg_list = []
    error_std_list = []

    for i in range(0, 50):
        error_g = []
        angle_g = angle_list[i * 10]
        fnorm_g = fnorm_list[i * 10]
        for j in range(0, 10):
            index = i * 10 + j
            angle = angle_list[index]
            error = angle_error_list[index]
            if error < angle * 2:
                error_g.append(error)
        if len(error_g) > 0:
            error_avg = np.average(error_g)
            error_std = np.std(error_g)

            error_avg_list.append(error_avg)
            error_std_list.append(error_std)

            angle_glist.append(angle_g)
            error_glist.append(error_g)
            fnorm_glist.append(fnorm_g)

    return fnorm_glist, angle_glist, error_glist, error_avg_list, error_std_list

def aggregateError(angle_list, angle_error_list):
    angle_glist = []
    error_glist = []
    error_avg_list = []
    error_std_list = []


    for i in range(0, 50):
        angle_thres = 5 - i * 0.1
        error_g = []
        angle_g = angle_list[i * 10]
        for j in range(0, 10):
            index = i * 10 + j
            angle = angle_list[index]
            error = angle_error_list[index]
            # if error < angle * 2:
            if error < angle_thres * 2:
                error_g.append(error)
        if len(error_g) > 0:
            error_avg = np.average(error_g)
            error_std = np.std(error_g)

            error_avg_list.append(error_avg)
            error_std_list.append(error_std)

            angle_glist.append(angle_g)
            error_glist.append(error_g)

    return angle_glist, error_glist, error_avg_list, error_std_list


def readAndPlotError():
    dir = "D:/Research/OurPapers/2018/PAMI2018_Camera6dRelocation/Review_1st/data/5points_error/5/"
    angle_list, t_length_list, angle_error_list, t_acos_list = readError(dir)
    plotError(angle_list, t_length_list, angle_error_list, t_acos_list)


def writeMatchPoints(dir, PRefList, PCurList):
    import json
    num = len(PRefList)
    for i in range(0, num):
        P_ref = PRefList[i]
        P_cur = PCurList[i]
        pointsfile = dir + "{}_matchpoints.json".format(i + 1)
        with open(pointsfile, 'w') as f:
            points_dic = {"Points_ref": P_ref.tolist(), "Points_cur": P_cur.tolist()}
            points_json = json.dumps(points_dic, indent=2, separators=(',', ': '))
            f.write(points_json)


def length(v):
  return np.linalg.norm(v)


def vector_angle(v1, v2):
  v = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
  return math.degrees(math.acos(v))



def computeFnorm(poseList):
    num = len(poseList)
    fnorm_list = []
    for i in range(0, num):
        pose = poseList[i]  # type:Pose3
        se3_error = pose.toSE3() - Pose3.from6D([0,0,0, 0,0,0]).toSE3()
        fnorm = np.linalg.norm(se3_error)
        fnorm_list.append(fnorm)

    return fnorm_list

def computePoseError(poseGTList, poseList):
    num = len(poseGTList)
    angle_list = []
    t_length_list = []
    angle_error_list = []
    t_acos_list = []

    print("Compute Errors:")
    for i in range(0, num):
        print("Compute Error {}".format(i + 1))
        pose_gt = poseGTList[i]  #type:Pose3
        pose = poseList[i]  #type:Pose3

        tx, ty, tz, euler_z, euler_y, euler_x = pose.to6D()
        tx_gt, ty_gt, tz_gt, euler_z_gt, euler_y_gt, euler_x_gt = pose_gt.to6D()

        t_gt, axis_gt, angle_gt = pose_gt.to_t_aixsAngle()
        t, axis, angle = pose.to_t_aixsAngle()
#        t = [tx, ty, tz]

        #        angle_error_axis = [euler_x-euler_x_gt, euler_y-euler_y_gt, euler_z-euler_z_gt]
        #        angle_error = np.linalg.norm(angle_error_axis)

        # manner1: use pose*(pose^-1)
        pose_error = pose.compose(pose_gt.inverse())
        t_error, axis_error, angle_error = pose_error.to_t_aixsAngle()

        # manner2: use use angle - angle_gt to get error
        # angle_error = abs(angle - angle_gt)


        t_acos = vector_angle(t_gt, t)
        t_gt_length = length(t_gt)

        angle_list.append(angle_gt)
        t_length_list.append(t_gt_length)
        t_acos_list.append(t_acos)
        angle_error_list.append(angle_error)
    return angle_list, t_length_list, angle_error_list, t_acos_list


def plotError(angle_list, t_length_list, angle_error_list, t_acos_list):
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(angle_list, angle_error_list, 'ro')
    plt.show()

    plt.figure(2)
    plt.plot(t_length_list, t_acos_list, 'ro')
    plt.show()


def plotError2(angle_glist, error_avg_list, error_std_list):
    import matplotlib.pyplot as plt

    plt.errorbar(angle_glist, error_avg_list, error_std_list, fmt='r-', ecolor='b', capsize=3)
    plt.show()


if __name__ == "__main__":
    # cal5PointsError()
    # readAndPlotError()
    # testFivePoints_CV()
    directory = "C:/Users/tianf/Phd/Research/our_papers/ACR2PAMI/Review_1/data/5points_error/9/"
    num = 500
    # readRelativePoseAndComputeError(directory, num)
    readPoseAndComputeError_fnorm(directory, num)