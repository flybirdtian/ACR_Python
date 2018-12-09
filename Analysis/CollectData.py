from Camera.UnrealCVCamera import *
from MotionPlatform.PlatformUnrealCV import *
from UnrealCVBase.UnrealCVEnv import *
from Utils.POSE3 import Pose3
import random
import json


class DataGenerator:
    def __init__(self, init_pose: Pose3 = Pose3(), X = Pose3()):
        self.initPose = init_pose
        self.unreal_env = UnrealCVEnv(init_pose=init_pose)
        self.platform = PlatformUnrealCV(self.unreal_env, X=X)
        self.camera = UnrealCVCamera(self.unreal_env)
        self.unreal_env.connect()

    @classmethod
    def random3Unary(cls):
        rand3 = []
        for i in range(0, 3):
            rand3.append(random.uniform(-1, 1))
        # normlize axis
        unary3 = rand3 / np.linalg.norm(rand3)

        return unary3

    @classmethod
    def generateTranslation(cls):
        poseList = []  # type: list(Pose3)

        t_begin = 50
        t_step = 1
        angle = 0
        for i in range(0, 50):
            for j in range(0, 10):
                t3 = cls.random3Unary() * (t_begin - i * t_step)
                axis = cls.random3Unary()
                pose = Pose3().from_t_axisAngle(t3, axis, angle)
                poseList.append(pose)

        return poseList

    @classmethod
    def generateRotation(cls):
        poseList = []  # type: list(Pose3)

        angle_begin = 5
        angle_step = 0.1
        t = [0, 0, 0]
        for i in range(0, 50):
            for j in range(0, 10):
                angle = angle_begin - i * angle_step
                axis = cls.random3Unary()
                pose = Pose3().from_t_axisAngle(t, axis, angle)
                poseList.append(pose)

        return poseList

    @classmethod
    def generateMotionListLargeToSmallBoth(cls):
        poseList = []  # type: list(Pose3)

        angle_begin = 5
        t_begin = 50
        t_step = 1
        angle_step = 0.1
        for i in range(0, 50):
            for j in range(0, 10):
                t3 = cls.random3Unary() * (t_begin - i * t_step)
                angle = angle_begin - i * angle_step
                axis = cls.random3Unary()
                pose = Pose3().from_t_axisAngle(t3, axis, angle)
                poseList.append(pose)

        return poseList

    @classmethod
    def writeData_simple(cls, dir, listA_GT):
        num = len(listA_GT)
        for i in range(0, num):
            A_GT = listA_GT[i]

            posefile = dir + "{}_pose.json".format(i + 1)
            with open(posefile, 'w') as f:
                pose_dic = {"A_GT": A_GT.toSE3().tolist()}
                pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
                f.write(pose_json)

    @classmethod
    def saveImg(cls, rgb_img, depth_img, directory, base_name):
        rgb_name = base_name + ".png"
        depth_name = base_name + "_depth.png"
        path_rgb = os.path.join(directory, rgb_name)
        path_depth = os.path.join(directory, depth_name)
        cv2.imwrite(path_rgb, rgb_img)
        cv2.imwrite(path_depth, depth_img)

    def moveCameraAndRecordImage(self, listA_GT, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        num = len(listA_GT)
        for i in range(0, num):
            self.platform.goHome()
            base_name = "{}_ref".format(i+1)
            rgb_img, depth_img = self.camera.getImage()
            self.saveImg(rgb_img, depth_img, directory, base_name)

            A_GT = listA_GT[i]
            self.platform.movePose(A_GT)

            base_name = "{}_cur".format(i+1)
            rgb_img, depth_img = self.camera.getImage()
            self.saveImg(rgb_img, depth_img, directory, base_name)


if __name__ == "__main__":
    directory = "C:/Users/tianf/Phd/Research/our_papers/ACR2PAMI/Review_1/data/5points_error/10/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    initPose = Pose3().from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa
    X = Pose3()
    data_generator = DataGenerator(initPose, X=X)
    listA_GT = data_generator.generateMotionListLargeToSmallBoth()
    # listA_GT = data_generator.generateTranslation()
    # listA_GT = data_generator.generateRotation()
    data_generator.writeData_simple(directory, listA_GT)
    data_generator.moveCameraAndRecordImage(listA_GT, directory)


