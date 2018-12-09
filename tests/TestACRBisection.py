from CameraRelocation.ACRBisection import *
import argparse
import os
import cv2
from Utils.POSE3 import Pose3

def test():
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3.from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa

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


def test2(data_dir):
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3().from6D(np.array([0, 1300, 1000, 0, 0, 0]))  # center of the room

    X = Pose3.fromCenter6D([0, 0, 0, 0, 0, 0])
    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()
    pose = Pose3.fromCenter6D([-50, 30, 60, 0.7, -0.5, -0.2])

    for i in range(0, 10):
        directory = os.path.join(data_dir, "{}".format(i+1))
        if not os.path.exists(directory):
            os.makedirs(directory)

        # reset pose
        myACR.platform.goHome()
        ref_image, ref_image_depth = myACR.camera.getImage()

        img_path = os.path.join(directory, "rgb_ref.png")
        cv2.imwrite(img_path, ref_image)

        myACR.initSettings(data_dir=directory, refImage=ref_image)
        platform.movePose(movingPose=pose)
        myACR.relocation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE Rent 3D training')
    parser.add_argument('data_dir', type=str, metavar='S',
                        help='data storage directory')

    args = parser.parse_args()

    test2(args.data_dir)
