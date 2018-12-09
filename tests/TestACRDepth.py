from CameraRelocation.ACRDepth import *
import argparse


def test1():
    from Camera.ZEDCamera import ZEDCamera
    from MotionPlatform.Platform5Axis import Platform5Axis

    camera = ZEDCamera()  # type: CameraBase
    platform = Platform5Axis(url='http://127.0.0.1:10241/')
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    # myACR.platform.autoHorizon()
    # myACR.platform.goHome()
    directory = "C:/Code/bird/AFGCD-master/data/test5"
    if not os.path.exists(directory):
        os.mkdir(directory)
    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    ref_image = cv2.imread(img_path)
    ref_image_depth = cv2.imread(img_depth_path, cv2.IMREAD_UNCHANGED)

    myACR.initSettings(data_dir=directory, refImage= ref_image, refImage_depth=ref_image_depth)

    myACR.relocation()


def test2():

    from Camera.ZEDCamera import ZEDCamera
    from MotionPlatform.Platform5Axis import Platform5Axis

    camera = ZEDCamera()  # type: CameraBase
    platform = Platform5Axis(url='http://127.0.0.1:10241/')

    myACR = ACRD(camera=camera, platform=platform)

    myACR.openAll()

    myACR.camera.setParameters({'GAIN':40, 'EXPOSURE': 100, 'SATURATION':4, 'BRIGHTNESS':4, 'CONTRAST':4})

    # myACR.platform.autoHorizon()
    # myACR.platform.goHome()

    # myACR.platform.translate(10, 8, 11)
    # myACR.platform.rotate(3, 1, 2)

    ref_image, ref_image_depth = myACR.camera.getImage()
    rots = np.array([-2, -1, -3])  # z, y, x
    trans = np.array([15, 0, -15])  # x, y, z

    myACR.platform.translate(trans[0], trans[1], trans[2])
    myACR.platform.rotate(rots[2], rots[1], rots[0])


    directory = "D:/temp/acr"
    if not os.path.exists(directory):
        os.mkdir(directory)

    myACR.writeHandInfo(directory, -1, rots, trans)

    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)

    myACR.initSettings(data_dir=directory, refImage= ref_image, refImage_depth=ref_image_depth)
    input('Press to continue...')
    myACR.relocation()


def test3(data_dir):
    from Camera.UnrealCVCamera import UnrealCVCamera, CameraCalibration
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3().from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa

    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase)
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    ref_image, ref_image_depth = myACR.camera.getImage()

    pose = Pose3.fromCenter6D([30, 20, 10, 1.2, 0, -1.3])
    platform.movePose(movingPose=pose)

    directory = data_dir
    if not os.path.exists(directory):
        os.mkdir(directory)

    # myACR.writeHandInfo(directory, -1, rots, trans)

    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)

    myACR.initSettings(data_dir=directory, refImage=ref_image, refImage_depth=ref_image_depth)
    input('Press to continue...')
    myACR.relocation()


def stress_test():
    myACR = ACRD()
    myACR.openAll()
    import time
    for i in range(20):
        myACR.platform.autoHorizon()
        # print("Going to sleep ...")
        time.sleep(5)
        # print("Sleep done")
        # myACR.platform.goHome()
        myACR.platform.translate(10, 8, 11)
        myACR.platform.rotate(3, 1, 2)
        myACR.platform.translate(-10, -8, -11)
        myACR.platform.rotate(-3, -1, -2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE Rent 3D training')
    parser.add_argument('data_dir', type=str, metavar='S',
                        help='data storage directory')

    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # test1()
    # test2()
    #stress_test()
    test3(args.data_dir)
