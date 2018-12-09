import numpy as np
import matplotlib.pyplot as plt
from Utils.POSE3 import Pose3
from CameraRelocation.ACRBisection import ACRBisection
from Camera.CameraBase import CameraCalibration
import os
import cv2
from UnrealCVBase.UnrealCVEnv import UnrealCVEnv
from CameraRelocation.ACRDepth import ACRD


def oneStepReloclizationWithScale(X: Pose3, dataDir, unrealbase: UnrealCVEnv)-> Pose3:
    '''
    Execute only one step ACR with Known translation scale
    :param X: hand-eye relative pose
    :param dataDir: data directory to store all related data
    :param unrealbase: unrealcv base class
    :return: the eye error after one step relocalization
    '''
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()

    unrealbase.resetPose()
    initPose = unrealbase.get_pose()
    ref_image, ref_image_depth = camera.getImage()

    # Get reference image and set it into ACR
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    img_path = os.path.join(dataDir, "rgb_ref.png")
    cv2.imwrite(img_path, ref_image)
    myACR.initSettings(data_dir=dataDir, refImage=ref_image)

    # move away
    motionPose = Pose3.fromCenter6D([-50, 30, 60, 4, -3, -6])
    scale = np.linalg.norm(motionPose.to6D()[0:3])
    myACR.init_S = scale
    myACR.cur_S = scale

    unrealbase.move_relative_pose_eye(motionPose)

    # set maxStep = 1 and do relocation
    myACR.maxStep = 1
    myACR.relocation()

    # get pose after relocation
    finalPose = unrealbase.get_pose()

    # compute pose errors
    poseError = finalPose.compose(initPose.inverse())

    return poseError

def oneStepReloclization(X: Pose3, dataDir, unrealbase: UnrealCVEnv) -> Pose3:
    '''
    Execute only one step ACR
    :param X: hand-eye relative pose
    :param dataDir: data directory to store all related data
    :param unrealbase: unrealcv base class
    :return: the eye error after one step relocalization
    '''
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()

    unrealbase.resetPose()
    initPose = unrealbase.get_pose()
    ref_image, ref_image_depth = camera.getImage()

    # Get reference image and set it into ACR
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    img_path = os.path.join(dataDir, "rgb_ref.png")
    img_depth_path = os.path.join(dataDir, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)
    myACR.initSettings(data_dir=dataDir, refImage=ref_image)

    # move away
    motionPose = Pose3.fromCenter6D([-50, 30, 60, 0.6, -1, -0.2])
    unrealbase.move_relative_pose_eye(motionPose)

    # set maxStep = 1 and do relocation
    myACR.maxStep = 1
    myACR.relocation()

    # get pose after relocation
    finalPose = unrealbase.get_pose()

    # compute pose errors
    poseError = finalPose.compose(initPose.inverse())

    return poseError


def oneStepRelocalizationDepth(X: Pose3, dataDir, unrealbase: UnrealCVEnv)-> Pose3:
    '''
    Execute only one step ACR
    :param X: hand-eye relative pose
    :param dataDir: data directory to store all related data
    :param unrealbase: unrealcv base class
    :return: the eye error after one step relocalization
    '''
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    unrealbase.resetPose()
    initPose = unrealbase.get_pose()
    ref_image, ref_image_depth = camera.getImage()

    # Get reference image and set it into ACR
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    img_path = os.path.join(dataDir, "rgb_ref.png")
    img_depth_path = os.path.join(dataDir, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)
    myACR.initSettings(data_dir=dataDir, refImage=ref_image, refImage_depth=ref_image_depth)

    # move away
    motionPose = Pose3.fromCenter6D([-50, 30, 60, 0.6, -1, -0.2])
    unrealbase.move_relative_pose_eye(motionPose)

    # set maxStep = 1 and do relocation
    myACR.maxStep = 1
    myACR.relocation()

    # get pose after relocation
    finalPose = unrealbase.get_pose()

    # compute pose errors
    poseError = finalPose.compose(initPose.inverse())

    return poseError


def oneRelocation(X, dataDir, unrealbase) -> int:
    from Camera.UnrealCVCamera import UnrealCVCamera
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV

    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=X)
    myACR = ACRBisection(camera=camera, platform=platform)
    myACR.openAll()

    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    myACR.platform.goHome()
    ref_image, ref_image_depth = myACR.camera.getImage()

    img_path = os.path.join(dataDir, "rgb_ref.png")
    img_depth_path = os.path.join(dataDir, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)
    myACR.initSettings(data_dir=dataDir, refImage=ref_image)

    pose = Pose3.fromCenter6D([-50, 30, 60, 0.6, -1, -0.2])
    platform.movePose(movingPose=pose)

    iterNum = myACR.relocation()

    return iterNum


def randAxis():
    v3 = 2*(np.random.rand(3) - 0.5*np.ones(3))
    return v3/np.linalg.norm(v3)


def writeIteration(dir, iter):
    import json
    file = os.path.join(dir, "iteration.json")
    with open(file, 'w') as f:
        pose_dic = {"iterations": iter}
        pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
        f.write(pose_json)

def writePose(dir, poseErrorList):
    import json
    file = os.path.join(dir, "poseError.json")
    with open(file, 'w') as f:
        pose_dic = {"poseError": poseErrorList}
        pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
        f.write(pose_json)


def GenerateData(baseDir, isXKnown = False):
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3().from6D(np.array([0, 1300, 1000, 0, 0, 0]))
    unrealbase = UnrealCVEnv(init_pose=initPose)

    allIterations = []
    # axis = [1, 1, 1]
    axis = [0, 1, 0]
    axis = axis / np.linalg.norm(axis)
    t_bar = [1, -1, 1]
    t_bar = axis / np.linalg.norm(t_bar)

    for i in range(11, 14):
        iters = []
        if isXKnown:
            angle = 0
            t_s = 0
        else:
            angle = i * 3
            t_s = i * 5

        t = t_bar * t_s
        X = Pose3.from_t_axisAngle(t, axis, angle)
        if isXKnown:
            subDir = os.path.join(baseDir, "I_{}".format(i+1))
        else:
            subDir = os.path.join(baseDir, "Angle_{}_t_{}".format(angle, t_s))
        if not os.path.exists(subDir):
            os.mkdir(subDir)
        for j in range(0, 10):
            subSubDir = os.path.join(subDir, "{}".format(j+1))
            iter = oneRelocation(X, subSubDir, unrealbase)
            writeIteration(subSubDir, iter)
            iters.append(iter)
        writeIteration(subDir, iters)
        allIterations.append(iters)

    writeIteration(baseDir, allIterations)


def GenerateDataOneStepACR(baseDir, isXKnown = False):
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    initPose = Pose3.from6D(np.array([0, 1300, 1000, 0, 0, 0]))
    unrealbase = UnrealCVEnv(init_pose=initPose)

    allPoseErrors = []
    # axis = [1, 1, 1]
    axis = [0, 1, 0]
    axis = axis / np.linalg.norm(axis)
    t_bar = [1, -1, 1]
    t_bar = axis / np.linalg.norm(t_bar)

    for i in range(0, 14):
        poseErrors = []
        if isXKnown:
            angle = 0
            t_s = 0
        else:
            angle = i * 3
            t_s = i * 5

        t = t_bar * t_s
        X = Pose3.from_t_axisAngle(t, axis, angle)
        if isXKnown:
            sub_dir = os.path.join(baseDir, "I_{}".format(i+1))
        else:
            sub_dir = os.path.join(baseDir, "Angle_{}_t_{}".format(angle, t_s))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        for j in range(0, 10):
            subSubDir = os.path.join(sub_dir, "{}".format(j+1))
            # poseErr = oneStepRelocalization(X, subSubDir, unrealbase)
            poseErr = oneStepReloclizationWithScale(X, subSubDir, unrealbase)

            poseError = poseErr.toSE3().tolist()
            writePose(subSubDir, poseError)
            poseErrors.append(poseError)

        writePose(sub_dir, poseErrors)
        allPoseErrors.append(poseErrors)

    writePose(baseDir, allPoseErrors)


def getAvgStd(a2dList):
    num = len(a2dList)
    avgs = []
    stds = []
    for i in range(0, num):
        avg = np.average(a2dList[i])
        std = np.std(a2dList[i])
        avgs.append(avg)
        stds.append(std)
    return avgs, stds

def readPoseErrors(file):
    len1 = 14
    len2 = 10
    all_pose_errors = []
    import json
    with open(file, 'r') as f:
        info = json.load(f)
        allPoseErrors_se3 = info["poseError"]
        for i in range(0, len(allPoseErrors_se3)):
            pose_errors = []
            poseErrors_se3 = allPoseErrors_se3[i]
            for j in range(0, len(poseErrors_se3)):
                poseError_se3 = poseErrors_se3[j]
                pose_error = Pose3.fromSE3(poseError_se3)
                pose_errors.append(pose_error)

            all_pose_errors.append(pose_errors)
    return all_pose_errors


def splitPoseErrors(all_pose_errors):
    all_angle_errors = []
    all_trans_errors = []
    for i in range(0, len(all_pose_errors)):
        pose_errors = all_pose_errors[i]
        angle_errors = []
        trans_errors = []
        for j in range(0, len(pose_errors)):
            pose_error = pose_errors[j]  # type: Pose3
            t, axis, angle = pose_error.to_t_aixsAngle()
            angle_errors.append(angle)
            trans_errors.append(np.linalg.norm(t))

        all_angle_errors.append(angle_errors)
        all_trans_errors.append(trans_errors)
    return all_angle_errors, all_trans_errors

def get_avg_std(all_errors):
    avgs = []
    stds = []

    for i in range(0, len(all_errors)):
        errors = all_errors[i]
        avgs.append(np.average(errors))
        stds.append(np.std(errors))
    return avgs, stds

def evaluate_pose_errors(base_dir, file1, file2):

    all_pose_errors1 = readPoseErrors(file1)
    all_pose_errors2 = readPoseErrors(file2)

    all_angle_errors1, all_trans_errors1 = splitPoseErrors(all_pose_errors1)
    all_angle_errors2, all_trans_errors2 = splitPoseErrors(all_pose_errors2)

    import json
    file = os.path.join(base_dir, "angle_trans_errors.json")
    with open(file, 'w') as f:
        pose_dic = {"angle_error_1": all_angle_errors1, "angle_error_2": all_angle_errors2,
                    "trans_error_1": all_trans_errors1, "trans_error_2": all_trans_errors2}
        pose_json = json.dumps(pose_dic, indent=2, separators=(',', ':'))
        f.write(pose_json)

    angle_avgs1, angle_stds1 = get_avg_std(all_angle_errors1)
    trans_avgs1, trans_stds1 = get_avg_std(all_trans_errors1)

    angle_avgs2, angle_stds2 = get_avg_std(all_angle_errors2)
    trans_avgs2, trans_stds2 = get_avg_std(all_trans_errors2)

    plt.figure(figsize=(9, 6))

    xAngle = np.int32(list(range(0, 14))) * 3
    line1 = plt.errorbar(xAngle, angle_avgs1, angle_stds1, fmt='b-', ecolor='b', capsize=3, label='X=I')
    line2 = plt.errorbar(xAngle, angle_avgs2, angle_stds2, fmt='r-', ecolor='r', capsize=3, label="X=X_GT")

    # line1 = plt.errorbar(xAngle, trans_avgs1, trans_stds1, fmt='b-', ecolor='b', capsize=3, label='X=I')
    # line2 = plt.errorbar(xAngle, trans_avgs2, trans_stds2, fmt='r-', ecolor='r', capsize=3, label="X=X_GT")

    plt.legend()
    plt.ylabel('Errors')
    plt.xlabel('Angle of X')

    plt.show()

def justDraw():
    angle_error_2 = [[2.9994764313394504, 2.8214601907587595, 2.7322276166887582, 2.8037691553148294, 2.7305675533325795,
      2.727871812344496, 2.877198158929578, 2.769384394579418, 2.7985911141384574, 2.7888015567382607],
     [2.881560811902719, 2.8047712749193994, 2.88068939833218, 2.7290232686896383, 2.7280045874852963,
      2.7296402982984107, 2.732334271513467, 2.7311565125500916, 2.9385468919549704, 2.7282230934175047],
     [2.809997266352156, 2.8758339292002377, 2.802394075821235, 2.7441100706102746, 2.789226418832422,
      2.8716839307184387, 2.8445629998597335, 2.753347790172177, 2.727963110702188, 2.8263482650239116],
     [2.7831161482854836, 2.7748923790075866, 2.826004243622175, 2.7852992249285, 2.797015036766275, 2.728216991008643,
      2.7632799747975874, 2.733507064165243, 2.882245890649006, 2.7314753608718436],
     [2.740229177354268, 2.7285386027668186, 2.731901249644288, 2.8755966000870594, 2.7972048885051444,
      2.9253032006323845, 2.7279931758139107, 2.785693448706595, 2.861786328531297, 2.7859590522395705],
     [2.8386058471017988, 2.729272294941452, 2.7356251532333182, 2.7330454065789076, 2.9269586924710556,
      2.838908790168438, 2.792412055833716, 2.76985026348299, 2.7415936516703407, 2.8102479270515013],
     [2.779602606173591, 2.7343895049666385, 2.839121377705936, 2.7966992968616897, 2.77270020949534,
      2.7283423595050653, 2.7280005263799953, 2.7282008004079383, 2.7283714741168437, 2.7995877422794737],
     [2.7282244035077508, 2.8752128795520626, 2.8400775546838024, 2.7279099612080384, 2.9144809560276626,
      2.731265003176488, 2.7310534304438994, 2.7430584087448047, 2.7819371215552295, 2.732006087496766],
     [2.764586716155223, 2.7290100949700284, 2.7329589446738023, 2.779413438494804, 2.940154212511153,
      2.7708982874322863, 2.728151782415709, 2.748743633555649, 2.929784715433482, 2.9214588901545113],
     [2.770824755716959, 2.746713014047658, 2.8157193267956084, 2.7687167538689645, 2.753954504753848,
      2.732551594924901, 2.874366724469888, 2.751346828192952, 2.7286795651236164, 2.7296281987735913],
     [2.7633264016276433, 2.778942589181335, 2.7279527249146915, 2.8112832065103692, 2.732017780580617,
      2.7293825993805356, 2.8472531729262203, 2.7278865460537163, 2.75614332735791, 2.779608958045339],
     [2.840687317192012, 2.7647929276793355, 2.778842780312391, 2.9271418507839138, 2.78341725227683,
      2.7991956836530814, 2.851771657635992, 2.727870143058431, 2.7758886818010753, 2.7288429528908518],
     [2.7313323271581544, 2.7296072804493114, 2.7767420372877734, 2.7670048121908244, 2.732234150375994,
      2.890840872912233, 2.7682546405623505, 2.728528139717377, 2.783372504818955, 2.75922726327595],
     [2.786454974373681, 2.7314695404279434, 2.729568443180061, 2.826311022719904, 2.778319783751722,
      2.8063518369499403, 2.9993784188475994, 2.7329534434259424, 2.7548284525625073, 2.935501554739026]]

    trans_error_1 = [[30.902781960925275, 20.661042783634375, 43.24978741314374, 28.36353641010028, 20.176374747915112,
      27.440048384644946, 33.911451541831674, 29.845518640479476, 41.05329968120322, 4.719344708836807],
     [41.41544245969542, 11.723233996414463, 30.679663154194586, 59.861269027042034, 7.800075528045997,
      6.582129077994464, 32.99515753953987, 15.634303960894359, 4.750155095646155, 35.49066481994098],
     [11.831107635413849, 19.188664408569704, 40.52741505955482, 41.57838122303697, 26.709418003938804,
      36.031645113573894, 21.253973645762215, 32.75326573732428, 30.521546743001306, 66.40677852027855],
     [29.71261092668514, 29.679857720746433, 32.10774781489594, 47.42585268856125, 28.172977140179164,
      14.904326166962464, 18.547820121471, 20.718816676444494, 48.90227690520146, 64.71839924994572],
     [29.539563463546955, 27.770730354428856, 27.284989616078597, 28.957890077877614, 71.64780236390698,
      23.09977106698534, 30.957520955383014, 28.99413816816566, 24.581871102107126, 34.254254576257814],
     [72.98605471525212, 74.84818738205134, 16.198773753558168, 37.10139864118313, 25.032988580744895,
      33.95861568910774, 22.95396647605655, 19.681503220599915, 78.60227437510963, 18.62310251754925],
     [17.91219904898449, 26.066663168984086, 28.571405162488908, 74.22995316921848, 25.84693659870863,
      22.264293565775617, 33.79724232565303, 18.1853429739283, 29.401252226945218, 27.815313322010102],
     [31.438034786722582, 24.837279743739145, 42.51436959016147, 32.39598681569398, 23.457934840687113,
      32.76266756770356, 28.250771654296305, 37.14206213165172, 37.53090647397108, 25.87089438145009],
     [31.258735661852157, 22.138595505673518, 46.832700015747456, 36.25405819286926, 26.407582668457803,
      41.39745999517385, 31.01789546910383, 25.677102565320126, 33.445357773608734, 22.143936528805106],
     [85.48579072619626, 46.62983460834158, 37.52351527560615, 54.0722870027655, 43.58208590830032, 32.13193446464614,
      31.550743869312498, 52.9337135241139, 47.975304443056295, 37.0675595283099],
     [36.71627804333358, 37.48458686708871, 36.318982250163245, 38.17770816376737, 34.52639365692599, 45.66919719353302,
      88.87329455102446, 34.888871770472385, 54.39805445371336, 40.092989454066426],
     [43.02820173334844, 103.36033344561471, 41.83684368061312, 43.07601777255861, 37.59181350341836, 41.97312202783088,
      43.93286452110697, 36.191918509845884, 43.63515433698545, 37.676945850418534],
     [52.05875425948244, 91.06821502608271, 52.12736357883103, 94.72598178732764, 42.3016758755007, 46.332773083736555,
      53.2081971763068, 51.828930621355774, 49.91402756774323, 53.613164359991885],
     [48.72114112709297, 97.36990550576934, 58.10384452775389, 49.90044478492009, 56.193962239301406, 48.87630984908374,
      47.73763673442769, 68.55173513313436, 57.27830270357221, 113.94171925034675]]



def drawSomething():

    # original data
    # xAngle = [10.125, 20.556, 26.565, 32.005, 36.870]
    # X_I = [[13, 13, 12, 11, 13, 11], [13, 13, 15, 10, 13, 13], [15, 15, 12, 14, 15, 14], [13, 15, 21, 16, 18], [15, 16, 14, 21, 17, 18]]
    # X_I_2 = [[15, 15, 13, 14, 15, 13], [16, 15, 17, 11, 15, 15], [17, 16, 15, 17, 18, 17], [15, 17, 25, 19, 22], [18, 20, 18, 24, 20, 21]]
    # X_E = [[12, 12, 12, 13, 11, 13], [9, 9, 9, 10, 9, 12],     [8, 12, 8, 10, 8, 9],   [11, 15, 12, 12, 12, 12], [10, 12, 11, 9, 10, 11]]
    # X_E_2 = [[14, 15, 14, 14, 12, 15], [11, 10, 11, 11, 12, 14], [9, 14, 9, 12, 10, 11], [12, 16, 14, 14, 13, 14], [11, 14, 12, 10, 13, 14]]

    # For real platform
    X_I = [[15, 15, 13, 14, 15, 13], [16, 15, 11, 15, 15], [17, 16, 15, 17, 18, 17], [15, 17, 25, 19, 22], [18, 20, 18, 24, 20, 21]]
    X_E = [[12, 12, 12, 13, 11, 13], [11, 10, 11, 11, 12, 14], [9, 14, 9, 12, 10, 11], [12, 14, 14, 13, 14], [11, 14, 12, 10, 13, 14]]


    # original data test2
    # X_I = [[13, 11, 15, 13, 14, 16, 13, 12, 11, 16], [14, 18, 15, 16, 14, 14, 19, 10, 10, 16],
    #  [15, 12, 12, 16, 18, 12, 13, 15, 12, 12], [14, 13, 15, 11, 14, 12, 14, 14, 13, 14],
    #  [21, 15, 15, 16, 14, 13, 13, 15, 23, 17], [10, 16, 12, 13, 11, 19, 15, 13, 11, 15],
    #  [11, 15, 13, 14, 17, 12, 11, 9, 16, 15], [20, 18, 15, 15, 14, 14, 18, 15, 18, 10],
    #  [16, 11, 14, 16, 14, 16, 13, 10, 18, 14], [19, 20, 16, 19, 15, 17, 17, 16, 16, 15],
    #  [20, 20, 18, 18, 14, 21, 17, 15, 15, 13], [13, 21, 23, 11, 14, 17, 18, 18, 18, 19],
    #  [15, 13, 18, 14, 16, 12, 22, 15, 17, 15], [14, 18, 21, 17, 21, 20, 14, 20, 15, 18]]
    #
    # X_E = [[10, 13, 13, 22, 12, 15, 14, 16, 19, 14], [17, 17, 15, 21, 15, 13, 12, 23, 15, 9],
    #  [18, 15, 10, 16, 13, 15, 14, 24, 12, 12], [16, 16, 14, 12, 15, 24, 16, 11, 13, 16],
    #  [18, 18, 14, 15, 18, 11, 20, 15, 14, 12], [16, 11, 14, 14, 9, 11, 15, 17, 10, 15],
    #  [11, 11, 15, 13, 16, 17, 11, 16, 16, 14], [11, 11, 17, 17, 13, 17, 13, 15, 15, 10],
    #  [13, 13, 16, 12, 12, 12, 12, 14, 13, 20], [14, 13, 13, 11, 11, 20, 15, 15, 11, 12],
    #  [10, 12, 11, 19, 15, 14, 15, 17, 12, 12], [14, 16, 17, 11, 13, 16, 16, 13, 16, 13],
    #  [13, 13, 12, 18, 14, 12, 15, 15, 13, 14], [18, 23, 15, 15, 13, 18, 14, 11, 14, 14]]

    # data without noise test2-1
    # X_I = [[13, 11, 15, 13, 14, 13, 12, 11], [14, 15, 16, 14, 14, 10, 10, 16],
    #  [15, 12, 12, 16, 12, 13, 15, 12, 12], [14, 13, 15, 11, 14, 12, 14, 14, 13, 14],
    #  [15, 15, 16, 14, 13, 13, 15, 17], [16, 12, 13, 11, 15, 13, 11, 15],
    #  [11, 15, 13, 14, 17, 12, 11, 16, 15], [18, 15, 15, 14, 14, 18, 15, 18],
    #  [16, 14, 16, 14, 16, 13, 14], [19, 16, 19, 15, 17, 17, 16, 16, 15],
    #  [20, 20, 18, 18, 14, 17, 15, 15, 13], [13, 21, 11, 14, 17, 18, 18, 18, 19],
    #  [15, 18, 14, 16, 15, 17, 15], [14, 18, 17, 20, 14, 20, 15, 18]]
    #
    # X_E = [[10, 13, 13, 12, 15, 14, 16, 14], [15, 15, 13, 12, 15],
    #  [ 15, 10, 16, 13, 15, 14, 12, 12], [16, 16, 14, 12, 15, 16, 11, 13, 16],
    #  [14, 15, 11, 15, 14, 12], [16, 11, 14, 14, 11, 15, 17, 15],
    #  [11, 11, 15, 13, 16, 17, 11, 16, 16, 14], [11, 11, 17, 17, 13, 17, 13, 15, 15],
    #  [13, 13, 16, 12, 12, 12, 12, 14, 13], [14, 13, 13, 11, 11, 15, 15, 11, 12],
    #  [12, 11, 15, 14, 15, 12, 12], [14, 16, 11, 13, 16, 16, 13, 16, 13],
    #  [13, 13, 12, 14, 12, 15, 15, 13, 14], [15, 15, 13, 14, 11, 14, 14]]

    # data without noise test2-2
    # X_I = [[13, 11, 15, 13, 14, 13, 12, 11], [14, 15, 16, 14, 14, 16],
    #  [15, 12, 12, 16, 12, 13, 15, 12, 12], [14, 13, 15, 11, 14, 12, 14, 14, 13, 14],
    #  [15, 15, 16, 14, 13, 13, 15, 17], [16, 12, 13, 11, 15, 13, 11, 15],
    #  [11, 15, 13, 14, 17, 12, 11, 16, 15], [18, 15, 15, 14, 14, 18, 15, 18, 10],
    #  [16, 11, 14, 16, 14, 16, 13, 14], [19, 16, 19, 15, 17, 17, 16, 16, 15],
    #  [18, 18, 14, 17, 15, 15], [13, 14, 17, 18, 18, 18, 19],
    #  [15, 18, 14, 16, 15, 17, 15], [14, 18, 17, 20, 14, 20, 15, 18]]
    #
    # X_E = [[10, 13, 13, 12, 15, 14, 16, 14], [17, 15, 15, 13, 12, 15],
    #  [15, 16, 13, 15, 14, 12, 12], [16, 16, 14, 12, 15, 16, 13, 16],
    #  [18, 18, 14, 15, 18, 11, 15, 14, 12], [16, 11, 14, 14, 9, 11, 15, 17, 10, 15],
    #  [11, 11, 15, 13, 16, 17, 11, 16, 16, 14], [11, 11, 17, 17, 13, 17, 13, 15, 15, 10],
    #  [13, 13, 16, 12, 12, 12, 12, 14, 13], [14, 13, 13, 11, 11, 15, 15, 11, 12],
    #  [10, 12, 11, 15, 14, 15, 17, 12, 12], [14, 16, 17, 11, 13, 16, 16, 13, 16, 13],
    #  [13, 13, 12, 18, 14, 12, 15, 15, 13, 14], [18, 15, 15, 13, 18, 14, 11, 14, 14]]

    # data without noise test2-3
    X_I = [[13, 11, 15, 13, 14, 13, 12, 11], [14, 15, 16, 14, 14, 10, 10, 16],
     [15, 12, 12, 16, 12, 13, 15, 12, 12], [14, 13, 15, 11, 14, 12, 14, 14, 13, 14],
     [15, 15, 16, 14, 13, 13, 15], [16, 12, 13, 15, 13, 15],
     [11, 15, 13, 14, 17, 12, 11, 16, 15], [18, 15, 15, 14, 14, 18, 15, 18],
     [16, 14, 16, 14, 16, 13, 14], [19, 16, 19, 15, 17, 17, 16, 16, 15],
     [20, 18, 18, 14, 17, 15, 15], [13, 14, 17, 18, 18, 18, 19],
     [15, 18, 16, 15, 17, 15], [18, 17, 20, 14, 20, 15, 18]]

    X_E = [[10, 13, 13, 12, 15, 14, 16, 14], [15, 15, 13, 12, 15],
     [ 15, 10, 16, 13, 15, 14, 12, 12], [16, 16, 14, 12, 15, 16, 11, 13, 16],
     [14, 15, 11, 15, 14, 12], [16, 11, 14, 14, 11, 15, 17, 15],
     [11, 11, 15, 13, 16, 17, 11, 16, 16, 14], [11, 11, 17, 17, 13, 17, 13, 15, 15],
     [13, 13, 16, 12, 12, 12, 12, 14, 13], [14, 13, 13, 15, 15, 12],
     [12, 11, 15, 14, 15, 17, 12, 12], [14, 16, 17, 11, 13, 16, 16, 13, 16, 13],
     [13, 13, 12, 18, 14, 12, 15, 15, 13, 14], [18, 15, 15, 13, 18, 14, 11, 14, 14]]

    # original data test3
    # X_E = [[10, 13, 11, 11, 13, 11, 12, 19, 15, 14], [13, 15, 17, 13, 12, 13, 15, 9, 9, 12],
    #  [18, 10, 13, 11, 15, 11, 15, 16, 16, 13], [17, 14, 18, 20, 12, 18, 23, 11, 16, 18],
    #  [14, 10, 11, 17, 19, 11, 12, 13, 13, 8], [14, 10, 14, 13, 14, 13, 12, 10, 19, 15],
    #  [7, 19, 18, 13, 15, 15, 11, 17, 17, 17], [13, 11, 8, 13, 15, 9, 11, 9, 15, 10],
    #  [12, 16, 14, 11, 7, 9, 18, 19, 19, 13], [14, 17, 11, 16, 18, 19, 13, 16, 14, 10],
    #  [13, 11, 12, 17, 14, 13, 12, 13, 19, 14], [12, 17, 12, 10, 13, 10, 10, 12, 9, 18],
    #  [20, 9, 13, 19, 14, 9, 14, 11, 13, 17], [10, 19, 14, 13, 15, 15, 14, 11, 8, 11]]
    #
    # X_I = [[11, 17, 15, 14, 13, 10, 12, 8, 17, 15], [13, 17, 18, 18, 15, 14, 14, 15, 19, 14],
    #  [16, 18, 12, 11, 17, 13, 14, 16, 11, 10], [12, 14, 10, 16, 13, 16, 11, 19, 13, 10],
    #  [13, 12, 19, 20, 11, 19, 14, 17, 11, 13], [12, 14, 12, 9, 15, 13, 11, 16, 14, 13],
    #  [14, 11, 13, 21, 14, 15, 10, 11, 15, 14], [16, 17, 12, 18, 15, 14, 15, 15, 15, 15],
    #  [15, 15, 16, 25, 17, 12, 13, 12, 13, 18], [13, 12, 15, 12, 11, 19, 11, 13, 12, 15],
    #  [16, 17, 23, 16, 11, 16, 29, 16, 26, 15], [11, 19, 11, 17, 14, 13, 16, 18, 17, 19],
    #  [18, 21, 19, 15, 13, 12, 24, 12, 17, 17], [16, 20, 21, 22, 19, 19, 20, 15, 19, 13]]

    # X_E = [[10, 13, 11, 11, 13, 11, 12, 15, 14], [13, 15, 13, 12, 13, 15, 9, 9, 12],
    #  [10, 13, 11, 15, 11, 15, 16, 16, 13], [17, 14, 18, 12, 18, 11, 16, 18],
    #  [14, 10, 11, 17, 19, 11, 12, 13, 13, 8], [14, 10, 14, 13, 14, 13, 12, 10, 15],
    #  [18, 13, 15, 15, 11, 17, 17, 17], [13, 11, 8, 13, 15, 9, 11, 9, 15, 10],
    #  [12, 16, 14, 11, 7, 9, 13], [14, 17, 11, 16, 13, 16, 14, 10],
    #  [13, 11, 12, 17, 14, 13, 12, 13, 14], [12, 17, 12, 10, 13, 10, 10, 12],
    #  [13, 14, 14, 11, 13, 17], [10, 14, 13, 15, 15, 14, 11, 11]]
    #
    # X_I = [[11, 15, 14, 13, 10, 12, 17, 15], [13, 17, 18, 18, 15, 14, 14, 15, 14],
    #  [16, 12, 11, 17, 13, 14, 16, 11, 10], [12, 14, 10, 16, 13, 16, 11, 13, 10],
    #  [13, 12, 19, 11, 19, 14, 17, 13], [12, 14, 12, 15, 13, 11, 16, 14, 13],
    #  [14, 11, 13, 21, 14, 15, 11, 15, 14], [16, 17, 12, 15, 14, 15, 15, 15, 15],
    #  [15, 15, 16, 17, 12, 13, 12, 13, 18], [13, 12, 15, 12, 11, 11, 13, 12, 15],
    #  [16, 17, 16, 11, 16, 16, 15], [11, 19, 11, 17, 14, 13, 16, 18, 17, 19],
    #  [18, 19, 15, 13, 12, 24, 12, 17, 17], [16, 20, 19, 19, 20, 15, 19, 13]]

    plt.figure(figsize=(9, 6))

    xAngle = np.int32(list(range(0, 14))) * 3

    X_I_avg, X_I_std = getAvgStd(X_I)
    X_E_avg, X_E_std = getAvgStd(X_E)

    line1 = plt.errorbar(xAngle, X_I_avg, X_I_std, fmt='b-', ecolor='b', capsize=3, label='X=I')
    line2 = plt.errorbar(xAngle, X_E_avg, X_E_std, fmt='r-', ecolor='r', capsize=3, label="X=X_GT")
    plt.legend()
    plt.ylabel('Iterations')
    plt.xlabel('Angle of X')


    plt.show()


if __name__ == "__main__":
    # drawSomething()

    baseDir = r"C:\Users\tianf\Phd\Research\our_papers\ACR2PAMI\Review_1\data\oneStepACR\test3ACR_with_scale2"

    # baseDir = os.path.join(baseDir, 'unknownx')
    # if not os.path.exists(baseDir):
    #     os.mkdir(baseDir)
    # GenerateDataOneStepACR(baseDir, isXKnown=False)


    file1 = os.path.join(baseDir, 'unknownx', 'poseError.json')
    file2 = os.path.join(baseDir, 'knownx', 'poseError.json')

    evaluate_pose_errors(baseDir, file1, file2)
