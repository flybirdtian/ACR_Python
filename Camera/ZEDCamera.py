"""
A minimal restful interface for ZED
Grab RGB*2+Depth*2 images
"""


import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import cv2
import os

import numpy as np
import time

from Camera.CameraBase import CameraCalibration, CameraBase
from pytypes import override
import typing

class ZEDCamera(CameraBase):
    def __init__(self):
        super(ZEDCamera, self).__init__()
        # Create a PyZEDCamera object
        self.zed = zcam.PyZEDCamera()
        self.camera_settings_table = self.get_camera_settings_table()
        # runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_FILL
        self.runtime_parameters = zcam.PyRuntimeParameters()
        self.camera_settings_value = None
        self.image_mat = core.PyMat()  # the image mat, useful for all capturing

    def _start(self):
        # Open the camera
        if self.available():
            return True

        # Create a PyInitParameters object and set configuration parameters
        init_params = zcam.PyInitParameters()
        init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720  # Use 2K video mode
        # init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD1080  # Use HD1080 video mode
        # init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720
        # init_params.camera_fps = 10  # 30 is default
        # init_params.enable_right_side_measure = True
        init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_QUALITY
        init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER
        init_params.depth_minimum_distance = 300 # 300mm, 30cm


        err = self.zed.open(init_params)
        if err != tp.PyERROR_CODE.PySUCCESS:
            print("We failed to open the ZED camera, exit!")
            # exit(1)
            return False

        time.sleep(4)

        self._set_default_camera_settings()
        self.camera_settings_value = self.getParameters()

        K1, K2 = self.get_camera_parameters()
        K1 = np.array(K1).astype(np.float32).reshape(3, 3)
        K2 = np.array(K2).astype(np.float32).reshape(3, 3)
        self.cameraCalibration = CameraCalibration(leftK=K1, rightK=K2)

        return True

    def available(self):
        return self.zed.is_zed_connected() and self.zed.is_opened()

    def stop(self):
        self._set_default_camera_settings()
        self.zed.close()
        return True


    def get_camera_parameters(self):
        info = core.PyCameraInformation(self.zed, self.zed.get_resolution())
        print("serial num:{}".format(info.serial_number))
        print("firmware_version:{}".format(info.firmware_version))
        py_calib = info.calibration_parameters
        # print("All calibration_parameters:{}".format(py_calib))
        print("R\n:{}".format(py_calib.R))
        print("t\n:{}".format(py_calib.T))

        cam_1 = py_calib.left_cam
        K1 = [cam_1.fx, 0, cam_1.cx,  0, cam_1.fy, cam_1.cy, 0, 0, 1]
        print("K left:{}".format(K1))

        cam_2 = py_calib.right_cam
        K2 = [cam_2.fx, 0, cam_2.cx, 0, cam_2.fy, cam_2.cy, 0, 0, 1]
        print("K right:{}".format(K2))

        return K1, K2

    def grab_rgb_and_depth(self):
        # to get right image, we need to grab twice, do not know why
        # first grab
        self.zed.grab(self.runtime_parameters)
        # second grab
        if not self.zed.grab(self.runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            return None, None

        self.zed.retrieve_image(self.image_mat, sl.PyVIEW.PyVIEW_LEFT)
        im_left = self.image_mat.get_data()

        # self.zed.retrieve_image(self.image_mat, sl.PyVIEW.PyVIEW_RIGHT)
        # im_right = self.image_mat.get_data()
        #
        # self.zed.retrieve_image(self.image_mat, sl.PyVIEW.PyVIEW_DEPTH)
        # im_depth_view = self.image_mat.get_data()

        self.zed.retrieve_measure(self.image_mat, sl.PyMEASURE.PyMEASURE_DEPTH)
        im_measure = np.uint16(self.image_mat.get_data())

        print("Get images done")
        return im_left, im_measure

    def run(self):
        pass

    @override
    def open(self) -> bool:
        return self._start()

    @override
    def close(self) -> bool:
        return self.stop()

    @override
    def getImage(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        # self.grab_rgb_and_depth(pre_grab=True) # grab first
        rgb_image, depth_image = self.grab_rgb_and_depth()
        if (rgb_image.shape[2] > 3):
            rgb_image = rgb_image[:,:, :3]
        return rgb_image, depth_image

    def _set_default_camera_settings(self):
        # self.setParameters({'EXPOSURE':-1, 'WHITEBALANCE':-1})
        # print(self.camera_settings_value)

        d = self.get_camera_settings_table() # will assign value below

        for (k, v) in d.items():
            self.zed.set_camera_settings(v, -1, use_default=True)

    def get_camera_settings_table(self):
        """
        https://www.stereolabs.com/developers/documentation/API/v2.0.1/group__Enumerations.html
        CAMERA_SETTINGS_BRIGHTNESS
        Defines the brightness control. Affected value should be between 0 and 8.
        CAMERA_SETTINGS_CONTRAST
        Defines the contrast control. Affected value should be between 0 and 8.
        CAMERA_SETTINGS_HUE
        Defines the hue control. Affected value should be between 0 and 11.
        CAMERA_SETTINGS_SATURATION
        Defines the saturation control. Affected value should be between 0 and 8.
        CAMERA_SETTINGS_GAIN
        Defines the gain control. Affected value should be between 0 and 100 for manual control. If ZED_EXPOSURE is set to -1, the gain is in auto mode too.
        CAMERA_SETTINGS_EXPOSURE
        Defines the exposure control. A -1 value enable the AutoExposure/AutoGain control,as the boolean parameter (default) does. Affected value should be between 0 and 100 for manual control.
        CAMERA_SETTINGS_WHITEBALANCE
        Defines the color temperature control. Affected value should be between 2800 and 6500 with a step of 100. A value of -1 set the AWB ( auto white balance), as the boolean parameter (default) does.
        CAMERA_SETTINGS_AUTO_WHITEBALANCE
        Defines the status of white balance (automatic or manual). A value of 0 disable the AWB, while 1 activate it.

        And pyzed/defines.py
        PyCAMERA_SETTINGS_AUTO_WHITEBALANCE = None # (!) real value is ''
        PyCAMERA_SETTINGS_BRIGHTNESS = None # (!) real value is ''
        PyCAMERA_SETTINGS_CONTRAST = None # (!) real value is ''
        PyCAMERA_SETTINGS_EXPOSURE = None # (!) real value is ''
        PyCAMERA_SETTINGS_GAIN = None # (!) real value is ''
        PyCAMERA_SETTINGS_HUE = None # (!) real value is ''
        PyCAMERA_SETTINGS_LAST = None # (!) real value is ''
        PyCAMERA_SETTINGS_SATURATION = None # (!) real value is ''
        PyCAMERA_SETTINGS_WHITEBALANCE = None # (!) real value is ''
        """
        d = dict()

        d['AUTO_WHITEBALANCE'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_AUTO_WHITEBALANCE
        d['BRIGHTNESS'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_BRIGHTNESS
        d['CONTRAST'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_CONTRAST
        d['EXPOSURE'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_EXPOSURE
        d['GAIN'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_GAIN
        d['HUE'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_HUE
        d['SATURATION'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_SATURATION
        d['WHITEBALANCE'] = sl.PyCAMERA_SETTINGS.PyCAMERA_SETTINGS_WHITEBALANCE

        return d

    @override
    def getParameters(self) -> dict:

        d = self.get_camera_settings_table() # will assign value below

        for (k, v) in d.items():
            d[k] = self.zed.get_camera_settings(v)

        return d

    @override
    def setParameters(self, params : dict) -> bool:
        for (k, v) in params.items():
            k = str(k).upper()
            v = int(v)

            if k in self.camera_settings_table:
                setting_name = self.camera_settings_table[k]
                print("PARMS {},{}:{}".format(k, setting_name,v))
                self.zed.set_camera_settings(setting_name, v)
            else:
                print("It seems that PARMS {}:{} is not valid, continue".format(k, v))
                # return False

        self.camera_settings_value = self.getParameters() # update the settings
        return True



def test_grab(write2disk=False):
    R_ZED = ZEDCamera()

    R_ZED.open()

    for i in range(1):
        rgb_image, depth_image = R_ZED.getImage()
        cv2.imshow('1', rgb_image)
        cv2.imshow('2', depth_image)
        cv2.waitKey(0)

    R_ZED.close()

# {'WHITEBALANCE': 4600, 'BRIGHTNESS': 4, 'CONTRAST': 4, 'EXPOSURE': 54, 'HUE': 0, 'GAIN': 98, 'AUTO_WHITEBALANCE': 1, 'SATURATION': 4}
def test_info():
    R_ZED = ZEDCamera() # write to disk for the comparision in images

    R_ZED.open()

    R_ZED.get_camera_parameters()

    print(R_ZED.camera_settings_table)
    print(R_ZED.camera_settings_value)
    R_ZED.getImage()

    R_ZED.setParameters({'WHITEBALANCE':4200})
    R_ZED.getImage()
    print(R_ZED.camera_settings_value)

    R_ZED.setParameters({'AUTO_WHITEBALANCE': 1})
    R_ZED.getImage()
    print(R_ZED.camera_settings_value)


    v = R_ZED.camera_settings_value
    v['BRIGHTNESS'] = 0
    R_ZED.setParameters(v)
    print(R_ZED.camera_settings_value)

    dir = "C:/Code/bird/AFGCD-master/data/data2"
    ref_path = os.path.join(dir, 'ref.png')
    ref_depth_path = os.path.join(dir, 'ref_depth.png')

    image, image_depth = R_ZED.getImage()
    cv2.imwrite(ref_path, image)
    cv2.imwrite(ref_depth_path, image_depth)

    R_ZED.close()



if __name__ == "__main__":

    # test_grab(write2disk=False)
    # test_grab(write2disk=True)
    test_info()
