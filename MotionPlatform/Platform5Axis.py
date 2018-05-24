"""
Our six-dof hardware
"""

import numpy as np
from MotionPlatform import PlatformBase as platform
import requests
from Utils.POSE3 import Pose3

# def generate_cmd_and_parse_response_json(url = 'http://127.0.0.1:5000/', cmd='/'):
#     if cmd[0] == '/':
#         cmd = cmd[1:]
#     r = requests.get(url + cmd)
#     assert r.status_code == 200
#     assert r.headers['content-type'] == "application/json"
#     print(r.json())
#     return r.json()


def generate_cmd_and_parse_response_text(url = 'http://127.0.0.1:5000/', cmd='/'):
    if cmd[0] == '/':
        cmd = cmd[1:]

    try:
        r = requests.get(url + cmd)
        print (r.status_code)
        print(r.text)
        assert r.status_code == 200
        return r.text
    except:
        c = input("Seems failed to perform the command, "
                  "press to continue or q to quit. PLEASE MAKE SURE WHAT YOU ARE DOING")
        if c == 'q':
            exit(-1)
        else:
            return "True" # this is ugly


class Platform5Axis(platform.PlatformBase):

    def __init__(self, url='http://localhost:1234/'):
        super(Platform5Axis, self).__init__()
        self.url = url

    def open(self):
        return generate_cmd_and_parse_response_text(self.url) == 'Hello,world.'

    def close(self):
        return True

    def rotate(self, pitch, yaw, roll):
        if np.allclose([pitch, yaw, roll], np.zeros(3)): # won't more too small
            return True

        rtn = generate_cmd_and_parse_response_text(self.url, '/rotate?pitch={}&roll={}&yaw={}'.format(pitch, roll, yaw))
        print ("Finished rotation:{}".format(rtn))
        return rtn == "True" # rtn is string

    def translate(self, x, y, z):
        if np.allclose([x, y, z], np.zeros(3)):  # won't more too small
            return True

        rtn = generate_cmd_and_parse_response_text(self.url, '/translate?x={}&y={}&z={}'.format(x, y, z))
        print ("Finished translation:{}".format(rtn))
        return rtn == "True"

    def movePose(self, movingPose: Pose3):
        motion = movingPose.toCenter6D()
        rots = np.array(motion[3:])  # sequence: rz, ry, rx
        trans = np.array(motion[:3]) * -1  # sequence: tx, ty, tz
        rs = self.rotate(rots[2], rots[1], rots[0])
        ts = self.translate(trans[0], trans[1], trans[2])
        return rs and ts


    def autoHorizon(self):
        rtn = generate_cmd_and_parse_response_text(self.url, '/autohorizon')
        print ("Finished autohorizon:{}".format(rtn))
        return rtn == "True"

    def stopAutoHorizon(self):
        rtn = generate_cmd_and_parse_response_text(self.url, '/freeze')
        print ("Finished freezing:{}".format(rtn))
        return rtn == "True"

    def goHome(self):
        rtn = generate_cmd_and_parse_response_text(self.url, '/gohome')
        print ("Finished homing:{}".format(rtn))
        return rtn == "True"


if __name__ == "__main__":
    platform5Axis = Platform5Axis(url='http://127.0.0.1:10241/')

    rotation = np.array([2, 10, -5.2]) * -1

    platform5Axis.rotate(*rotation)

    translation = np.array([-10, 20.3, -0.02])

    platform5Axis.translate(*translation)