"""
The [pose3 class](https://github.com/openMVG/openMVG/blob/master/src/openMVG/geometry/pose3.hpp)

[Rotation registration with android phone euler angles](https://github.com/openMVG/openMVG/issues/551)
"""

import numpy as np
import math
import transforms3d

class Pose3(object):

    def __init__(self, R=np.eye(3), C=np.zeros(3)):
        assert np.allclose(np.linalg.inv(R), R.T)
        C = C.reshape(3,1)

        self.rotation_ = R
        self.center_ = C

    def rotation(self):
        return self.rotation_

    def center(self):
        return self.center_

    def translation(self):
        return -np.dot(self.rotation_, self.center_)

    def apply_point(self, p):
        p = np.array(p).reshape(3, 1)
        return np.dot(self.rotation(), p - self.center())

    def compose(self, P):
        """
        compose current pose to P, that is, Return = self * P
        :param P:
        :return:
        """
        assert type(P) == Pose3
        return Pose3( np.dot(self.rotation(), P.rotation()),  P.center() +  np.dot(P.rotation().T, self.center()))

    def inverse(self):
        return Pose3(self.rotation().T, -( np.dot( self.rotation(), self.center())))

    def __copy__(self):
        return Pose3(self.rotation(), self.center())

    def copy(self):
        return Pose3(self.rotation(), self.center())

    def toCenter6D(self):
        """
        translate pose to 6D representation: cx, cy, cz, euler_z, euler_y, euler_x
        :return: pose using euler angle and center
        """
        return np.append(self.center().T, rotm2zyxEulDegree (self.rotation()))

    def to6D(self):
        """
        translate pose to 6D representation: tx, ty, tz, euler_z, euler_y, euler_x
        :return: pose using euler angle and translation
        """
        t = self.translation().T.ravel()
        euler = rotm2zyxEulDegree(self.rotation())
        # return np.append(t, euler)
        return np.append(np.array(t), np.array(euler))

    def toRt(self):
        t = self.translation().ravel()
        R = self.rotation()
        return R, t

    def toSE3(self):
        R, t = self.toRt()
        Rt34 = np.hstack((R.reshape(3,3), t.reshape(3,1)))
        T_se3 = np.vstack((Rt34, np.array([0, 0, 0, 1])))
        return T_se3

    def to_t_aixsAngle(self):
        """
        transform Pose to [t, axis, angle], the angle is in degree
        :return:
        """
        t = self.translation().T.ravel()
        axis, angle = transforms3d.axangles.mat2axangle(self.rotation())
        if angle < 0:
            angle = -angle
            axis = -1 * axis
        angle = math.degrees(angle)
        return t, axis, angle

    def frobeniusNorm(self):
        t_se3 = self.toSE3()
        return np.linalg.norm(t_se3, 'fro')

    def sameAs(self, P):
        assert np.allclose(self.rotation(), P.rotation()) and np.allclose(self.center(), P.center())

    def debug(self):
        info = "R:{}\neuler_zyx:{}\nC:{}\nt:{}\n".format(self.rotation(),
                                                         rotm2zyxEulDegree(self.rotation()).T, self.center().T, self.translation().T)
        print (info)
        return info

    def display(self):
        pose6D = self.to6D()
        t, axis, angle = self.to_t_aixsAngle()
        print("R:{}\n Euler:{}\nAxis:{}\nAngle:{}\nt:{}\n"
              .format(self.rotation(), pose6D[:3], axis, angle, t))


    @classmethod
    def fromRt(cls, R, t):
        assert np.allclose(R.T, np.linalg.inv(R))
        R = np.asarray(R)
        C = -R.T.dot(t)

        return Pose3(R, C)

    @classmethod
    def from6D(cls, relative6D):
        """
        Pose from 6D (tx, ty, tz, roll, yaw, pitch)
        :param relative6D:
        :return:
        """
        relative6D = np.array(relative6D)
        R = zyxEulDegree2Rotm(*relative6D[3:])
        t = relative6D[:3]

        return cls.fromRt(R, t)

    @classmethod
    def fromCenter6D(cls, center6D):
        """
        Pose from 6D (cx, cy, cz, roll, yaw, pitch)
        :param pose:
        :return:
        """
        center6D = np.array(center6D)
        assert len(center6D) == 6
        R = zyxEulDegree2Rotm (*center6D[3:])
        return Pose3(R, center6D[:3])

    @classmethod
    def from_t_axisAngle(cls, t, axis, angle):
        """
        transform [t, axis, angle] to Pose3, the angle is in degree
        :param t:  translation
        :param axis: axis
        :param angle: angle, in degree
        :return: Pose3
        """
        t = np.float32(t)
        angle = math.radians(angle)
        R = transforms3d.axangles.axangle2mat(axis, angle)
        R = np.asarray(R)
        return cls.fromRt(R, t.T)

    @classmethod
    def fromSE3(cls, se3):
        se3mat = np.matrix(se3) # type: np.matrix
        R = se3mat[0:3, 0:3]
        t = np.array(se3mat[0:3, 3]).ravel()
        return cls.fromRt(R,t)


"""
Rotation transform
"""

def zyxEulDegree2Rotm(z_degree, y_degree, x_degree):
    z_rad = math.radians(z_degree)
    y_rad = math.radians(y_degree)
    x_rad = math.radians(x_degree)

    return transforms3d.euler.euler2mat(z_rad, y_rad, x_rad, 'rzyx')

def AxisAngle2RMat(axis, angle):
    return transforms3d.axangles.axangle2mat(axis, angle)

def RMat2AxisAngle(RMat):
    axis, angle = transforms3d.axangles.mat2axangle(RMat)
    return axis, angle

def rotm2zyxEulDegree(R):
    zyx_rad = transforms3d.euler.mat2euler(R, 'rzyx')
    zyx_degree = list(map(math.degrees, zyx_rad))
    return np.array(zyx_degree)