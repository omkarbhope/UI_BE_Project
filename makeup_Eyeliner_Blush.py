from __future__ import division
import cv2
import numpy as np
from numpy.linalg import eig, inv
from scipy.interpolate import interp1d, splprep, splev,interp2d
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.pylab import *
from skimage import color
from scipy import misc
import time


class makeup(object):
    """
    Class that handles application of color, and performs blending on image.
    """

    def __init__(self, img):
        """ Initiator method for class """
        self.red_l = 0
        self.green_l = 0
        self.blue_l = 0
        self.red_b = 0
        self.green_b = 0
        self.blue_b = 0
        self.image = img
        self.height, self.width = self.image.shape[:2]
        self.im_copy = self.image.copy()

        self.intensity = 0.8

        self.x = []
        self.y = []
        self.xleft=[]
        self.yleft=[]
        self.xright=[]
        self.yright=[]

    def __draw_liner(self, eye, kind):
        """
        Draws eyeliner.
        """
        eye_x = []
        eye_y = []
        x_points = []
        y_points = []
        for point in eye:
            x_points.append(int(point.split()[0]))
            y_points.append(int(point.split()[1]))
        curve = interp1d(x_points, y_points, 'quadratic')
        for point in np.arange(x_points[0], x_points[len(x_points) - 1] + 1, 1):
            eye_x.append(point)
            eye_y.append(int(curve(point)))
        if kind == 'left':
            y_points[0] -= 1
            y_points[1] -= 1
            y_points[2] -= 1
            x_points[0] -= 1
            x_points[1] -= 1
            x_points[2] -= 1
            curve = interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                else:
                    eye_y.append(int(curve(point)) - 3)
        elif kind == 'right':
            x_points[3] += 1
            x_points[2] += 1
            x_points[1] += 1
            y_points[3] -= 1
            y_points[2] -= 1
            y_points[1] -= 1
            curve = interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                elif count:
                    eye_y.append(int(curve(point)) - 3)
        curve = zip(eye_x, eye_y)
        points = []
        for point in curve:
            points.append(np.array(point, dtype=np.int32))
        #points = np.asarray(curve)
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(self.im_copy, [points],0)
        return

    def __create_eye_liner(self, eyes_points):
        """
        Apply eyeliner.
        """
        left_eye = eyes_points[0].split('\n')
        right_eye = eyes_points[1].split('\n')
        right_eye = right_eye[0:4]
        self.__draw_liner(left_eye, 'left')
        self.__draw_liner(right_eye, 'right')
        return

    def __inter_plot(self, lx=[], ly=[], k1='quadratic'):
        """
        Interpolate with interp1d.
        """
        unew = np.arange(lx[0], lx[-1]+1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return unew, f2(unew)


    def __blush(self, x_right, y_right, x_left, y_left):

        intensity = 0.3
        # Create blush shape
        mask = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(mask, np.array(c_[x_right, y_right], dtype='int32'), 1)
        cv2.fillConvexPoly(mask, np.array(c_[x_left, y_left], dtype='int32'), 1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0) * intensity

        # Add blush color to image
        # val = color.rgb2lab((self.im_copy / 255.))
        val = cv2.cvtColor(self.im_copy, cv2.COLOR_RGB2LAB).astype(float)
        val[:, :, 0] = val[:, :, 0] / 255. * 100.
        val[:, :, 1] = val[:, :, 1] - 128.
        val[:, :, 2] = val[:, :, 2] - 128.
        LAB = color.rgb2lab(np.array((self.red_b / 255., self.green_b / 255., self.blue_b / 255.)).reshape(1, 1, 3)).reshape(3,)

        mean_val = np.mean(np.mean(val, axis=0), axis = 0)
        mask = np.array([mask,mask,mask])
        mask = np.transpose(mask, (1,2,0))
        lab = np.multiply((LAB - mean_val), mask)

        val[:, :, 0] = np.clip(val[:, :, 0] + lab[:,:,0], 0, 100)
        val[:, :, 1] = np.clip(val[:, :, 1] + lab[:,:,1], -127, 128)
        val[:, :, 2] = np.clip(val[:, :, 2] + lab[:,:,2], -127, 128)

        self.im_copy = (color.lab2rgb(val) * 255).astype(np.uint8)
        # val[:, :, 0] = (np.clip(val[:, :, 0] + lab[:,:,0], 0, 100) / 100 * 255).astype(np.uint8)
        # val[:, :, 1] = (np.clip(val[:, :, 1] + lab[:,:,1], -127, 128) + 127).astype(np.uint8)
        # val[:, :, 2] = (np.clip(val[:, :, 2] + lab[:,:,2], -127, 128) + 127).astype(np.uint8)

        # self.im_copy = cv2.cvtColor(val, cv2.COLOR_LAB2RGB)


    def apply_liner(self, landmarks):
        """
        Applies black liner on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
        Output:
            1. The face applied with eyeliner.
        """
        liner = self.__get_upper_eyelids(landmarks)
        eyes_points = liner.split('\n\n')
        self.__create_eye_liner(eyes_points)
        return self.im_copy


    def __get_boundary_points(self, landmarks, flag):
        """
        Find out the boundary of blush.
        """
        if flag == 0:
            # Right Cheek
            r = (landmarks[15, 0] - landmarks[35, 0]) / 3.5
            center = (landmarks[15] + landmarks[35]) / 2.0
        elif flag == 1:
            # Left Cheek
            r = (landmarks[1, 0] - landmarks[31, 0]) / 3.5
            center = (landmarks[1] + landmarks[31]) / 2.0

        points_1 = [center[0] - r, center[1]]
        points_2 = [center[0], center[1] - r]
        points_3 = [center[0] + r, center[1]]
        points_4 = [center[0], center[1] + r]
        points_5 = points_1

        points = np.array([points_1, points_2, points_3, points_4, points_5])

        x, y = points[0:5, 0], points[0:5, 1]

        tck, u = splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)


    def apply_blush(self, landmarks, R, G, B):
        """
        Applies blush on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
            2. Color of blush in the order of r, g, b.
        Output:
            1. The face applied with blush.
        """

        # Find Blush Loacations
        x_right, y_right = self.__get_boundary_points(landmarks, 0)
        x_left, y_left = self.__get_boundary_points(landmarks, 1)

        # Apply Blush
        self.red_b = R
        self.green_b = G
        self.blue_b = B
        self.__blush(x_right, y_right, x_left, y_left)

        return self.im_copy

    def __get_upper_eyelids(self, landmarks, flag=None):
        """
        Find out landmarks corresponding to upper eyes.
        """
        if landmarks is None:
            return None
        liner = ""
        for point in landmarks[36:40]:
            liner += str(point).replace('[', '').replace(']', '') + '\n'
        liner += '\n'
        for point in landmarks[42:46]:
            liner += str(point).replace('[', '').replace(']', '') + '\n'
        return liner

    def apply_makeup(self, landmarks):
        self.im_copy = self.apply_blush(landmarks, 223., 91., 111.)
        self.im_copy = self.apply_liner(landmarks)
        return self.im_copy




