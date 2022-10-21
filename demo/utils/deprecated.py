# The DEPRECATED codes. Just for reference
# Includes the LK-tracking algorithm and landmark alignment.

import cv2 as cv
import numpy as np
from collections import OrderedDict

"""
LK-Tracking
"""
pyramid_source = []
pyramid_target = []
patch_source_pyramid = []
JT_source_pyramid = []
Hinv_source_pyramid = []
windows = []


class Window:
    def __init__(self, center_x, center_y, window_size):
        self.center_x = center_x
        self.center_y = center_y
        self.window_size = window_size
        # The displacement vectors.
        # Important for simulating the internal calculation vector: g and d. Also they record the final results.
        self.Dx = 0
        self.Dy = 0

        self.map_x = None
        self.map_y = None
        self.generate_map()

    def generate_map(self):
        epsilon = 0.001
        start_x = self.center_x - self.window_size//2
        start_y = self.center_y - self.window_size//2
        # print(start_x, start_y)
        # When window_size is odd, we must use this form to enforce the size of map to be window_size.
        crop_x = np.arange(start_x, start_x+self.window_size - epsilon, 1.0).astype(np.float32).reshape(1, self.window_size)
        self.map_x = np.repeat(crop_x, self.window_size, axis=0)
        crop_y = np.arange(start_y, start_y + self.window_size - epsilon, 1.0).astype(np.float32).reshape(self.window_size, 1)
        self.map_y = np.repeat(crop_y, self.window_size, axis=1)

    def pyrDown(self):
        """
        When down-sample the original patch, the corresponding point position should be /2.
        However the maps' coordinate should not be dimply /2, therefore the maps need regenerate.
        """
        self.center_x = self.center_x / 2
        self.center_y = self.center_y / 2
        self.Dx = self.Dx / 2
        self.Dy = self.Dy / 2
        self.generate_map()

    def pyrUp(self):
        """
        When calculating the pyramidal LK and moving to the next (bigger) pyramid, the patch size will be doubled.
        Thus the corresponding point position should be *2.

        Here we should consider the displacement vector (Dx, Dy), to simulate the equation: g_(L-1) = 2*(g_L + d_L)
        (d_L calculated in this level iteration and g_L is inherited from the former level iteration, both stored in
        displacement vector)
        """
        self.center_x = self.center_x * 2
        self.center_y = self.center_y * 2
        self.Dx = self.Dx * 2
        self.Dy = self.Dy * 2
        self.generate_map()

    def move(self, delta_x, delta_y):
        self.Dx += delta_x
        self.Dy += delta_y

    def crop(self, img):
        # Notice!!: map_column calculated from x, while map_row calculated from y.
        #           Which contradict to the matrix index.
        patch = cv.remap(img, self.map_x + self.Dx,
                         self.map_y + self.Dy, cv.INTER_LINEAR)
        return patch


def generate_weight(patch_size):
    """
    Generate the weight matrix
    :param patch_size: (Int) The patch_size
    :return: The weight map (patch_size * patch_size * 1).
    """
    center = [patch_size // 2, patch_size // 2]
    sigma_x = sigma_y = patch_size // 2
    maps = np.fromfunction(lambda x, y: ((x - center[0])/sigma_x) ** 2 +
                                        ((y - center[1])/sigma_y) ** 2,
                           (patch_size, patch_size),
                           dtype=int)
    return np.expand_dims(np.exp(maps/-2.0), -1)


def craft_pyramid(image, level, pyramid_container):
    pyramid_container.clear()
    pyramid_container.append(image)
    for i in range(level - 1):
        image = cv.pyrDown(image)
        pyramid_container.append(image)


def lk_track(face_source, face_target, landmarks_source, window_size, pyramid_level):
    # Create the image pyramid for both source and target.
    craft_pyramid(face_source, pyramid_level, pyramid_source)
    craft_pyramid(face_target, pyramid_level, pyramid_target)

    # Generate the weight map
    weight_map = generate_weight(window_size)

    # Create windows for cropping patches.
    windows.clear()
    for landmark in landmarks_source:
        x, y = landmark
        # windows.append(Window(x, y, patch_size, face_source.shape[0], face_source.shape[0]))
        windows.append(Window(x, y, window_size))

    # Initialize the patches of both the source.
    # Notice that here both using the same window, i.e., d = 0.
    # Afterwards, patch_target will be changed while patch_source will fixed.
    patch_source_pyramid.clear()
    JT_source_pyramid.clear()
    Hinv_source_pyramid.clear()

    for level in range(pyramid_level):
        patch_source = []
        for window in windows:
            patch_source.append(window.crop(pyramid_source[level]))
            if level < pyramid_level - 1:
                window.pyrDown()

        # Calculate the Jacobian and Hessen matrix of patch_source
        JT_source = []
        Hinv_source = []
        for patch in patch_source:
            """
            # cv.Sobel(_, _, x, y, ...), x indicating the horizontal, 
            # while it's in fact the y axis, for the y is the column.
            # horizontal means increase at column.
            """
            gradient_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
            gradient_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
            gradient_x_w = gradient_x * weight_map
            gradient_y_w = gradient_y * weight_map

            J_x = np.reshape(gradient_x, (-1, 1))
            J_y = np.reshape(gradient_y, (-1, 1))
            J_x_w = np.reshape(gradient_x_w, (-1, 1))
            J_y_w = np.reshape(gradient_y_w, (-1, 1))

            J = np.concatenate((J_x, J_y), axis=1)
            J_w = np.concatenate((J_x_w, J_y_w), axis=1)
            JT_w = np.transpose(J_w)
            H = np.matmul(JT_w, J)
            Hinv = np.linalg.inv(H)
            # Noticed that we only collect the weighted JT here.
            JT_source.append(JT_w)
            Hinv_source.append(Hinv)

        # Collect all the pre-processed data in each level.
        patch_source_pyramid.append(patch_source)
        JT_source_pyramid.append(JT_source)
        Hinv_source_pyramid.append(Hinv_source)
    #
    # """
    # Sequential Execution
    # """
    max_iter_step = 15
    for level in range(pyramid_level-1, -1, -1):
        epsilon_der1 = 1.0 + level
        for patch_s, window, JT, Hinv in zip(patch_source_pyramid[level], windows, JT_source_pyramid[level], Hinv_source_pyramid[level]):
            count = 1
            while True:
                # Patch of target. which will move in each iteration.
                patch_t = window.crop(pyramid_target[level])
                # Calculate the residual
                r = patch_t - patch_s
                r = np.reshape(r, (-1, 1))
                der1 = np.matmul(JT, r)
                der1_norm = np.linalg.norm(der1)
                delta = - np.matmul(Hinv, der1)
                if der1_norm < epsilon_der1 or count > max_iter_step:
                    if level != 0:
                        # When reach the final level, stop the up-sample.
                        window.pyrUp()
                    break
                else:
                    window.move(delta[0][0], delta[1][0])
                    count += 1
    predictions = []
    for window in windows:  # type: Window
        predictions.append([window.center_x + window.Dx, window.center_y + window.Dy])
    return np.array(predictions)


def track_bidirectional(faces, locations):
    patch_size = 15
    frames_num = len(faces)
    pyramid_level = 4

    forward_pts = [locations[0].copy()]
    for i in range(1, frames_num):
        feature_old = faces[i-1] / 255.0
        feature_new = faces[i] / 255.0
        location_old = forward_pts[i - 1]
        forward_pt = lk_track(feature_old, feature_new, location_old, patch_size, pyramid_level)
        forward_pts.append(forward_pt)

    feedback_pts = [None] * (frames_num - 1) + [forward_pts[-1].copy()]
    for i in range(frames_num - 2, -1, -1):
        feature_old = faces[i+1] / 255.0
        feature_new = faces[i] / 255.0
        location_old = feedback_pts[i - 1]
        feedback_pt = lk_track(feature_old, feature_new, location_old, patch_size, pyramid_level)
        feedback_pts[i] = feedback_pt

    return forward_pts, feedback_pts


"""
Landmark alignment
"""
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])  # Used for landmark alignment, DEPRECATED


def landmark_align(shape):
    desiredLeftEye = (0.35, 0.25)
    desiredFaceWidth = 2
    desiredFaceHeight = 2
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0)  # .astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0)  # .astype("int")
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))  # - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = 0  # desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    n, d = shape.shape
    temp = np.zeros((n, d + 1), dtype="int")
    temp[:, 0:2] = shape
    temp[:, 2] = 1
    aligned_landmarks = np.matmul(M, temp.T)
    return aligned_landmarks.T  # .astype("int"))
