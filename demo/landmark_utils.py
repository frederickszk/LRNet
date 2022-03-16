from tqdm import tqdm
import numpy as np
import dlib
from collections import OrderedDict
import cv2
from calib_utils import track_bidirectional

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def shape_to_face(shape, width, height, scale=1.2):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    scale: the scale parameter of face, to enlarge the bounding box
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
    # face_center: the center coordinate of face (1*2 list [x_c, y_c])
    face_size: the face is rectangular( width = height = size)(int)
    """
    x_min, y_min = np.min(shape, axis=0)
    x_max, y_max = np.max(shape, axis=0)

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    face_size = int(max(x_max - x_min, y_max - y_min) * scale)
    # Enforce it to be even
    # Thus the real whole bounding box size will be an odd
    # But after cropping the face size will become even and
    # keep same to the face_size parameter.
    face_size = face_size // 2 * 2

    x1 = max(x_center - face_size // 2, 0)
    y1 = max(y_center - face_size // 2, 0)

    face_size = min(width - x1, face_size)
    face_size = min(height - y1, face_size)

    x2 = x1 + face_size
    y2 = y1 + face_size

    face_new = [int(x1), int(y1), int(x2), int(y2)]
    return face_new, face_size


def predict_single_frame(frame):
    """
    :param frame: A full frame of video
    :return:
    face_num: the number of face (just to verify if successfully detect a face)
    shape: landmark locations
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) < 1:
        return 0, None
    face = faces[0]

    landmarks = predictor(frame, face)
    face_landmark_list = [(p.x, p.y) for p in landmarks.parts()]
    shape = np.array(face_landmark_list)

    return 1, shape


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


def check_and_merge(location, forward, feedback, P_predict, status_fw=None, status_fb=None):
    num_pts = 68
    check = [True] * num_pts

    target = location[1]
    forward_predict = forward[1]

    # To ensure the robustness through feedback-check
    forward_base = forward[0]  # Also equal to location[0]
    feedback_predict = feedback[0]
    feedback_diff = feedback_predict - forward_base
    feedback_dist = np.linalg.norm(feedback_diff, axis=1, keepdims=True)

    # For Kalman Filtering
    detect_diff = location[1] - location[0]
    detect_dist = np.linalg.norm(detect_diff, axis=1, keepdims=True)
    predict_diff = forward[1] - forward[0]
    predict_dist = np.linalg.norm(predict_diff, axis=1, keepdims=True)
    predict_dist[np.where(predict_dist == 0)] = 1  # Avoid nan
    P_detect = (detect_dist / predict_dist).reshape(num_pts)

    for ipt in range(num_pts):
        if feedback_dist[ipt] > 2:  # When use float
            check[ipt] = False

    if status_fw is not None and np.sum(status_fw) != num_pts:
        for ipt in range(num_pts):
            if status_fw[ipt][0] == 0:
                check[ipt] = False
    if status_fw is not None and np.sum(status_fb) != num_pts:
        for ipt in range(num_pts):
            if status_fb[ipt][0] == 0:
                check[ipt] = False
    location_merge = target.copy()
    # Merge the results:
    """
    Use Kalman Filter to combine the calculate result and detect result.
    """

    Q = 0.3  # Process variance

    for ipt in range(num_pts):
        if check[ipt]:
            # Kalman parameter
            P_predict[ipt] += Q
            K = P_predict[ipt] / (P_predict[ipt] + P_detect[ipt])
            location_merge[ipt] = forward_predict[ipt] + K * (target[ipt] - forward_predict[ipt])
            # Update the P_predict by the current K
            P_predict[ipt] = (1 - K) * P_predict[ipt]
    return location_merge, check, P_predict


def detect_frames_track(frames, fps, use_visualization, visualize_path, video):
    frames_num = len(frames)
    assert frames_num != 0
    frame_height, frame_width = frames[0].shape[:2]
    """
    Pre-process:
    To detect the original results,
    and normalize each face to a certain width, 
    also its corresponding landmarks locations and 
    scale parameter.
    """
    face_size_normalized = 400
    faces = []
    locations = []
    shapes_origin = []
    shapes_para = []  # Use to recover the shape in whole frame. ([x1, y1, scale_shape])
    face_size = 0
    skipped = 0

    """
    Use single frame to detect face on Dlib (CPU)
    """
    # ----------------------------------------------------------------------------#

    print("Detecting:")
    for i in tqdm(range(frames_num)):
        frame = frames[i]
        face_num, shape = predict_single_frame(frame)

        if face_num == 0:
            if len(shapes_origin) == 0:
                skipped += 1
                # print("Skipped", skipped, "Frame_num", frames_num)
                continue
            shape = shapes_origin[i - 1 - skipped]

        face, face_size = shape_to_face(shape, frame_width, frame_height, 1.2)
        faceFrame = frame[face[1]: face[3],
                    face[0]:face[2]]
        if face_size < face_size_normalized:
            inter_para = cv2.INTER_CUBIC
        else:
            inter_para = cv2.INTER_AREA
        face_norm = cv2.resize(faceFrame, (face_size_normalized, face_size_normalized), interpolation=inter_para)
        scale_shape = face_size_normalized / face_size
        shape_norm = np.rint((shape - np.array([face[0], face[1]])) * scale_shape).astype(int)
        faces.append(face_norm)
        shapes_para.append([face[0], face[1], scale_shape])
        shapes_origin.append(shape)
        locations.append(shape_norm)

    """
    Calibration module.
    """
    segment_length = 2
    locations_sum = len(locations)
    if locations_sum == 0:
        return []
    locations_track = [locations[0]]
    num_pts = 68
    P_predict = np.array([0] * num_pts).reshape(num_pts).astype(float)
    print("Tracking")
    for i in tqdm(range(locations_sum - 1)):
        faces_seg = faces[i:i + segment_length]
        locations_seg = locations[i:i + segment_length]

        # ----------------------------------------------------------------------#
        """
        Numpy Version (DEPRECATED)
        """

        # locations_track_start = [locations_track[i]]
        # forward_pts, feedback_pts = track_bidirectional(faces_seg, locations_track_start)
        #
        # forward_pts = np.rint(forward_pts).astype(int)
        # feedback_pts = np.rint(feedback_pts).astype(int)
        # merge_pt, check, P_predict = check_and_merge(locations_seg, forward_pts, feedback_pts, P_predict)

        # ----------------------------------------------------------------------#
        """
        OpenCV Version
        """

        lk_params = dict(winSize=(15, 15),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Use the tracked current location as input. Also use the next frame's predicted location for
        # auxiliary initialization.

        start_pt = locations_track[i].astype(np.float32)
        target_pt = locations_seg[1].astype(np.float32)

        forward_pt, status_fw, err_fw = cv2.calcOpticalFlowPyrLK(faces_seg[0], faces_seg[1],
                                                                 start_pt, target_pt, **lk_params,
                                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        feedback_pt, status_fb, err_fb = cv2.calcOpticalFlowPyrLK(faces_seg[1], faces_seg[0],
                                                                  forward_pt, start_pt, **lk_params,
                                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        forward_pts = [locations_track[i].copy(), forward_pt]
        feedback_pts = [feedback_pt, forward_pt.copy()]

        forward_pts = np.rint(forward_pts).astype(int)
        feedback_pts = np.rint(feedback_pts).astype(int)

        merge_pt, check, P_predict = check_and_merge(locations_seg, forward_pts, feedback_pts, P_predict, status_fw,
                                                     status_fb)

        # ----------------------------------------------------------------------#

        locations_track.append(merge_pt)

    """
    If us visualization, write the results to the visualize output folder.
    """
    if locations_sum != frames_num:
        print("INFO: Landmarks detection failed in some frames. Therefore we disable the "
              "visualization for this video. It will be optimized in future version.")
    else:
        if use_visualization:
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            origin_video = cv2.VideoWriter(visualize_path + video + "_origin.avi",
                                           fourcc, fps, frame_size)
            track_video = cv2.VideoWriter(visualize_path + video + "_track.avi",
                                          fourcc, fps, frame_size)

            print("Visualizing")
            for i in tqdm(range(frames_num)):
                frame_origin = frames[i].copy()
                frame_track = frames[i].copy()
                shape_origin = shapes_origin[i]
                para_shift = shapes_para[i][0:2]
                para_scale = shapes_para[i][2]
                shape_track = np.rint(locations_track[i] / para_scale + para_shift).astype(int)
                for (x, y) in shape_origin:
                    cv2.circle(frame_origin, (x, y), 2, (0, 0, 255), -1)
                for (x, y) in shape_track:
                    cv2.circle(frame_track, (x, y), 2, (0, 255, 0), -1)
                origin_video.write(frame_origin)
                track_video.write(frame_track)
            origin_video.release()
            track_video.release()

    aligned_landmarks = []
    for i in locations_track:
        shape = landmark_align(i)
        shape = shape.ravel()
        shape = shape.tolist()
        aligned_landmarks.append(shape)

    return aligned_landmarks
