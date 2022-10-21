from tqdm import tqdm
import numpy as np
import cv2
import face_alignment
from os.path import join
import demo.utils.shared as shared
from datetime import datetime
# from .deprecated import track_bidirectional, landmark_align  # DEPRECATED codes, just for reference.


def shape_to_face(shape, width, height, scale=1.2):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    scale: the scale parameter of face, to enlarge the bounding box
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
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


def predict_single_frame(frame, fa, face_box=None):
    """
    Extract landmarks from a single frame
    :param
    frame: An image frame of the video;
    fa: FaceAlignment module.
    face_box: Pre-detected face box [Type: tuple][Shape: (4,)]
    :return:
    face_num: The number of face (just to verify if successfully detect a face)
    shape: Landmark locations
    score: The average confidence score of the detected landmarks.
    """
    if face_box is None:  # Using the blazeface as face detector
        preds, scores, _ = fa.get_landmarks_from_image(frame, return_landmark_score=True)
    else:  # Using the retinaface as face detector
        face_box = np.expand_dims(np.array(face_box), axis=0)
        preds, scores, _ = fa.get_landmarks_from_image(frame, detected_faces=face_box, return_landmark_score=True)

    if preds is None:
        return 0, None, 0
    else:
        score = np.average(scores[0])
        shape = preds[0].astype(int)
        return 1, shape, score


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


def detect_frames_track(frames, video, fps):
    """

    :param frames: Image sequence (the video) [Type: list][Shape: (N, (H, W, C))]
    :param video: The name of the video which includes suffixes (str)
    :param fps: The fps of the video (Int)
    :return: calibrated_normalized_landmarks [Type: list][Shape: (N, 136)]
    """
    frames_num = len(frames)
    assert frames_num != 0
    frame_height, frame_width = frames[0].shape[:2]
    """
    Pre-process:
        - Detect the original face bounding-box and landmarks.
        - Normalize each face to a certain width. (For calibration)
        - Also normalize its corresponding landmarks locations and record the scale parameter (For visualization).
    """
    face_size_normalized = 400
    faces = []
    locations = []
    shapes_origin = []
    shapes_para = []  # Use to recover the shape in whole frame. ([x1, y1, scale_shape])
    skipped = 0  # Use to record how many frames have been discarded at the beginning of the video.
    """
    Note for [skipped]: We only discard the frames (no faces/landmarks are detected) at the beginning of the video.
    If the detection failed at an intermediate frame, we would set the former frame's results as its results.
    (And try to simulate this frame's landmark through landmark calibration.)
    Because of the diversity of the test videos, this section can easily be problematic.
    If you encounter problems, feel free to contact me on repo's issue.
    """

    face_boxes = None
    lm_scores = []

    print("Face Bounding-Box Detecting:")
    if shared.face_detector_selection == 'blazeface':
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='blazeface')
    else:
        from demo.utils.FaceDetector.functions import get_detector, predict_face_box_batch
        net = get_detector()  # Initialize the retinaface detector.
        face_boxes, confidence_scores = predict_face_box_batch(net, frames)
        assert len(frames) == len(face_boxes)
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='folder')

    print("Landmark Detecting:")
    for i in tqdm(range(frames_num)):
        frame = frames[i]

        if shared.face_detector_selection == 'blazeface':
            face_num, shape, score = predict_single_frame(frame, fa)
        else:
            face_box = face_boxes[i]
            face_num, shape, score = predict_single_frame(frame, fa, face_box)

        lm_scores.append(score)

        if face_num == 0:
            print("Landmark detection failed at index: {}".format(i))
            if len(shapes_origin) == 0:
                skipped += 1
                print("Skipped frame num:", skipped, ".Frame_num", frames_num)
                continue
            shape = shapes_origin[-1]

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
    Record the average confidence score of the detected landmarks in the whole video.
        - Useful for eliminate some failed samples (such as some in DFDC dataset)
    """
    avg_score = np.average(lm_scores)
    record = open(shared.log_file, mode='a')
    dt = datetime.now()
    record.write(video + ' ' + str(avg_score) + ' ' + dt.isoformat(sep='/') + '\n')
    record.close()

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
        print("Warning: Failed to detect the first {} frames, which will be skipped".format(skipped))
    if shared.use_visualization:
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        frame_size = (frames[0].shape[1], frames[0].shape[0])
        origin_video = cv2.VideoWriter(join(shared.visualize_path, video + "_origin.avi"),
                                       fourcc, fps, frame_size)
        track_video = cv2.VideoWriter(join(shared.visualize_path, video + "_track.avi"),
                                      fourcc, fps, frame_size)

        print("Visualizing")
        for i in tqdm(range(frames_num)):
            if i < skipped:
                # Skip the beginning frames
                continue
            frame_origin = frames[i].copy()
            frame_track = frames[i].copy()
            shape_origin = shapes_origin[i - skipped]
            para_shift = shapes_para[i - skipped][0:2]
            para_scale = shapes_para[i - skipped][2]
            shape_track = np.rint(locations_track[i - skipped] / para_scale + para_shift).astype(int)
            for (x, y) in shape_origin:
                cv2.circle(frame_origin, (x, y), 3, (0, 0, 255), -1)
            for (x, y) in shape_track:
                cv2.circle(frame_track, (x, y), 3, (0, 255, 0), -1)
            origin_video.write(frame_origin)
            track_video.write(frame_track)
        origin_video.release()
        track_video.release()

    # -------------------------------------------#
    """
    Landmark Alignment (DEPRECATED)
    """
    # aligned_landmarks = []
    # for i in locations_track:
    #     shape = landmark_align(i)
    #     shape = shape.ravel()
    #     shape = shape.tolist()
    #     aligned_landmarks.append(shape)
    # return aligned_landmarks
    # -------------------------------------------#
    """
    Landmark Normalization
        - Target: [-1, 1]
        - Updated at 2022/10/21
    """
    calibrated_normalized_landmarks = []
    for i in locations_track:
        normalized_base = face_size_normalized // 2
        shape = i - [normalized_base, normalized_base]
        shape = shape / normalized_base
        shape = shape.ravel()
        shape = shape.tolist()
        calibrated_normalized_landmarks.append(shape)

    return calibrated_normalized_landmarks
