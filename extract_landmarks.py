import argparse
import os
import numpy as np
import cv2
from landmark_utils import detect_frames_track


def detect_track(input_path, video, use_visualization, visualize_path):
    vidcap = cv2.VideoCapture(os.path.join(input_path, video))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            break
    raw_data = detect_frames_track(frames, fps, use_visualization, visualize_path, video)

    vidcap.release()
    return np.array(raw_data)


def main(args):

    input_path = args.input_path
    visualize = args.visualize

    """
    Prepare the environment
    """
    assert os.path.exists(input_path), "Input path does not exist."
    output_path = "./landmarks/"
    if not os.path.exists(output_path):
        os.makedirs("./landmarks/")

    if visualize:
        print("Settings: Visualize the extraction results.")
    else:
        print("Settings: NOT visualize the extraction results.")

    # use_visualization:
    visualize_path = "./visualize/"
    if visualize:
        if not os.path.exists(visualize_path):
            os.makedirs("./visualize/")

    videos = os.listdir(input_path)
    for video in videos:
        if video.startswith('.'):
            continue

        print("Extract landmarks from {}.".format(video))
        raw_data = detect_track(input_path, video, visualize, visualize_path)

        if len(raw_data) == 0:
            print("No face detected", video)
        else:
            np.savetxt(output_path + video + ".txt", raw_data, fmt='%1.5f')
        print("Landmarks data saved!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract landmarks sequences from input videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_path', type=str, default='./input/',
                        help="Input videos path (folder)"
                        )
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="If visualize the extraction results."
                        )
    args = parser.parse_args()
    main(args)
