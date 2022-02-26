import os
from os.path import join
import time
from tqdm import tqdm
import numpy as np
from LC import calibrator, readin_csv


"""
Utils for OpenFace
"""


def calibrate_batch(video_path, csv_path, output_path):
    videos = os.listdir(csv_path)
    videos.sort()
    start = time.time()
    for video in tqdm(videos):
        # Pre-process the address
        if video.startswith(('.', '?')):
            continue
        video_name = video.split('.')[0]

        # Start from the break-point
        if os.path.exists(join(output_path, video_name + ".txt")):
            print(video_name + " exist, skipped")
            continue
        landmark_sequence = readin_csv(join(csv_path, video_name))
        results = calibrator(join(video_path, video_name + '.mp4'), landmark_sequence)

        if len(results) == 0:
            print("No face detected", video)
        else:
            np.savetxt(join(output_path, video_name + ".txt"), results, fmt='%1.5f')

    end = time.time()
    print("Calibration finished. Time cost is:", end - start)
