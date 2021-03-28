# LRNet
Landmark Recurrent Network: An efficient and robust framwork for Deepfakes detection
> The implementation for paper: **Improving the Efficiency and Robustness for Deepfakes Detection through Precise Geometric Features (CVPR2021)**.


# Description

> At current version, it is still the demo code of our proposed framework *LRNet*. It reproduces the whole procedure from a input video sample to its corresponding predicted label. We will update soon to optimize some components of it.

## Folders

`input`

The input sample videos. Here we provide a real and a fake video.

`landmarks`

The extracted landmarks. Each video is corresponding to a .txt file.

`model_weights`

The two-stream RNN's weight.

`visualize`

The visualize results of landmrks extraction. The original landmarks detection (without calibration) is shown in "_origin.avi", and the calibrated one is shown in ".track.avi".

## Codes

`extract_landmarks.py`

Perform the landmarks extraction and calibration.

`landmark_utils.py`

Some utils of landmarks extraction, Kalman filter, landmarks alignment and so on.

`calib_utils.py`

Some utils of calibration module, including the LK operation and so on.

`classify.py`

Perform the classification on extracted landmarks. Including reading and embedding the data, building the two-stream RNN and inference the samples.

`shape_predictor_68_face_landmarks.dat`

DLIB landmarks detector parameters. **(Need to download MANUALLY.)**

# Requirements

## For landmarks extraction

- argparse
- numpy
- python-opencv
- tqdm
- imutils
- dlib
    - CPU version is sufficient.
    - We do not include the necessary detector model (~100.7Mb). You may download it at  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- PyTorch > 1.3
    - CPU version is sufficient.

## For classification

- numpy
- tqdm
- TensorFlow >2.0
    - CPU version is sufficient



# Useage

## Landmarks extraction

To extract the landmarks from input videos, please ensure that all the samples are put in `./input` folder and run

```python
python extract_landmarks.py
```

You can also specific your own input videos path by `-i` or disable the visualization by `-v`.

For detailed, please consult

```python
python extract_landmarks.py -h
```

> 1. Noted that the generated landmarks and visualized detection videos are put in the same folder as the code files, i.e. the "./", which cannot be specified. 
> 2. We have provided generated landmarks extraction results and visualization videos in `./landmarks` and `./visualize`. Also you can delete these folders and regenerate them by yourself.

## Classification

Before classification, make sure that the landmarks have been extracted and put in `./landmarks` folder.

Execute the following command:

```python
python classify.py
```

and you can see the prediction results (predicted label and score of the corresponding sample. Noticed the score that "0=real" and "1=fake".)



# TODO

- [x] Upload the demo
- [ ] Update the face/landmark detector.
- [ ] Update the calibration module. 
- [ ] Optimize the RNN part.



# Notice


1. This is only a demo to show the whole detection framework, thus several components of our codes need optimization, and we are continuing to organize the code. The calibration module will be further designed as an API to calibrate the landmarks extracted by any other landmark detectors.
2. Only TensorFlow is involved in building the network. PyTorch is used to extract patches from an image with bilinear interpolation. However, it's slightly slower in CPU and we consider to reimplement it with Numpy and openCV for easier use.
