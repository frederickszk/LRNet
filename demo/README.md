# LRNet-Demo
The demo of *LRNet*. It reproduces the whole procedure from a input video sample to its corresponding predicted label. 

# Folders & Codes

`./input`

The input sample videos. Here we provide a real and a fake video.

`./landmarks`

The extracted landmarks. Each video is corresponding to a .txt file.

`./model_weights`

The two-stream RNN's weight.

`./visualize`

The visualize results of landmrks extraction. The original landmarks detection (without calibration) is shown in "_origin.avi", and the calibrated one is shown in ".track.avi".


`extract_landmarks.py`

Perform the landmarks extraction and calibration.

`landmark_utils.py`

Some utils of landmarks extraction, Kalman filter, landmarks alignment and so on.

`calib_utils.py`(**DEPRECATED**. We now use OpenCV api to implement. Check Log 2021/5/18 for more details.)

Some utils of calibration module, including the LK operation and so on.

`classify.py`

Perform the classification on extracted landmarks. Including reading and embedding the data, building the two-stream RNN and inference the samples.

`shape_predictor_68_face_landmarks.dat` **(Need to download MANUALLY.)**

DLIB landmarks detector parameters. You can obtain it from this 
[Link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it at this directory.

# Requirements

## For landmarks extraction

- numpy
- [opencv-python](https://docs.opencv.org/4.5.5/da/df6/tutorial_py_table_of_contents_setup.html)
- tqdm
- [dlib](http://dlib.net/compile.html)
    - CPU version is sufficient.
    - We do not include the necessary detector model (~100.7Mb). You may download it at [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 
  and place it at this directory.


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

You can also specific your own input videos path by `-i` or enable the visualization by `-v`.

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

and you can see the prediction results (predicted label and score of the corresponding sample. Noticed the score that "0=real" and "1=fake".
