# LRNet
Landmark Recurrent Network: An efficient and robust framwork for Deepfakes detection
> The implementation for paper: **Improving the Efficiency and Robustness for Deepfakes Detection through Precise Geometric Features (CVPR2021)**. 
> 
>[[Paper Link](https://arxiv.org/abs/2104.04480)]



# Description

> In this root directory, there is the demo code of our proposed framework *LRNet*. It reproduces the whole procedure from a input video sample to its corresponding predicted label. 
>
> For easily training and evaluating the *LRNet*, you can enter the `./training/` for more details.
>
> If you would like to use your own landmark detector to generate the dataset, you can enter the `./calibrator` for more details.

## Folders

`input`

The input sample videos. Here we provide a real and a fake video.

`landmarks`

The extracted landmarks. Each video is corresponding to a .txt file.

`model_weights`

The two-stream RNN's weight.

`visualize`

The visualize results of landmrks extraction. The original landmarks detection (without calibration) is shown in "_origin.avi", and the calibrated one is shown in ".track.avi".

`training` **(new)**

Training and evaluation codes.

`calibrator` **(new)**

Landmark calibrator API.

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
    - We do not include the necessary detector model (~100.7Mb). You may download it at [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) .


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



# TODO & Update List

- [x] Upload the demo (2021/3/29)
- [ ] Update the face/landmark detector.
- [x] Update the calibration module. 
  - [x] Update the pyramidal LK with numpy and openCV. (2021/4/14)
  - [x] Implement the pyramidal LK with openCV API. (2021/5/18)
  - [x] Provide API for easier use. (2021/11/13, **LATEST**)
- [ ] Optimize the RNN part.
  - [x] Update the training codes. (2021/11/13, **LATEST**)
  - [x] For PyTorch version, optimize the dropout layer. (2021/11/13, **LATEST**)
  - [ ] Model structure exploration
- [ ] Gradually release the datasets
  - [ ] FF++
    - [x] c23 (2021/11/13, **LATEST**)



# Log
- 2021/5/18
  - **Update the model weights.** Now we provide weights trained on FF++ (./model_weights/ff) 
    and Deeperforensics-1.0 (./model_weights/deeper). We use `deeper` by default, 
    because the given examples are from this dataset. You can change to `ff` in `classify.py` 
    L107--L113.
  - **Provide faster LK tracker.** We now use openCV optical api to achieve LK track. It would be faster and more stable.
  - **Update GPU support.** We restrict Tensorflow not to use up the memory of GPU when it is supported. Although CPU is enough for inference. The GPU memory occupation may be < 1G.
- 2021/11/13
  - **Update the training and testing codes.**
  - **Release FF++(c23) landmark datasets.**
  - **Update the plug-and-use landmark calibrator API.**



# Citation

If our work helps your research, welcome for the citation:

```latex
@inproceedings{sun2021improving,
  title={Improving the Efficiency and Robustness of Deepfakes Detection through Precise Geometric Features},
  author={Sun, Zekun and Han, Yujie and Hua, Zeyu and Ruan, Na and Jia, Weijia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3609--3618},
  year={2021}
}
```



