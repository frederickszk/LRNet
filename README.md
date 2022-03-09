# LRNet
Landmark Recurrent Network: An efficient and robust framwork for Deepfakes detection
> The implementation for paper: **Improving the Efficiency and Robustness for Deepfakes Detection through Precise Geometric Features (CVPR2021)**. 
> 
>[[Paper Link](https://arxiv.org/abs/2104.04480)]

# Overview
LRNet is a light-weight and effective Deepfakes detection framework. 
It analyzes the sequence of geometric features in facial videos 
to capture the *temporal artifacts* (such as unnatural expressions) and make predictions.

It takes three steps to discriminate a forged face video:

1. **Extract the 2-D facial landmark sequence from the video**. It can be conveniently realized by some open-source toolkit 
   like Dlib, OpenFace, MediaPipe, etc.
2. **De-noise the sequence by our carefully-devised calibration module**. It utilizes optical flow and Kalman filter
techniques to alleviate the noise caused by different landmark detectors.
3. **Embed the landmark sequence into two feature sequences and input them into two RNN**. The output of two RNN 
   are merged to obtain the final prediction.
   
<div align="center">
<img src="https://s2.loli.net/2022/03/01/TFZO86nwbiry2xl.png" alt="LRNet.png" width="80%;"/>
</div>

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

# TODO & Update List

- [x] Upload the demo (2021/3/29)
- [ ] Update the face/landmark detector in demo.
- [x] Update the calibration module. 
  - [x] Update the pyramidal LK with numpy and openCV.
  - [x] Implement the pyramidal LK with openCV API.
  - [x] Provide API for easier use.
  - [x] Provide utils for calibrating the dataset (under OpenFace workflow). (2022/2/27, **LATEST**)
- [ ] Optimize the RNN part.
  - [x] Update the training codes.
  - [x] For PyTorch version, optimize the dropout layer.
  - [x] Greatly refactor the training and evaluation codes for PyTorch version. (2022/3/10, **LATEST**)
  - [ ] Model structure exploration
- [ ] Gradually release the datasets
  - [ ] FF++
    - [x] c23
    - [ ] raw
    - [ ] c40

# Log

<details>
    <summary> 2021/5/18</summary>
	<ul>
        <li>
            <strong>Update the model weights.</strong> Now we provide weights trained on FF++ (./model<em>weights/ff) and Deeperforensics-1.0 (./model</em>weights/deeper). We use <code>deeper</code> by default, because the given examples are from this dataset. You can change to <code>ff</code> in <code>classify.py</code> L107--L113.
        </li>
     	<li>
            <strong>Provide faster LK tracker.</strong> We now use openCV optical api to achieve LK track. It would be faster and more stable.
		</li>
        <li>
            <strong>Update GPU support.</strong> We restrict Tensorflow not to use up the memory of GPU when it is supported. Although CPU is enough for inference. The GPU memory occupation may be &lt; 1G.
        </li>
    </ul>
</details>

<details>
    <summary> 2021/11/13</summary>
	<ul>
        <li>
            <strong>Update the training and testing codes.</strong>
        </li>
     	<li>
            <strong>Release FF++(c23) landmark datasets.</strong> 
		</li>
        <li>
            <strong>Update the plug-and-use landmark calibrator API.</strong> 
        </li>
    </ul>
</details>

<details>
    <summary> 2022/3/10</summary>
	<ul>
        <li>
            <strong>Provide utils for calibrating the dataset.</strong> Currently it is devised for OpenFace work flow. We consider add more workflow supports in the future.
        </li>
     	<li>
            <strong>Greatly refactor the training and evaluation codes for PyTorch version.</strong> They can help perform more flexible training and evaluating strategies, which is convenient for further research.
		</li>
    </ul>
</details>







