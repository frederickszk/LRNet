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

> To go through the whole procedure of *LRNet*'s forgery detection (from an input video sample to its corresponding predicted label),
> you can enter the `./demo/` for more details.
>
> For easily training and evaluating the *LRNet*, you can enter the `./training/` for more details.
>
> If you would like to use your own landmark detector to generate the dataset, you can enter the `./calibrator/` for more details.




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

- [x] Upload the demo.
  - [x] Organize the codes into the sub-folder. (2022/3/16)
- [x] Update the calibration module. 
  - [x] Update the pyramidal LK with numpy and openCV.
  - [x] Implement the pyramidal LK with openCV API.
  - [x] Provide API for easier use.
  - [x] Provide utils for calibrating the dataset (under OpenFace workflow). (2022/2/27)
- [ ] Optimize the model.
  - [x] Update the training codes.
  - [x] For PyTorch version, optimize the dropout layer.
  - [x] Greatly refactor the training and evaluation codes for PyTorch version. (2022/3/10)
  - [x] Update the _Jupyter Notebook_ version training/evaluation codes. (2022/3/20, **LATEST**)
  - [x] Update the weights for the whole FF++ dataset. (2022/4/28, **LATEST**)
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

<details>
    <summary> 2022/3/16</summary>
	<ul>
        <li>
            <strong>Organize the demo codes.</strong> Now the demo codes are gathered in the <code>./demo/</code> folder
            , making the project's home page clean and tidy. 
        </li>
        <li>
            <strong>Discussion: the selection of landmark detector in demo.</strong> 
            We also try several landmark detectors and find that <code>Dlib</code> is relatively good solution. 
            Although it's accuracy and stability are not the SOTA, it helps alleviate preparing  
            burdensome dependencies (especially for deep-learning-based models). Besides, it's inference speed is fast
            and suitable for reproducing the LRNet's framework. 
            You could also replace it with other advanced landmark detectors.
        </li>
    </ul>
</details>






