# LRNet-Calibrator

The calibrator API for denoising the landmarks, which is adopted in LRNet.

# Requirements

- numpy
- python-opencv



# Getting started

> Because the landmarks detection techniques are evolving, you may use better detector to preprocessing the video data and obtain the landmarks. Therefore we provide an **plug-and-use** API for calibrating the landmarks and obtaining better training data.

### Usage

The API is in package `LC.py` (Landmark Calibrator). You can call it by:

```python
from LC import calibrator
results = calibrator(video_file, landmark_sequence)
```

The `video_file` is your video file name end with ".mp4", etc. The `landmark_sequence` is a sequence contains the landmark positions in each video frame with the shape of ![](http://latex.codecogs.com/gif.latex?N \times 68 \times 2). And this API return the calibrated and aligned landmark sequence `result` with the shape of ![](http://latex.codecogs.com/gif.latex?N \times 136). Note that the `result` is also normalized into 0~1, which can be used for training and testing the model.



### Example

For easier to understand the process, we provide an example in `./example/` folder. This video comes from DF(c23), and the corresponding landmark results are detected by [[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)] toolkit. We can achieve a success API by:

```python
from LC import calibrator, readin_csv
video_name = "./example/000_003"
landmark_sequence = readin_csv(video_name)
results = calibrator(video_name + '.mp4', landmark_sequence)
np.savetxt(video_name+".txt", results, fmt='%1.5f')
```

Then you would see a new `000_003.txt` in folder `./example/`

You can also uncomment the example code in `LC.py` (L288~291) and directly run it to get the same result.