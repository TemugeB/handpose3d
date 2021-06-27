**Real time 3D hand pose estimation using MediaPipe **

This is a demo on how to obtain 3D coordinates of hand keypoints using MediaPipe and two calibrated cameras. Two cameras are required as there is no way to obtain 3D coordinates from a single camera. My blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html).

![input1](media/output_kpts.gif "input1") ![input2](media/output2_kpts.gif "input2") 
![output](media/fig_0.gif "output")

**MediaPipe**  
Install mediapipe in your virtual environment using:
```
pip install mediapipe
```

**Requirements**  
```
Python3.8
Opencv
matplotlib
```

**Usage: Getting real time 3D coordinates**  
As a demo, I've included two short video clips and corresponding camera calibration parameters. Simply run as:
```
python handpose3d.py
```
If you want to use webcam, call the program with camera ids. For example, cameras registered to 0 and 1:
```
python handpose3d.py 0 1
```
Make sure the corresponding camera parameters are also updated for your cameras.

The 3D coordinate in each video frame is recorded in ```frame_p3ds```. Use this for real time application. If keypoints are not found, then the keypoints are recorded as (-1, -1, -1). **Warning**: The code also saves keypoints for all previous frames. If you run the code for long periods, then you will run out of memory. To fix this, remove append calls to: