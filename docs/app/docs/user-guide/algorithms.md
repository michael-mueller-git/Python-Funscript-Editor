# Algorithms

## Create Funscript Action using OpenCV Tracker

Idea: By using [OpenCV Tracker](https://learnopencv.com/object-tracking-using-opencv-cpp-python/), we can determine the relative movements in a static camera setup and map them into Funscript actions using simple signal processing.

### Limitations

- Static camera setup
- Fixed reference point of relative movement in video required
- No video cuts within a tracking sequence allowed
- No change of position of the performers
- Features in the video which are visible in all following frames of the tracking sequence required.

### Process

1. Selection of the features for the Woman and Men in the video, which should be tracked.
2. Predict the feature positions in the following video frames by OpenCV Tracker.
3. Calculate the difference between the predicted tracking boxes.
4. Map the relative difference to an absolute difference score by user input.
5. Filter all local min and max points to get the final action positions for the Funscript.

### Improvements

- You can change the OpenCV tracker which predicts the position. OpenCV offers several trackers which differ in prediction accuracy and processing speed. See also [OpenCV Tracker](https://learnopencv.com/object-tracking-using-opencv-cpp-python/).

- You can set the number of frames that are interpolated by the `skip_frames` parameter. 0 means that the OpenCV tracker delivers a prediction for each frame. This is slower but more accurate. Or if greater than zero, the individual frames are skipped and then the tracking boxes are interpolated, which increases the processing speed but decreases the accuracy. I have set the value to 1, i.e. every 2nd frame is skipped and interpolated. Which provides a good mix of accuracy and speed.

- It is recommended to use a low resolution video e.g. 1080p or 4K for generating the funscript actions, as the processing speed is higher.
