# Usage

If you want to use this application, I assume that you already know the basic scripting procedures. If this is not the case, please learn first the basic functions how to create a funscript!

First of all I want to mention that this application is not a fully automated solution, it's just an extension that helps you scripting **static camera scenes** much faster.

## Process

1. Select the Region of Interest in the Video where the action happens.
2. Select an features for the Woman and Men in the video, which should be tracked.
3. The application try to track the selected feature in the following video frames.
4. When the user stop the Tracking or the application can not find the selected feature in the next video frame the application calculate the difference between the predicted tracking boxes for each frame.
5. Now you can cut the raw tracking result e.g. to remove an tracker shift
6. Then the user set the minimum and maximum position for the tracked video sequence.
7. Finally set the desired post processing parameter.
8. The Lua Extension automatically import the result to OpenFunscripter.

## Options

## Unsupervised vs. Supervised Tracking

The Unsupervised Tracking simply track the selected feature in the video until the feature is not visible anymore or the user stop the tracking process.

The supervised tracking expand this function by an user input of an search area. As soon as the predicted tracking box leaves this area, we can trigger an event. This event can currently represent either an abort condition for the tracking or an ignore condition. Therefore you have to define the region where you expect the selected features in the future video frames for each tracked feature when you select the supervised tracking option. When you use the supervised tracking process as abort condition it will increases the quality of the generated funscript actions due to the additional quality check. With the ignore condition we ignore all tracking prediction outside the specified area and use the last valid tracking position inside the specified area until the tracking insert the defined supervised tracking area again. This method is useful to script actions where the actor not always have contact.

In short: The Supervised options track a feature inside a specified area. It asks for pair of parameters: first the small area containing the feature to track, then a larger bounding area that must contain the feature area at every moment. If you use more than 1 tracker then it goes `(feature1, bounding area1)`, `(feature2, bounding area2)` etc.
You can then specify blocking/non blocking: the first means that when the feature "escape" the bounding box the process stop, the second option keeps going and if the feature gets back into that bounding box it continues tracking the movement.

## Settings

All important settings can be set in the settings dialog before the tracking starts.

Description:

- Video Type: Select the type of your video format.
- Tracking Metric: Select the metric for the movement detection.
- Tracking Method: Select the desired tracking method. Available methods depend on the selected `Tracking Metric` setting.
- Processing Speed: Select the processing speed. This option set the skip frames factor. For an higher processing speed we track only every `n % skip_frame == 0` frame and interpolate the skipped frames. Recommend settings is `1 (normal)` for 60 FPS Videos (and above) and `0 (accurate)` for Videos with 30 FPS or less.
- Number of Tracker: Select the number of trackers per target. We use the average of all tracker for the output. Using a higher number of Tracker should increase the accuracy. For most video scenes i recommend to use only `1` tracker because the accuracy increases only slightly but the processing speed is significantly higher.
