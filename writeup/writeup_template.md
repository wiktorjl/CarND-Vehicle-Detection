

[//]: # (Image References)
[image1]: ./cars_notcars.png
[image2]: ./hog.png
[image3]: ./grid.png
[image4]: ./detect_car.png
[image5]: ./heat.png
[image6]: ./heat_label_final.png
[video1]: ./final.mp4
[code]: ../vehicle_detection.py
## Vehicle detection
##### Wiktor Lukasik

[Source code][code]

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I have read complete data set (KTTI and GTI) into memory, splitting it into two sets, `vehicle` and `non-vehicle`. The code for this starts in method `regenerate_features` (line 315), where I read paths by making a call to `get_file_paths` (line 233) and then read actual image by calling `read_training_image` (line 241) via `extract_features_from_paths` (line 247). Here are random images for both classes:

![alt text][image1]

As images are extracted, I read features from each, including HOG features in method `extract_features` (line 116-130). Here are relevant parameters:

`color_space: YUV,
orient: 11,
pix_per_cell: 8,
cell_per_block: 2,
hog_channel: 'ALL'`

I have initially explored various options for these parameters by reviewing their visualizations. However, perceived quality in the visualized HOG features did not always translate to better image classification. For this reason, I have modified HOG parameters and validated them at a later stage, when classifying images. Here is example of HOG features using the above parameters, you can clearly see outlines of vehicles in the first three images:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I did not test all combinations as there would be too many (100-150 combinations). Instead, I have optimized them independently and estimated the correct combination.

I have later adjusted them again when classifying images. Overall, I did not find much value in tweaking `orient`, `pix_per_cell`, or `cell_per_block`. Changing color space did impact performance (RGB was great for detecting cars but did not do well with bright and dark areas, while YUV was a good balance - used it in the previous vision project). Also, limiting to only single HOG channel improved feature extraction speed, but was detrimental to quality.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Model training is in method `train_model` on line 276. I used code from class and for the most part did not modify it. The only addition was penalty parameter C, which I have set to a large value to lower bias and increase variance.

I am using all available features (hog, color, spatial) and in most arrangements of configuration values my model achieves very high accuracy (98-99%) so I did not spend much time trying to improve it. Instead, I prioritized other methods that help deal with misclassification.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My approach is to use larger windows in the lower parts of the image and smaller in the upper part. I limit my search to 390-600 vertical space. Since in the video the car is in the left lane, I also ignore left 400 pixels with assumption that would change if the car moves to another lane.

Smaller windows are 64 pixels, larger are 128 pixels. I have experimented with both larger and smaller windows but results were poor.

Relevant code is in method `detect_cars`, starting in line 360.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As noted above, I have used parameter C to improve my classifier. Also, instead of using binary classification of `predict` method, I have used `decision_function` to get the confidence score of the prediction. I have determined empirically that threshold of 0.8 does a good job sieving out false positives.

Example of classification before applying heatmap:
![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Additionally, I smooth the results by always looking at the last 30 windows.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and final boundaries:
![alt text][image6]





---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues I have encountered:
* Too many false positives - addressed with parameter C, confidence threshold, heatmaps
* Issues detecting white car - tweaking parameters


Ideas for improvement:
* Lane detection - look for objects only within lanes.
* Adjusting for light - this solution uses static parameters (color space, etc..) but road conditions change and we could adjust these parameters to match current conditions.
* Predicting motion - in my final video I loose track of the white car and then find it again. Knowing it is a car moving in a lane, I could keep the bounding box for a while.
* Given known approximate sizes of cars and location of lanes, we could do better job of predicting window sizes.
* Stereoscopic camera could produce images that are easier to analyze.
* Convolutional Neural Net could do better job with image recognition
