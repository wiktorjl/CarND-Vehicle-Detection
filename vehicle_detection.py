import pickle
import glob
import pickle
import sys
import time

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from demo import *

config = {
    "color_space": 'YUV',
    "orient": 11,
    "pix_per_cell": 8,
    "cell_per_block": 2,
    "hog_channel": 'ALL',
    "spatial_size": (16, 16),
    "hist_bins": 16,
    "spatial_feat": True,
    "hist_feat": True,
    "hog_feat": True
}


### IMAGE STUFF
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(img, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat, vis):

    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    feature_image = feature_image.astype(np.float32) / 255

    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # 8) Append features to list
        img_features.append(hog_features)
    # 9) Return concatenated array of features
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler, color_space='YCrCb',
                   spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    n_miss = 0
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = extract_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat, vis=False)

        # print("# of features: ", len(features   ))
        # 5) Scale extracted features to be fed to classifier
        # print("XXX ", np.array(features).reshape(1, -1).shape)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        # prediction = clf.predict(test_features)
        decision = clf.decision_function(test_features)
        # prediction = clf.predict_proba(test_features)
        # 7) If positive (prediction == 1) then save the window
        # if prediction[0][1] > 0.8:
        if decision > 0.8:
                on_windows.append(window)
        else:
            n_miss += 1
    # 8) Return windows for positive detections
    if n_miss > 0:
        r = float(len(on_windows))/n_miss
    else:
        r = len(on_windows)
    # print("Search window hit-miss ratio:", r)
    return on_windows



def get_file_paths():
    cars = np.array(glob.glob('train_data/vehicle/*/*.png'))
    notcars = np.array(glob.glob('train_data/nonvehicle/*/*.png'))

    print("File paths: Cars = ", len(cars), ", Non-Cars = ", len(notcars))

    return cars, notcars

def read_training_image(path):
    image = cv2.imread(path)
    cimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # feature_image = image.astype(np.float32) / 255
    return cimage, image

def extract_features_from_paths(image_paths):
    ret = []
    cnt = 0
    for image_path in image_paths:
        sys.stdout.write("\rExtracting feature: " +  str(cnt) + "/" + str(len(image_paths)))
        sys.stdout.flush()
        cnt += 1
        image, orig_image = read_training_image(image_path)

        if cnt == 1:
            print("Image outputed as example:", image_path)
            cv2.imwrite("example_train_image.png", orig_image)

        features = extract_features(image, config["color_space"], config["spatial_size"], config["hist_bins"], config["orient"],
                     config["pix_per_cell"], config["cell_per_block"], config["hog_channel"], config["spatial_feat"],
                     config["hist_feat"], config["hog_feat"], False)
        ret.append(features)
    print("\n")
    return ret


def store_features(list_of_features, name):
    print("Storing features in", name)
    pickle.dump(list_of_features, open(name, 'wb'))

def load_features(name):
    print("Loading features for", name)
    return pickle.load(open(name, 'rb'))

def train_model():
    print("Training new model")
    car_features = load_features("car_features")
    not_car_features = load_features("not_car_features")
    print("Loaded features: Car = ", str(len(car_features)), ", Non-Car: ", str(len(not_car_features)))

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    Y  = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    print("Creating scaler....")
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    print("Done.")

    r = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y,  test_size=0.2, random_state=r)

    print("Now training model...")
    svc = LinearSVC(C = 1000)
    t = time.time()
    svc.fit(X_train, y_train)
    te = time.time()
    print("Done! Time to train: ", round(te - t, 2))
    score = svc.score(X_test, y_test)
    print(score)

    return {"score": score}, svc, X_scaler


def save_model(model, scaler):
    pickle.dump(model, open("model", 'wb'))
    pickle.dump(scaler, open("scaler", 'wb'))
    print("Model saved.")

def load_model():
    model = pickle.load(open("model", 'rb'))
    scaler = pickle.load(open("scaler", 'rb'))
    return model, scaler

def regenerate_features():
    paths_cars, paths_not_cars = get_file_paths()
    features_cars = extract_features_from_paths(paths_cars)
    features_not_cars = extract_features_from_paths(paths_not_cars)
    store_features(features_cars, "car_features")
    store_features(features_not_cars, "not_car_features")


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image and final rectangles
    return img, rects


prev_windows = []
def detect_cars(imgpath=None, img=None, svc=None, scaler=None, average=False, config=None):
    global prev_windows

    if img is None:
        vehicle_img = cv2.imread(imgpath)
        vehicle_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
    else:
        vehicle_img = img

    output_img = np.copy(vehicle_img)

    sets = [
        [[390, 500], (64,64), (255, 255, 0), (0.75,0.75)],
        [[390, 600], (128,128), (255,0,255), (0.5,0.5)]
        # [[390, 700], (200,200), (0,255,0), (0.75,0.75)]
    ]

    windows = []
    for s in sets:
        windows.extend(slide_window(vehicle_img, x_start_stop=[400, 1280], y_start_stop=s[0], xy_window=s[1], xy_overlap=s[3]))

    hot_windows = search_windows(img=vehicle_img, windows= windows, clf=svc, scaler=scaler, color_space = config["color_space"],
                                 spatial_size = config["spatial_size"], hist_bins = config["hist_bins"], orient = config["orient"],
                                 pix_per_cell = config["pix_per_cell"], cell_per_block = config["cell_per_block"],
                                 hog_channel = config["hog_channel"], spatial_feat = config["spatial_feat"],
                                 hist_feat = config["hist_feat"], hog_feat = config["hog_feat"])


    if average:
        prev_windows.append(hot_windows)
        prev_windows = prev_windows[-30:]
        heatmap_img = np.zeros_like(vehicle_img[:, :, 0])

        if average:
            for pv in prev_windows:
                heatmap_img = add_heat(heatmap_img, pv)
        else:
            heatmap_img = add_heat(heatmap_img, hot_windows)


        heatmap_img = apply_threshold(heatmap_img, 7)
        labels = label(heatmap_img)

        i, r = draw_labeled_bboxes(output_img, labels)
        return i
    else:
        return draw_boxes(output_img, hot_windows)


def run_train_model():
    regenerate_features()
    model_props, model, scaler = train_model()
    print(model_props)
    save_model(model, scaler)

def run_demo_image_detection():
    model, scaler = load_model()
    demo_detection_image(model, scaler, config)


def process_image_save_frames(img):
    return process_image(img, True)


def process_image(img, save_frames=False):
    if save_frames:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frames/frame" + str(int(round(time.time() * 1000))) + ".jpg", rgb)

    return detect_cars(img=img, svc=model, scaler=scaler, average=True, config=config)

def output_video(src="test_video.mp4", dst = "test_videos_output/VIDEO2.mp4", start=None, stop=None, save_frames=False):
    if start is not None and stop is not None:
        clip1 = VideoFileClip(src).subclip(start,stop)
    else:
        clip1 = VideoFileClip(src)

    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(dst, audio=False)


def run_pipeline():
    output_video() #start=38, stop=42, save_frames=True)

model, scaler = load_model()
run_pipeline()
# run_train_model()
# demo_cars_not_cars()
# demo_hog()
# demo_sliding_window()
# demo_detection_image(model, scaler, config)
# run_demo_image_detection()
# demo_heat()
