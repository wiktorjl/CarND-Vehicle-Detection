def demo_detection_image(svc, scaler, config):
    # img = demo_search_window(svc, scaler)
    # plt.imshow(img)
    # plt.show()
    # cv2.waitKey(2000)

    f, (ax1,ax2,ax3) = plt.subplots(3, 2, figsize=(24, 9))
    f.tight_layout()
    ax1[0].imshow(demo_search_window(imgpath="test_images/test1.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    ax1[1].imshow(demo_search_window(imgpath="test_images/test2.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    ax2[0].imshow(demo_search_window(imgpath="test_images/test3.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    ax2[1].imshow(demo_search_window(imgpath="test_images/test4.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    ax3[0].imshow(demo_search_window(imgpath="test_images/test5.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    ax3[1].imshow(demo_search_window(imgpath="test_images/test6.jpg", svc=svc, scaler=scaler, config=config, average=False), cmap="gray")
    plt.show()

    cv2.waitKey(1000)

def demo_draw_window(imgpath): #y_start_stop, xy_size=(64,64), color=(255,255,0)):
    image = mpimg.imread(imgpath)

    sets = [
        [[390, 500], (64,64), (255, 255, 0), (0.75,0.75)],
        [[390, 600], (128,128), (255,0,255), (0.5,0.5)]
        # [[390, 700], (200,200), (0,255,0), (0.75,0.75)]
    ]


    winn = 0
    for s in sets:
        windows = slide_window(image, y_start_stop=s[0], xy_window=s[1], xy_overlap=s[3])
        image = draw_boxes(image, windows, color=s[2])
        winn += len(windows)
    print("Count = ", winn)

    return image

def demo_cars_not_cars():
    cars, notcars = get_file_paths()
    random.shuffle(cars)
    random.shuffle(notcars)
    cars = cars[:6]
    notcars = notcars[:6]
    cars = [cv2.imread(x) for x in cars]
    notcars = [cv2.imread(x) for x in notcars]

    f, (ax1, ax2) = plt.subplots(2, 6, figsize=(24, 9))
    f.tight_layout()
    ax1[0].imshow(cars[0])
    ax1[1].imshow(cars[1])
    ax1[2].imshow(cars[2])
    ax1[3].imshow(cars[3])
    ax1[4].imshow(cars[4])
    ax1[5].imshow(cars[5])
    ax2[0].imshow(notcars[0])
    ax2[1].imshow(notcars[1])
    ax2[2].imshow(notcars[2])
    ax2[3].imshow(notcars[3])
    ax2[4].imshow(notcars[4])
    ax2[5].imshow(notcars[5])
    plt.show()

    cv2.waitKey(1000)

def demo_hog():
    cars, notcars = get_file_paths()
    random.shuffle(cars)
    random.shuffle(notcars)
    cars = cars[:6]
    notcars = notcars[:6]
    cars = [cv2.imread(x) for x in cars]
    notcars = [cv2.imread(x) for x in notcars]

    def hog(image):
        return get_hog_features(img=image[:, :, 0], orient=config["orient"], pix_per_cell=config["pix_per_cell"],
                         cell_per_block=config["cell_per_block"], vis=True, feature_vec=False)[1]

    hcars = [hog(x) for x in cars]
    hnotcars = [hog(x) for x in notcars]


    f, (ax1, ax2) = plt.subplots(2, 6, figsize=(24, 9))
    f.tight_layout()
    ax1[0].imshow(cars[0], cmap="gray")
    ax1[0].set_title("Car")
    ax1[1].imshow(cars[1], cmap="gray")
    ax1[1].set_title("Car")
    ax1[2].imshow(cars[2], cmap="gray")
    ax1[2].set_title("Car")
    ax1[3].imshow(notcars[3], cmap="gray")
    ax1[3].set_title("Not a Car")
    ax1[4].imshow(notcars[4], cmap="gray")
    ax1[4].set_title("Not a Car")
    ax1[5].imshow(notcars[5], cmap="gray")
    ax1[5].set_title("Not a Car")

    ax2[0].imshow(hcars[0], cmap="gray")
    ax2[0].set_title("Car")
    ax2[1].imshow(hcars[1], cmap="gray")
    ax2[1].set_title("Car")
    ax2[2].imshow(hcars[2], cmap="gray")
    ax2[2].set_title("Car")
    ax2[3].imshow(hnotcars[3], cmap="gray")
    ax2[3].set_title("Not a Car")
    ax2[4].imshow(hnotcars[4], cmap="gray")
    ax2[4].set_title("Not a Car")
    ax2[5].imshow(hnotcars[5], cmap="gray")
    ax2[5].set_title("Not a Car")


    plt.show()

    cv2.waitKey(1000)


def demo_heat():
    cars = ["frames/preheat1.jpg", "frames/preheat2.jpg", "frames/preheat3.jpg"]
    cars = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in cars]

    sets = [
        [[390, 500], (64,64), (255, 255, 0), (0.75,0.75)],
        [[390, 600], (128,128), (255,0,255), (0.5,0.5)]
        # [[390, 700], (200,200), (0,255,0), (0.75,0.75)]
    ]

    heatimgs = []
    labels_img = []
    labeled_imgs = []
    for car in cars:
        windows = []
        for s in sets:
            # windows = slide_window(img, y_start_stop=[400, 528], xy_window=(64,64))
            # windows.extend(slide_window(img, y_start_stop=[400, 656], xy_window=(128,128)))
            windows.extend(
                slide_window(car, x_start_stop=[400, 1280], y_start_stop=s[0], xy_window=s[1], xy_overlap=s[3]))

        hot_windows = search_windows(img=car, windows=windows, clf=model, scaler=scaler,
                                     color_space=config["color_space"],
                                     spatial_size=config["spatial_size"], hist_bins=config["hist_bins"],
                                     orient=config["orient"],
                                     pix_per_cell=config["pix_per_cell"], cell_per_block=config["cell_per_block"],
                                     hog_channel=config["hog_channel"], spatial_feat=config["spatial_feat"],
                                     hist_feat=config["hist_feat"], hog_feat=config["hog_feat"])


        prev_windows.append(hot_windows)
        heatmap_img = np.zeros_like(car[:, :, 0])
        heatmap_img = add_heat(heatmap_img, hot_windows)
        heatimgs.append(heatmap_img)
        # heatmap_img = apply_threshold(heatmap_img, 7)
        # labels = label(heatmap_img)
        # i, r = draw_labeled_bboxes(output_img, labels)
        # return i
        labels= label(heatmap_img)
        # labels = labels.astype(np.float32)
        labels_img.append(labels[0])
        labeled_img, r = draw_labeled_bboxes(car, labels)
        labeled_imgs.append(labeled_img)


    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 3, figsize=(24, 9))
    f.tight_layout()
    ax1[0].imshow(cars[0], cmap="gray")
    ax1[0].set_title("Car 1")
    ax1[1].imshow(cars[1], cmap="gray")
    ax1[1].set_title("Car 2")
    ax1[2].imshow(cars[2], cmap="gray")
    ax1[2].set_title("Car 3")

    ax2[0].imshow(heatimgs[0], cmap="hot")
    ax2[0].set_title("Heatmap 1")
    ax2[1].imshow(heatimgs[1], cmap="hot")
    ax2[1].set_title("Heatmap 2")
    ax2[2].imshow(heatimgs[2], cmap="hot")
    ax2[2].set_title("Heatmap 3")

    ax3[0].imshow(labels_img[0], cmap="gray")
    ax3[0].set_title("Label 1")
    ax3[1].imshow(labels_img[1], cmap="gray")
    ax3[1].set_title("Label 2")
    ax3[2].imshow(labels_img[2], cmap="gray")
    ax3[2].set_title("Label 3")

    ax4[0].imshow(labeled_imgs[0], cmap="gray")
    ax4[0].set_title("Final 1")
    ax4[1].imshow(labeled_imgs[1], cmap="gray")
    ax4[1].set_title("Final 2")
    ax4[2].imshow(labeled_imgs[2], cmap="gray")
    ax4[2].set_title("Final 3")

    plt.show()

    cv2.waitKey(1000)

def demo_sliding_window():
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(demo_draw_window("test_images/test1.jpg"), cmap="gray")
    # ax1[1].imshow(demo_draw_window("test_images/test2.jpg"), cmap="hot")
    # ax2[0].imshow(demo_draw_window("test_images/test3.jpg"), cmap="hot")
    # ax2[1].imshow(demo_draw_window("test_images/test4.jpg"), cmap="hot")
    # ax3[0].imshow(demo_draw_window("test_images/test5.jpg"), cmap="hot")
    # ax3[1].imshow(demo_draw_window("test_images/test6.jpg"), cmap="hot")
    plt.show()

    cv2.waitKey(1000)
