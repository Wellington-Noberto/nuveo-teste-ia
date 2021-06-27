import os
from shutil import copyfile
import cv2
import numpy as np
import pandas as pd
import json


def generate_csv(results_array):
    """ Generates a .csv file based on the reference model provided
    Args:
        results_array (list of list): Returns the following parameters, (img_path, center_x, center_y)
    """
    # Pandas Dataframe
    df = pd.DataFrame(results_array)
    # Save in .csv file
    df.to_csv('results.csv', index=False, header=None)


def detect_object(cfg_file, weight_file, test_path):
    """ Runs the YOLO model to detect wally in the image
    Args:
        cfg_file (str): Darknet trained model config file
        weight_file (str): Darknet trained model weights
        test_path (str): Path to the folder containing the test images
    Returns:
        test_results (list of list): Returns the following parameters, (img_path, center_x, center_y)
    """
    score_thresh = 0.5
    nms_thresh = 0.5

    # darknet trained model config file
    cfg_file = os.path.join(os.getcwd(), cfg_file)
    # darknet trained model weights
    weight_file = os.path.join(os.getcwd(), weight_file)

    # Loads the model on OpenCv
    model = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)

    # Get the list of test images
    test_images = os.listdir(test_path)
    # List of results
    test_results = []
    for img_path in test_images:
        # Adding the relative path to the images filename
        img_full_path = os.path.join(test_path, img_path)
        # Read images
        img = cv2.imread(img_full_path)
        # Get the images' size
        img_width, img_height, _ = img.shape

        # Setting input parameters
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        # Get the results out of the output layer
        last_layer = model.getUnconnectedOutLayersNames()
        output_layer = model.forward(last_layer)

        boxes = []
        confidences = []
        # Iterates over each output
        for out in output_layer:
            # Iterates over each detection of a class output
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Considering only detections with scores higher than 0.5
                if confidence > score_thresh:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    w = int(detection[2] * img_width)
                    h = int(detection[3] * img_height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Applying Non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                test_results.append([img_path, center_x, center_y])
                break
        # Returning zeros if no objects were detected
        if len(boxes) == 0:
            center_x = 0
            center_y = 0
            test_results.append([img_path, center_x, center_y])

    return test_results


def convert_json(folder_path):
    """ Converts all .json annotation files to darknet pattern .txt annotation files
    Args:
        folder_path: Training folder containing the annotation files
    """
    # Reading file in the folder
    files = os.listdir(folder_path)
    # Collecting only .json files
    json_files = []
    for file in files:
        if file.endswith('.json'):
            json_files.append(file)

    for json_filename in json_files:
        with open(os.path.join(folder_path, json_filename), 'r') as ann_file:
            annotation = json.load(ann_file)

        # Since this problem only has one class, we then define the only label as 0
        label_class = 0
        # Getting images as bbox infos
        img_height = annotation['imageHeight']
        img_width = annotation['imageWidth']
        bbox = np.array(annotation['shapes'][0]['points'])
        # Getting the coordinates
        x_min, y_min = bbox.min(0)
        x_max, y_max = bbox.max(0)
        # Normalizing the points to [0, 1]
        x_min, x_max = x_min / img_width, x_max / img_width
        y_min, y_max = y_min / img_height, y_max / img_height
        # converting to YOLO format: x, y, width, height
        width = round(x_max - x_min, 6)
        height = round(y_max - y_min, 6)
        x = round(x_min + width, 6)
        y = round(y_min + height, 6)

        # Saving in a .txt file
        # We are considering that each image only has a single object
        txt_filename = folder_path + '/' + json_filename[:-5] + '.txt'
        with open(txt_filename, 'w') as txt_file:
            txt_file.write(str(label_class) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height))


def prepare_data(train_folder, dst_folder):
    """ Creates a folder with training files

    Args:
        train_folder: Path to the folder containing all training files
        dst_folder (str): Folder path where all files should be saved
    """
    local_dst_folder = os.path.join(train_folder, dst_folder)
    if not os.path.exists(local_dst_folder):
        os.makedirs(local_dst_folder)

    # Reading file in the folder
    files = os.listdir(train_folder)
    # Collecting only .json files
    img_files = []
    for file in files:
        if file.endswith('.jpg'):
            img_files.append(file)

    for img_filename in img_files:
        txt_filename = img_filename.replace('.jpg', '.txt')

        img_curr, txt_curr = os.path.join(train_folder, img_filename), os.path.join(train_folder, txt_filename)
        img_new, txt_new = os.path.join(local_dst_folder, img_filename), os.path.join(local_dst_folder, txt_filename)
        # Copying files
        copyfile(img_curr, img_new)
        copyfile(txt_curr, txt_new)

    # Create a .txt file with each .jpg filename
    with open('train.txt', 'w') as txt_file:
        for i in files:
            if i.endswith('.jpg'):
                txt_file.write('dst_folder' + '/' + i)
                txt_file.write('\n')
