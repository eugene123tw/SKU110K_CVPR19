from __future__ import print_function

import csv
import glob
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.keras_retinanet.utils.visualization import \
    draw_detections
from object_detector_retinanet.utils import create_folder


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).

    Returns
        The input with the ImageNet mean subtracted.
    """
    x = x.astype(np.float32)
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and 
    max_side.

    Args
        min_side: The image's min side will be equal to min_side after 
          resizing.
        max_side: If after resizing the image's max side is above max_side, 
          resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def find_csv_paths(csv_path):
    csv_paths = glob.glob(csv_path + '/test*.csv')
    return csv_paths


def read_csv_result(csv_path):
    df = pd.read_csv(csv_path)
    df = df.to_numpy()
    return df[:, :4], df[:, 4], df[:, 5], df[:, 6]


def predict(result_csv_folder, img_folder, save_path=None):

    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2',
                        'y2', 'confidence', 'hard_score'])

    csv_paths = find_csv_paths(result_csv_folder)
    image_names = [os.path.basename(path).replace(
        ".csv", "") for path in csv_paths]

    num_samples = len(image_names)
    num_classes = 1

    all_detections = [[None for i in range(
        num_classes)] for j in range(num_samples)]

    for i in range(len(image_names)):
        image_name = image_names[i]
        csv_path = csv_paths[i]
        raw_image = read_image_bgr(os.path.join(img_folder, image_name))
        image = preprocess_image(raw_image.copy())
        image, scale = resize_image(image)

        # run network
        image_boxes, image_scores, image_hard_scores, image_labels = read_csv_result(
            csv_path)

        image_detections = np.concatenate([
            image_boxes,
            np.expand_dims(image_scores, axis=1),
            np.expand_dims(image_labels, axis=1)], axis=1
        )
        results = np.concatenate([
            image_boxes,
            np.expand_dims(image_scores, axis=1),
            np.expand_dims(image_hard_scores, axis=1),
            np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(image_name, results)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for _, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'],
                             detection['x2'], detection['y2']])
            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                   detection['confidence'], detection['hard_score']]
            csv_data_lst.append(row)

        if save_path is not None:
            create_folder(save_path)
            draw_detections(raw_image,
                            np.asarray(filtered_boxes),
                            np.asarray(filtered_scores),
                            np.asarray(filtered_labels), color=(0, 0, 255))

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(num_classes):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]


if __name__ == '__main__':
    dataset_folder = "/home/yuchunli/_DATASET/SKU110K_fixed"
    predict(
        result_csv_folder=os.path.join(dataset_folder, 'results'),
        img_folder=os.path.join(dataset_folder, 'images')
    )
