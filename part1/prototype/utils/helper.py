import cv2
import numpy as np


def xywh_to_xyxy(bbox_list):

    '''Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2)'''

    return [(x, y, x+w, y+h) for x, y, w, h in bbox_list]


def bbox_correction(bbox_list, max_w=1280, max_h=720):

    '''Convert bounding boxes that are out of the image range to the image range'''

    for i, (x1, y1, x2, y2) in enumerate(bbox_list):
        bbox_list[i] = (max(0, x1), max(0, y1), min(max_w, x2), min(max_h, y2))

    return bbox_list


def crop_faces(frame, bbox_list):

    '''Crop faces from the given frame using the given bounding boxes'''

    return [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in bbox_list]


def resize_preprocess_faces(face_list, size=(75, 100)):

    '''Resize and preprocess faces for face recognition'''

    if len(face_list) == 0:
        return np.array([])

    feature_list = []

    for face in face_list:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, size)
        feature_list.append(face.flatten())

    return np.array(feature_list)