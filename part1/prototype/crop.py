import cv2
import glob, os
import numpy as np


class ResNet10SSD:

    '''This class is for the ResNet10 SSD model from OpenCV DNN module.'''

    def __init__(self, config_path, model_path):

        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    def detect_faces(self, frame, w, h, score=0.5):

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,117.0,123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        bbox_list = []

        # faces is 4-d array (1, 1, 200, 7)
        # last dim: 1: class_id, 2: score, 3-6: boxes
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > score:
                bbox = (faces[0, 0, i, 3:7] * np.array([w, h, w, h])).astype('int')
                bbox_list.append(bbox)

        return bbox_list


def draw_bbox(frame, bbox_list):

    '''Draw bounding boxes on the frame, given the list of tuples (x, y, w, h)'''

    for (x1, y1, x2, y2) in bbox_list:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    return frame


def main(name):
    
    image_list = glob.glob(f'../face-rec/ee4208 faces/{name}/**')
    face_det = ResNet10SSD('models/res10_ssd_deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel') 

    for i, im in enumerate(image_list):

        dir_name = os.path.dirname(im).replace('ee4208 faces', 'output')
        os.makedirs(dir_name, exist_ok=True)

        img = cv2.imread(im)
        h, w = img.shape[:2]

        bbox_list = face_det.detect_faces(img, w, h, score=0.8)

        img = draw_bbox(img, bbox_list)
        cv2.imshow('image preview', img)
        cv2.waitKey(0)

        bbox = bbox_list[0]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cropped = img[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (80,100))
        cv2.imwrite(os.path.join(dir_name, f'{i}.png'),cropped)


main('honey')
main('jane')
# main('philip')
main('veronica')
main('wai_yeong')
