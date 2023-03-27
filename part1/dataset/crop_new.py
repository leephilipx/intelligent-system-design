import cv2
import glob, os
import numpy as np
from tqdm import tqdm

from facealign import FaceAlignerModified


class ResNet10SSDModified:

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

    DATASET_VERSION = 2
    
    image_list = sorted(glob.glob(f'v{DATASET_VERSION}_ee4208_faces/{name}/**/*.*', recursive=True))
    face_det = ResNet10SSDModified('../prototype/artifacts/detection/res10_ssd_deploy.prototxt.txt',
                                   '../prototype/artifacts/detection/res10_300x300_ssd_iter_140000.caffemodel')
    face_align = FaceAlignerModified('../prototype/artifacts/alignment/shape_predictor_68_face_landmarks.dat')

    for i, im in tqdm(enumerate(image_list), desc=f'cropping faces from {name}', total=len(image_list)):

        dir_name = os.path.join(f'v{DATASET_VERSION}_output', name)
        os.makedirs(dir_name, exist_ok=True)

        img = cv2.imread(im)
        h, w = img.shape[:2]

        bbox_list = face_det.detect_faces(img, w, h, score=0.5)

        # img = draw_bbox(img, bbox_list)
        # cv2.imshow('image preview', img)
        # cv2.waitKey(0)

        if len(bbox_list) == 0:
            print(f'>> no face detected @ {i}_x.png')
            continue

        face_list = face_align.align_crop_preprocess_faces(img, bbox_list)
        for j, face in enumerate(face_list):
             cv2.imwrite(os.path.join(dir_name, f'{i}_{j}.png'), face)


if __name__ == '__main__':

    main(name='honey')
    main(name='jane')
    main(name='philip')
    main(name='veronica')
    main(name='wai_yeong')
    main(name='unknown')