import cv2
import numpy as np
# import mtcnn

from .helper import xywh_to_xyxy

'''For all classes in this script, the detect_faces method should return a list of tuples (x1, y1, x2, y2)'''


class HaarCascade:

    '''This class is for the Haar Cascade model from OpenCV.'''

    def __init__(self, path, scale_factor=1.3, min_neighbors=3):
        
        self.face_cascade = cv2.CascadeClassifier(path)
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def detect_faces(self, frame, score=0.5):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, self._scale_factor, self._min_neighbors)

        return xywh_to_xyxy(faces)
    

class ResNet10SSD:

    '''This class is for the ResNet10 SSD model from OpenCV DNN module.'''

    def __init__(self, config_path, model_path, w, h):

        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self._resize_wh = np.array([w, h, w, h])

    def detect_faces(self, frame, score=0.5):

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,117.0,123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        bbox_list = []

        # faces is 4-d array (1, 1, 200, 7)
        # last dim: 1: class_id, 2: score, 3-6: boxes
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > score:
                bbox = (faces[0, 0, i, 3:7] * self._resize_wh).astype('int')
                bbox_list.append(bbox)

        return bbox_list
    

# class MTCNN:

#     '''This class is for the MTCNN model from the mtcnn package.'''

#     def __init__(self):

#         self.detector = mtcnn.MTCNN()
    
#     def detect_faces(self, frame, score=0.5):

#         faces = self.detector.detect_faces(frame)

#         return [res['box'] for res in faces if res['confidence'] > score]