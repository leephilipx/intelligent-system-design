import cv2, dlib
import numpy as np
from collections import OrderedDict

'''Acknowledgement: This code is adapted from pyimagesearch.com imutils library'''


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


class FaceAlignerImutils:
    
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        
        #simple hack ;)
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            raise Exception("Only 68 landmarks supported")
            # (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            # (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        
        ## - additional line to fix the bug - ##
        eyesCenter = [int(x) for x in eyesCenter]

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        output = cv2.warpAffine(image, M, (self.desiredFaceWidth, self.desiredFaceHeight), flags=cv2.INTER_CUBIC)

        # return the aligned face and keypoints
        return output, shape



### OUR CODE BELOW ###

class FaceAligner:

    def __init__(self, path='artifacts/alignment/shape_predictor_68_face_landmarks.dat', size=(90, 120)):

        self.predictor = dlib.shape_predictor(path)
        self.fa = FaceAlignerImutils(self.predictor, desiredLeftEye=(0.23, 0.28),
                                     desiredFaceWidth=size[0], desiredFaceHeight=size[1])

    def align_faces(self, frame, bbox_list):

        if len(bbox_list) == 0:
            return np.array([]), np.array([])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list, kp_list = [], []

        for (x1, y1, x2, y2) in bbox_list:
            face, kpoints = self.fa.align(frame, gray, dlib.rectangle(x1, y1, x2, y2))
            face_list.append(face)
            kp_list.append(kpoints)
        
        return np.array(face_list), np.array(kp_list)