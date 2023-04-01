import cv2, traceback
import numpy as np

import utils.draw, utils.helper, utils.video
import utils.facedet, utils.facealign, utils.facerec

W, H = 1280, 720



def main(MODE='lda'):
    
    cap = utils.video.TestImagesStream('test_images', W, H)
    print('>> Initializing and warming up models ...')

    # Initialise models
    # face_det = utils.facedet.HaarCascade(scale_factor=1.3, min_neighbors=3), float('inf')
    face_det, fd_threshold = utils.facedet.ResNet10SSD(w=W, h=H), 0.6
    face_align = utils.facealign.FaceAligner()
    if MODE == 'pca':
        face_dr = utils.facerec.PCA('artifacts/classification/sk_pca_nclass_5.joblib')
    elif MODE == 'lda':
        face_dr = utils.facerec.FisherFace('artifacts/classification/sk_pca_nclass_5.joblib',
                                           'artifacts/classification/sk_lda_nclass_5.joblib')
    face_rec, fr_threshold = utils.facerec.MahalanobisDistance(f'artifacts/classification/wfj_{MODE}_nclass_5.npz'), 0.9999
    # face_rec, fr_threshold = utils.facerec.TfLiteMLP(f'artifacts/classification/keras_{MODE}_nclass_5.tflite'), 0.5
    # face_rec, fr_threshold = utils.facerec.LogisticRegression(f'artifacts/classification/logreg_{MODE}_nclass_5.joblib'), 0.5

    # Warmup models
    ret, frame = cap.read()
    face_det.detect_faces(frame)
    face_rec.predict(face_dr.transform(np.zeros((1, 90*120), dtype=np.float32)))
    print('>> Models warmed up successfully')

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Face detection, then face alignment and cropping
        bbox_list = face_det.detect_faces(frame, score=fd_threshold)
        bbox_list = utils.helper.bbox_correction(bbox_list, max_w=W, max_h=H)
        faces, keypoints = face_align.align_faces(frame, bbox_list)

        # Feature extraction, then face recognition
        features_x = utils.helper.preprocess(faces, resize=None)
        features_f = face_dr.transform(features_x)
        labels = face_rec.predict(features_f, conf=fr_threshold)

        # Draw results onto frame and display
        utils.draw.draw_bbox(frame, bbox_list, labels)
        utils.draw.draw_text(frame, f'FPS: 15', index=0)
        utils.draw.draw_text(frame, f'[R] Face recognizer: {MODE.upper()}, {face_rec.__class__.__name__} ' + \
                             f'(r={fr_threshold})', index=1)
        utils.draw.draw_text(frame, f'[D] Face detector: {face_det.__class__.__name__} ' + \
                             f'(d={fd_threshold})', index=2)
        cv2.imshow('Online Face Detection and Recognition: Toggle Classifier [R], Detector [D],  Keypoints [C]', frame)
        
        # Keyboard input
        key = cv2.waitKey(0)
        if key == 27:  # Press ESC key to exit
            break



if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pca','lda'], default='lda', help='Dimensionality reduction method')
    args = parser.parse_args()

    # Main loop
    try:
        main(MODE=args.mode)
    except KeyboardInterrupt:
        print('>> Keyboard interrupt detected. Exiting ...')
    except:
        traceback.print_exc()

    # Clean up
    cv2.destroyAllWindows() 