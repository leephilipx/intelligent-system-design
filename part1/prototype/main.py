import cv2, traceback
import numpy as np

import utils.draw, utils.helper, utils.video
import utils.facedet, utils.facealign, utils.facerec

W, H = 1280, 720
MODE = 'pca'



def main(cap):
    
    print('>> Initializing and warming up models ...')

    # Initialise models
    face_dets = [
        utils.facedet.HaarCascade(scale_factor=1.3, min_neighbors=3),
        utils.facedet.ResNet10SSD(w=W, h=H)
    ]
    face_align = utils.facealign.FaceAligner()
    if MODE == 'pca':
        face_dr = utils.facerec.PCA('artifacts/classification/sk_pca_nclass_5.joblib')
    elif MODE == 'lda':
        face_dr = utils.facerec.FisherFace('artifacts/classification/sk_pca_nclass_5.joblib',
                                           'artifacts/classification/sk_lda_nclass_5.joblib')
    face_recs = [
        utils.facerec.MahalanobisDistance(f'artifacts/classification/wfj_{MODE}_nclass_5.npz'),
        utils.facerec.TfLiteMLP(f'artifacts/classification/keras_{MODE}_nclass_5.tflite'),
        utils.facerec.LogisticRegression(f'artifacts/classification/logreg_{MODE}_nclass_5.joblib'),
        utils.facerec.SupportVectorMachine(f'artifacts/classification/svm_{MODE}_nclass_5.joblib'),
    ]

    # Initialise FPS counter
    fps = utils.video.FPS()

    # Warmup models
    ret, frame = cap.read()
    for fd in face_dets: fd.detect_faces(frame)
    for fr in face_recs: fr.predict(face_dr.transform(np.zeros((1, 90*120), dtype=np.float32)))
    print('>> Models warmed up successfully')

    # Pre-select models and thresholds
    fd_selector, fr_selector = 1, 0
    FD_ORIGINAL, FR_ORIGINAL = [float('inf'), 0.5], [2.0, 0.7, 0.7, float('inf')]
    fd_threshold, fr_threshold = [x for x in FD_ORIGINAL], [x for x in FR_ORIGINAL]

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Face detection, then face alignment, cropping and preprocessing
        bbox_list = face_dets[fd_selector].detect_faces(frame, score=fd_threshold[fd_selector])
        bbox_list = utils.helper.bbox_correction(bbox_list, max_w=W, max_h=H)
        features = face_align.align_crop_preprocess_faces(frame, bbox_list)

        # Feature extraction, then face recognition
        features = face_dr.transform(features)
        labels = face_recs[fr_selector].predict(features, fr_threshold[fr_selector])

        # Draw results onto frame and display
        utils.draw.draw_bbox(frame, bbox_list, labels)
        utils.draw.draw_text(frame, f'FPS: {fps.get():.0f}', index=0)
        utils.draw.draw_text(frame, f'[R] Face recognizer: {face_recs[fr_selector].__class__.__name__} ' + \
                             f'(r={fr_threshold[fr_selector]:.2f})', index=1)
        utils.draw.draw_text(frame, f'[D] Face detector: {face_dets[fd_selector].__class__.__name__} ' + \
                             f'(d={fd_threshold[fd_selector]:.2f})', index=2)
        cv2.imshow('Online Face Detection and Recognition', frame)
        
        # Keyboard input
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC key to exit
            break
        elif key == ord('d'):  # Toggle face detector
            fd_selector = (fd_selector + 1) % len(face_dets)
            fd_threshold[fd_selector] = FD_ORIGINAL[fd_selector]
        elif key == ord('r'):  # Toggle face recognizer
            fr_selector = (fr_selector + 1) % len(face_recs)
            fr_threshold[fr_selector] = FR_ORIGINAL[fr_selector]
        elif key == ord('l'):  # Increase face detection threshold
            fd_threshold[fd_selector] += 0.05
        elif key == ord('k'):  # Decrease face detection threshold
            fd_threshold[fd_selector] -= 0.05
        elif key == ord('p'):  # Increase face recognition threshold
            fr_threshold[fr_selector] += 0.05
        elif key == ord('o'):  # Decrease face recognition threshold
            fr_threshold[fr_selector] -= 0.05

        fps.log()



if __name__ == '__main__':

    # Initialise video capture
    special_kwargs = [[cv2.CAP_PROP_FRAME_WIDTH, W], [cv2.CAP_PROP_FRAME_HEIGHT, H]]
    cap = utils.video.VideoCaptureAsync(src=0, special_kwargs=special_kwargs)
    cap.start()

    # Main loop
    try:
        main(cap=cap)
    except KeyboardInterrupt:
        print('>> Keyboard interrupt detected. Exiting ...')
    except:
        traceback.print_exc()

    # Clean up
    cap.release()
    cv2.destroyAllWindows() 