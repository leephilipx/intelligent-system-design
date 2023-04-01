import cv2, traceback
import numpy as np

import utils.draw, utils.helper, utils.video
import utils.facedet, utils.facealign, utils.facerec

W, H = 1280, 720



def main(cap, MODE='lda'):
    
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
    ]

    # Initialise FPS counter
    fps = utils.video.FPS()

    # Warmup models
    ret, frame = cap.read()
    for fd in face_dets: fd.detect_faces(frame)
    for fr in face_recs: fr.predict(face_dr.transform(np.zeros((1, 90*120), dtype=np.float32)))
    print('>> Models warmed up successfully')

    # Pre-select models and thresholds
    bool_keypoints = False
    fd_selector, fr_selector = 1, 0
    MAHALANOBIS_CONF = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]
    FD_ORIGINAL, FR_ORIGINAL = [float('inf'), 0.6], [6, 0.5, 0.5]
    fd_threshold, fr_threshold = [x for x in FD_ORIGINAL], [x for x in FR_ORIGINAL]

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Face detection, then face alignment and cropping
        bbox_list = face_dets[fd_selector].detect_faces(frame, score=fd_threshold[fd_selector])
        bbox_list = utils.helper.bbox_correction(bbox_list, max_w=W, max_h=H)
        faces, keypoints = face_align.align_faces(frame, bbox_list)

        # Preprocessing, feature extraction, face recognition
        features_x = utils.helper.preprocess(faces, resize=None)
        features_f = face_dr.transform(features_x)
        fr_conf = MAHALANOBIS_CONF[fr_threshold[0]] if fr_selector == 0 else round(fr_threshold[fr_selector], 2)
        labels = face_recs[fr_selector].predict(features_f, conf=fr_conf)

        # Draw results onto frame and display
        utils.draw.draw_bbox(frame, bbox_list, labels)
        if bool_keypoints: utils.draw.draw_keypoints(frame, keypoints)
        utils.draw.draw_text(frame, f'FPS: {fps.get():.0f}', index=0)
        utils.draw.draw_text(frame, f'[R] Face recognizer: {MODE.upper()}, ' + \
                              f'{face_recs[fr_selector].__class__.__name__} ' + \
                             f'(r={fr_conf})', index=1)
        utils.draw.draw_text(frame, f'[D] Face detector: {face_dets[fd_selector].__class__.__name__} ' + \
                             f'(d={fd_threshold[fd_selector]:.2f})', index=2)
        cv2.imshow('Online Face Detection and Recognition: Toggle Classifier [R], Detector [D],  Keypoints [C]', frame)
        
        # Keyboard input
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC key to exit
            break
        elif key == ord('c'):  # Toggle keypoints
            bool_keypoints = not bool_keypoints
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
            if fr_selector == 0:
                fr_threshold[0] = min(len(MAHALANOBIS_CONF)-1, fr_threshold[0]+1)
            else:
                fr_threshold[fr_selector] += 0.05
        elif key == ord('o'):  # Decrease face recognition threshold
            if fr_selector == 0:
                fr_threshold[0] = max(0, fr_threshold[0]-1)
            else:
                fr_threshold[fr_selector] -= 0.05

        fps.log()



if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='lda', help='Dimensionality reduction method')
    args = parser.parse_args()

    # Initialise video capture
    special_kwargs = [[cv2.CAP_PROP_FRAME_WIDTH, W], [cv2.CAP_PROP_FRAME_HEIGHT, H]]
    cap = utils.video.VideoCaptureAsync(src=0, special_kwargs=special_kwargs)
    cap.start()

    # Main loop
    try:
        main(cap=cap, MODE=args.mode)
    except KeyboardInterrupt:
        print('>> Keyboard interrupt detected. Exiting ...')
    except:
        traceback.print_exc()

    # Clean up
    cap.release()
    cv2.destroyAllWindows() 