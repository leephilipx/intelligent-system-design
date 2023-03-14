import cv2, traceback

import utils.draw, utils.helper, utils.video
import utils.facedet, utils.facerec

W, H = 1280, 720


def main():

    # Initialise models
    face_dets = [
        utils.facedet.HaarCascade('models/haarcascade_frontalface_default.xml', scale_factor=1.3, min_neighbors=3),
        utils.facedet.ResNet10SSD('models/res10_ssd_deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel', w=W, h=H)
    ]
    face_dr = utils.facerec.PCA('models/sk_pca_nclass_5.joblib', n_components=60)
    face_rec = utils.facerec.MahalanobisClassifier('models/mu_fj_nclass_5.npz', VI=face_dr.get_VI())

    # Warmup models and pre-select face detector
    ret, frame = cap.read()
    for fd in face_dets: fd.detect_faces(frame)
    fd_selector = 1
    fr_multiplier = 1.5

    # Initialise FPS counter
    fps = utils.video.FPS()

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Face detection
        bbox_list = face_dets[fd_selector].detect_faces(frame, score=0.5)
        bbox_list = utils.helper.bbox_correction(bbox_list, max_w=W, max_h=H)

        # Crop faces, preprocess faces, and perform face recognition
        face_list = utils.helper.crop_faces(frame, bbox_list)
        features = utils.helper.preprocess_faces(face_list)
        features = face_dr.transform(features)
        labels = face_rec.predict(features, multiplier=fr_multiplier)

        # Draw results onto frame and display
        utils.draw.draw_bbox(frame, bbox_list, labels)
        utils.draw.draw_text(frame, f'FPS: {fps.get():.0f}', index=0)
        utils.draw.draw_text(frame, f'Face detector: {face_dets[fd_selector].__class__.__name__}', index=1)
        utils.draw.draw_text(frame, f'Mahalanobis multiplier: {fr_multiplier:.2f}', index=2)
        cv2.imshow('Async Video Feed', frame)
        
        # Keyboard input
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC key to exit
            break
        elif key == ord('d'):  # Toggle face detector
            fd_selector = (fd_selector + 1) % len(face_dets)
        elif key == ord('m'):  # Increase Mahalanobis multiplier
            fr_multiplier += 0.05
        elif key == ord('n'):  # Decrease Mahalanobis multiplier
            fr_multiplier -= 0.05

        fps.log()



if __name__ == '__main__':

    # Initialise video capture
    special_kwargs = [[cv2.CAP_PROP_FRAME_WIDTH, W], [cv2.CAP_PROP_FRAME_HEIGHT, H]]
    cap = utils.video.VideoCaptureAsync(src=0, special_kwargs=special_kwargs)
    cap.start()

    # Main loop
    try:
        main()
    except KeyboardInterrupt:
        print('>> Keyboard interrupt detected. Exiting ...')
    except:
        traceback.print_exc()

    # Clean up
    cap.release()
    cv2.destroyAllWindows() 