import cv2

import utils.draw
import utils.facedet
import utils.video


if __name__ == '__main__':

    # Initialise video capture
    special_kwargs = [[cv2.CAP_PROP_FRAME_WIDTH, 1280], [cv2.CAP_PROP_FRAME_HEIGHT, 720]]
    cap = utils.video.VideoCaptureAsync(src=0, special_kwargs=special_kwargs)
    cap.start()

    # Initialise models
    # face_detector = utils.facedet.HaarCascade('models/haarcascade_frontalface_default.xml', scale_factor=1.3, min_neighbors=3)
    face_detector = utils.facedet.ResNet10SSD('models/res10_ssd_deploy.prototxt.txt', 'models/res10_300x300_ssd_iter_140000.caffemodel', w=1280, h=720)
    
    # Initialise FPS counter
    fps = utils.video.FPS()

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Face detection
        bbox_list = face_detector.detect_faces(frame, score=0.5)
        frame = utils.draw.draw_bbox(frame, bbox_list)

        # Display FPS and frame
        frame = utils.draw.draw_text(frame, f'FPS: {fps.get():.0f}', loc='NW')
        cv2.imshow('Async Video Feed', frame)
        
        # Press ESC key to exit
        key = cv2.waitKey(1)
        if key == 27: break

        fps.log()

    cap.release()
    cv2.destroyAllWindows() 