import threading
import cv2
import time




class FPS:

    '''Class to calculate FPS of a video feed. Uses exponential weighting to smooth FPS values. See sample usage below.'''
	
    def __init__(self, weight=0.9, epsilon=1e-5):
        self._weight = weight
        self._epsilon = epsilon
        self._fps = 0.0
        self.log()

    def log(self):
        self._start = time.time()

    def get(self):
        new_fps =  1.0 / (time.time() - self._start + self._epsilon)
        self._fps = self._fps * self._weight + new_fps * (1-self._weight)
        return self._fps


class VideoCaptureAsync:

    '''Class to asynchronously read frames from a video feed. See sample usage below.'''
    '''Acknowledgement: This code is adapted from https://github.com/gilbertfrancois/video-capture-async'''

    def __init__(self, src=0, special_kwargs=[]):
        print(f'>> Initialising video capture: {src}')
        self.cap = cv2.VideoCapture(src)
        for kwarg in special_kwargs:
            self.cap.set(*kwarg)
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed is False:
            raise ValueError('>> Initialisation failed to grab frame from video source.')
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print('>> Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        print('>> Releasing video capture ...')
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


if __name__ == '__main__':

    # Initialise video capture
    special_kwargs = [[cv2.CAP_PROP_FRAME_WIDTH, 1280], [cv2.CAP_PROP_FRAME_HEIGHT, 720]]
    cap = VideoCaptureAsync(src=0, special_kwargs=special_kwargs)
    cap.start()

    # Initialise FPS counter
    fps = FPS()

    while True:
        
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret: break

        # Simulate some processing
        time.sleep(0.033)

        # Display FPS and frame
        frame = cv2.putText(frame, f'FPS: {fps.get():.0f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow('Async Video Feed', frame)
        
        # Press ESC key to exit
        key = cv2.waitKey(1)
        if key == 27: break

        fps.log()

    cap.release()
    cv2.destroyAllWindows()
    