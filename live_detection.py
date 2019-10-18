import cv2
import time
import queue
import threading


class VideoCapture:
    """
    bufferless VideoCapture
    """

    def __init__(self, name, width=1920, height=1080):
        self.cap = cv2.VideoCapture(name)

        if not self.cap.isOpened():
            print("Cannot open camera")
            return

        width_set = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        height_set = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

        if not (width_set and height_set):
            print("Set resolution failed")

        print("Resolution: {}x{}".format(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                         self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.q = queue.Queue()
        self._stop = False

        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while not self._stop:
            ret, img = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(img)

    def read(self):
        return self.q.get()

    def close(self):
        self._stop = True
        time.sleep(.25)
        self.cap.release()


if __name__ == "__main__":
    cap = VideoCapture(0)

    while True:
        frame = cap.read()

        # apply detection here

        cv2.imshow("frame", frame)
        if chr(cv2.waitKey(5) & 0xFF) == 'q':
            break
    cap.close()
