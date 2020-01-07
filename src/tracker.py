import cv2
import imutils
from imutils.video import FPS


class Tracker(object):
    def __init__(self, video):
        self.file_path = video
        self.video = cv2.VideoCapture(video)
        self.running = True
        self.pause = False

    def run(self, width=500):
        while self.running:
            if not self.pause:
                frame = self.video.read()
                frame = frame[1]

                # end of video
                if frame is None:
                    break

                frame = imutils.resize(frame, width=width)

                cv2.imshow("Frame", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.pause = not self.pause

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
