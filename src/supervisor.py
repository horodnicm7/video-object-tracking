import cv2
import imutils

from src.mosse import Mosse


class Supervisor(object):
    window_name = 'mosse tracker'

    def __init__(self, video):
        self.file_path = video
        self.video = cv2.VideoCapture(video)
        self.running = True
        self.paused = False
        self.trackers = []
        self.current_frame = None

    @property
    def gray_frame(self):
        return cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

    def __on_select(self, selection):
        self.trackers.append(Mosse(self.gray_frame, selection))

    def run(self, width=500):
        while self.running:
            if not self.paused:
                frame = self.video.read()[1]
                self.current_frame = frame

                # end of video
                if frame is None:
                    break

                for tracker in self.trackers:
                    tracker.update(self.gray_frame)

                frame = imutils.resize(frame, width=width)
                cv2.imshow(Supervisor.window_name, frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('r'):
                self.trackers = []

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
