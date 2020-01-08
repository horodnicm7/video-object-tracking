import cv2
import imutils

from src.mosse import Mosse


class Supervisor(object):
    def __init__(self, video, window_name='mosse_tracker'):
        self.window_name = window_name
        self.file_path = video
        self.video = cv2.VideoCapture(video)
        self.running = True
        self.paused = False
        self.trackers = []
        self.current_frame = None

        self.start_points = None
        self.rectangle = None

    @property
    def gray_frame(self):
        return cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

    def __on_select(self, selection):
        self.trackers.append(Mosse(self.gray_frame, selection))

    def on_mouse_move(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_points = (x, y)
        elif self.start_points:
            # if it's a click release, then there is a selection
            if flags & cv2.EVENT_FLAG_LBUTTON:
                x0, y0 = [min(x, y) for x, y in zip(self.start_points, (x, y))]
                x1, y1 = [max(x, y) for x, y in zip(self.start_points, (x, y))]

                # if the selection is larger than a point and not a straight line
                if x1 > x0 and y1 > y0:
                    self.rectangle = (x0, y0, x1, y1)

                return

            if self.rectangle:
                self.__on_select(self.rectangle)
            self.rectangle = None
            self.start_points = None

    def run(self, width=500):
        self.current_frame = self.video.read()[1]
        cv2.imshow(self.window_name, self.current_frame)

        cv2.setMouseCallback(self.window_name, self.on_mouse_move)

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
                cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(10)

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('r'):
                self.trackers = []

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
