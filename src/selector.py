import cv2


class Selector(object):
    def __init__(self, window_name, add_tracker):
        self.start_points = None
        self.rectangle = None
        self.window_name = window_name
        self.add_tracker = add_tracker
        cv2.setMouseCallback(self.window_name, self.on_click)

    def on_click(self, event, x, y, flags, _):
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
                self.add_tracker(self.rectangle)
            self.rectangle = None
            self.start_points = None
