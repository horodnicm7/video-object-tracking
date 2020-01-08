import cv2


class Mosse(object):
    def __init__(self, initial_frame, rectangle):
        self.__get_template_info(initial_frame, rectangle)

        template = cv2.getRectSubPix(initial_frame, (width, height), self.template_center)

    def __get_template_info(self, initial_frame, rectangle):
        left_x, left_y, right_x, right_y = rectangle

        # get the optimal fourier transform size of the template, because DFT performs better on images
        # of numbers that are multiple of 2,3 or 5. It might add some padding pixels too.
        width = cv2.getOptimalDFTSize(right_x - left_x)
        height = cv2.getOptimalDFTSize(right_y - left_y)

        optimal_left_x = int((left_x + right_x - width) / 2)
        optimal_left_y = int((left_y + right_y - height) / 2)

        self.template_center = (optimal_left_x + 0.5 * width, optimal_left_y + 0.5 * height)
        self.size = (width, height)

    def display_selection(self, frame):
        pass

    def update(self, frame):
        pass
