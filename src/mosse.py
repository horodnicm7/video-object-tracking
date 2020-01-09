import cv2
import numpy as np


class Mosse(object):
    EPS = 1e-5

    def __init__(self, initial_frame, rectangle):
        self.__get_template_info(initial_frame, rectangle)
        self.__reduce_image_noise_and_details()

        self.template = cv2.getRectSubPix(initial_frame, self.size, self.template_center)
        self.__prepare_convolution_terms()

    def preprocess_frame(self, frame, ):
        # the pixel values are transformed using a log function which helps with low contrast
        # lighting situations.
        image = np.log(np.float32(frame) + 1.0)

        # the pixel values are normalized to have a mean value of 0.0 and a norm of 1.0
        image = (image - image.mean()) / (image.std() + Mosse.EPS)
        return image * self.template

    def __prepare_convolution_terms(self):
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros(shape=self.G.shape)
        self.H2 = np.zeros_like(shape=self.G.shape)

        for _ in range(128):
            a = self.preprocess(rnd_warp(self.template))

            # compute the DFT of the image
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)

            # get correlation between G and A, without flags
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            # get correlation between A and A, without flags
            self.H2 += cv2.mulSpectrums(A, A, 0, conjB=True)

    def __reduce_image_noise_and_details(self):
        # apply Gaussian Blur on the original image, to reduce noise and the amount of details, for
        # faster processing
        self.g = np.zeros((self.size[::-1]), np.float32)
        self.g[self.size[1] // 2, self.size[0] // 2] = 1
        self.g = cv2.GaussianBlur(self.g, (-1, -1), 2.0)
        self.g /= g.max()

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
