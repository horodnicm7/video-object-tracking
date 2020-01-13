import cv2
import numpy as np


class Mosse(object):
    EPS = 1e-5

    def __init__(self, initial_frame, rectangle):
        self.__get_template_info(initial_frame, rectangle)
        self.__reduce_image_noise_and_details()

        self.template = cv2.getRectSubPix(initial_frame, self.size, self.template_center)
        self.__prepare_convolution_terms()
        self.update_kernel()
        self.update(initial_frame)

    @staticmethod
    def div_spec(A, B):
        Ar, Ai = A[..., 0], A[..., 1]
        Br, Bi = B[..., 0], B[..., 1]
        C = (Ar + 1j * Ai) / (Br + 1j * Bi)  # equation 10
        # make a list of lists. each list contains [real_part, image_part] of C
        C = np.dstack([np.real(C), np.imag(C)]).copy()
        return C

    def update_kernel(self):
        self.H = Mosse.div_spec(self.H1, self.H2)  # equation 10
        self.H[..., 1] *= -1

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # inverse Discrete Fourier Transform
        h, w = resp.shape
        _, peak_val, _, (mx, my) = cv2.minMaxLoc(resp)  # get peak value
        side_resp = resp.copy()

        # (image_to_draw_on, start_point, end_point, color, THICCness (-1 = fill the rectangle with color))
        cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()  # mean value/standard deviation
        psr = (peak_val - smean) / (sstd + Mosse.EPS)  # peak to sidelobe ratio
        return (mx - w // 2, my - h // 2), psr

    def update(self, frame, learning_rate=0.125):
        # get the current's frame template and preprocess it
        image = cv2.getRectSubPix(frame, self.size, self.template_center)
        image = self.preprocess_frame(image)

        (dx, dy), self.psr = self.correlate(image)

        # PSR under 8 means that the object is occluded or tracking has failed
        if self.psr < 8.0:
            return

        # update the templates center according to the new discovered position
        self.template_center = (self.template_center[0] + dx, self.template_center[1] + dy)
        image = cv2.getRectSubPix(frame, self.size, self.template_center)
        image = self.preprocess_frame(image)

        image_dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)  # fourier transformation
        # convolution (mulSpectrums: together with dft and idft, it may be used to calculate convolution
        H1 = cv2.mulSpectrums(self.G, image_dft, 0, conjB=True)
        H2 = cv2.mulSpectrums(image_dft, image_dft, 0, conjB=True)  # convolution

        self.H1 = H1 * learning_rate + self.H1 * (1.0 - learning_rate)  # equation 11
        self.H2 = H2 * learning_rate + self.H2 * (1.0 - learning_rate)  # equation 12

        self.update_kernel()

    def preprocess_frame(self, frame):
        # the pixel values are transformed using a log function which helps with low contrast
        # lighting situations.
        image = np.log(np.float32(frame) + 1.0)

        # the pixel values are normalized to have a mean value of 0.0 and a norm of 1.0
        image = (image - image.mean()) / (image.std() + Mosse.EPS)
        return image * self.hann_window

    def __prepare_convolution_terms(self):
        self.G = cv2.dft(self.g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

        a = self.preprocess_frame(self.template)

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
        self.g /= self.g.max()

        self.hann_window = cv2.createHanningWindow(self.size, cv2.CV_32F)

    def __get_template_info(self, initial_frame, rectangle):
        left_x, left_y, right_x, right_y = rectangle

        # get the optimal fourier transform size of the template, because DFT performs better on images
        # of numbers that are multiple of 2,3 or 5. It might add some padding pixels too.
        width = cv2.getOptimalDFTSize(right_x - left_x)
        height = cv2.getOptimalDFTSize(right_y - left_y)

        optimal_left_x = int((left_x + right_x - width) / 2)
        optimal_left_y = int((left_y + right_y - height) / 2)

        self.template_center = (optimal_left_x + 0.5 * (width-1), optimal_left_y + 0.5 * (height-1))
        self.size = (width, height)

    def display_selection(self, frame):
        left_x = int(self.template_center[0] - 0.5 * self.size[0])
        left_y = int(self.template_center[1] - 0.5 * self.size[1])
        right_x = int(self.template_center[0] + 0.5 * self.size[0])
        right_y = int(self.template_center[1] + 0.5 * self.size[1])

        cv2.rectangle(frame, (left_x, left_y), (right_x, right_y), (0, 0, 255))

        text = 'PSR: {}'.format(self.psr)
        cv2.putText(frame, text, (left_x + 1, right_y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, (left_x, right_y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if self.psr < 8.0:
            cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 0, 255))
            cv2.line(frame, (right_x, left_y), (left_x, right_y), (0, 0, 255))
