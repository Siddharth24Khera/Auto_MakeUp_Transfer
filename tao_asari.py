from os import path
import numpy as np
import cv2
from math import sqrt

__author__ = 'srlu'


class Aindane(object):
    """Implementation of AINDANE
    Adaptive and integrated neighborhood-dependent approach for nonlinear enhancement of color images
    Attributes:
        img_bgr: The image to be processed, read by cv2.imread
        img_gray: img_bgr converted to gray following NTSC
    """

    _EPS = 1e-6  # eliminate divide by zero error in I_conv/I

    def __init__(self, inp_img):
        """
        :param path_to_img : full path to the image file
        """
        self.img_bgr = np.copy(inp_img)
        self.img_gray = cv2.cvtColor(
            self.img_bgr,
            cv2.COLOR_BGR2GRAY  # cv2 bgr2gray doing the same as NTSC
        )  # I in the paper

        # to collect parameters
        self.z = None
        self.c = None
        self.p = None

    def _ale(self):
        """ale algorithm in SubSection 3.1 of the paper.
        Basically just the implementation of the following formula:
            In_prime = f(In, z)
        Calculates In and z, then return In_prime
        :return In_prime:
        """

        # Calculate In
        In = self.img_gray / 255.0  # 2d array, equation 2

        # Calculate z
        cdf = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256]).cumsum()
        L = np.searchsorted(cdf, 0.1 * self.img_gray.shape[0] * self.img_gray.shape[1], side='right')
        L_as_array = np.array([L])  # L as array, for np.piecewise
        z_as_array = np.piecewise(L_as_array,
                         [L_as_array <= 50,
                          50 < L_as_array <= 150,
                          L_as_array > 150
                          ],
                         [0, (L-50) / 100.0, 1]
                         )
        z = z_as_array[0]  # take the value out of array

        self.z = z

        # Result In_prime = f(In, z)
        In_prime = 0.5 * (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z))
        return In_prime

    def _ace(self, In_prime, c=5):
        """ace algorithm in SubSection 3.2 of the paper
        Implementation of:
            S = f(In_prime, E(P()))
        :param In_prime:
        :param c:
        :return S:
        """

        # image freq shift
        img_freq = np.fft.fft2(self.img_gray)
        img_freq_shift = np.fft.fftshift(img_freq)

        # gaussian freq shift
        sigma = sqrt(c**2 / 2)
        _gaussian_x = cv2.getGaussianKernel(
            int(round(sigma*3)),  # size of gaussian: 3*sigma(0.99...)
            int(round(sigma))  # cv2 require sigma to be int
        )
        gaussian = (_gaussian_x * _gaussian_x.T) / np.sum(_gaussian_x * _gaussian_x.T)  # normalize
        gaussian_freq_shift = np.fft.fftshift(
            np.fft.fft2(gaussian, self.img_gray.shape)  # gaussian kernel padded with 0, extend to image.shape
        )

        # "image freq shift" * "gaussian freq shift"
        image_fm = img_freq_shift * gaussian_freq_shift
        I_conv = np.real(np.fft.ifft2(np.fft.ifftshift(image_fm)))  # equation 6

        sigma_I = np.array([np.std(self.img_gray)])  # std of I,to an array, for np.piecewise
        P = np.piecewise(sigma_I,
                         [sigma_I <= 3,
                          3 < sigma_I < 10,
                          sigma_I >= 10
                          ],
                         [3, 1.0 * (27 - 2 * sigma_I) / 7, 1]
                         )[0]  # take the value out of array

        self.c = c
        self.p = P

        E = ((I_conv + self._EPS) / (self.img_gray + self._EPS)) ** P
        S = 255 * np.power(In_prime, E)
        return S

    def _color_restoration(self, S, lambdaa=[1, 1, 1]):
        S_restore = np.zeros(self.img_bgr.shape)
        for j in xrange(3):  # b,g,r
            S_restore[..., j] = S * (1.0 * self.img_bgr[..., j] / (self.img_gray + self._EPS)) * lambdaa[j]

        return np.clip(S_restore, 0, 255).astype('uint8')

    def aindane(self):
        """The algorithm put in a whole
        """

        In_prime = self._ale()
        S = self._ace(In_prime, c=240)
        return self._color_restoration(S, lambdaa=[1, 1, 1])


#################################################
def tao_asari_enhancement(image_path):
    aindane = Aindane(image_path)
    return aindane.aindane()

#################################################

if __name__ == '__main__':

    image_path = "./face.jpg"
    img = cv2.imread(image_path)
    cv2.namedWindow("Original")
    cv2.namedWindow("Enhanced Image")
    cv2.imshow("Original", img)
    enhanced = tao_asari_enhancement(img)
    print enhanced
    cv2.imshow("Enhanced Image",enhanced)
    cv2.waitKey(0)