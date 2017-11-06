import cv2
import numpy
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
numpy.set_printoptions(threshold=numpy.nan)


def min(m, n):
    if (m >= n):
        return n
    else:
        return m


def wls_filter(image_orig, k_array):

    # print k_array
    k_array = cv2.blur(k_array,(5,5))
    # print k_array
    lambda_ = 0.2
    alpha = 1.2
    small_eps = 1e-4
    image = image_orig.astype(numpy.float) / 255.0
    s = image.shape
    k = numpy.prod(s)
    dx = numpy.diff(image, 1, 1)
    dy = numpy.diff(image, 1, 0)

    beta_array = numpy.copy(k_array)


    # generate the k array from b array
    # [rk, ck] = k_array.shape
    # sigma = min(rk, ck) / 25
    # print beta_array
    # for m in range(0, rk - 1):
    #     for n in range(0, ck - 1):
    #         k_q = k_array[m][n]
    #         if k_q != 0 and k_q != 1:
    #             continue
    #         p = image[m][n]
    #         print m,n
    #
    #         for i in range(0, rk - 1):
    #             for j in range(0, ck - 1):
    #
    #                 k_q = k_array[i][j]
    #                 if k_q != 0 and k_q != 1:
    #                     continue
    #                 q = image[i][j]
    #                 to_minimize = 1 - (k_q * (numpy.exp(- (((q - p) ** 2) / (2 * (sigma))))))
    #                 if (to_minimize < beta_array[m, n]):
    #                     beta_array[m][n] = to_minimize

    # lets see what happens
    [r, c] = beta_array.shape
    for i in range(0, r - 1):
        for j in range(0, c - 1):
            dx[i, j] = dx[i, j] * beta_array[i, j]
            dy[i, j] = dy[i, j] * beta_array[i, j]

    dx = -lambda_ / (numpy.absolute(dx) ** alpha + small_eps)
    dy = -lambda_ / (numpy.absolute(dy) ** alpha + small_eps)


    dy = numpy.vstack((dy, numpy.zeros((1, s[1]))))
    dy = dy.flatten(1)



    dx = numpy.hstack((dx, numpy.zeros((s[0], 1))))
    dx = dx.flatten(1)

    a = spdiags(numpy.vstack((dx, dy)), [-s[0], -1], k, k)

    d = 1 - (dx + numpy.roll(dx, s[0]) + dy + numpy.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k)
    _out = spsolve(a, image.flatten(1)).reshape(s[::-1])
    out = numpy.rollaxis(_out, 1)
    detail = image - out
    return out, detail  # float


if __name__ == '__main__':
    # image_path = "./source.jpg"
    # image = cv2.imread(image_path)
    # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # base, detail = wls_filter(grayImage, k_array)
    # cv2.imshow("Original", grayImage)
    # cv2.imshow("Base", base)
    # cv2.imshow("Detail", detail)
    cv2.waitKey(0)