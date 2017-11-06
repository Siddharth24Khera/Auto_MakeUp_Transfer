import math
import numpy as np
import cv2

def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def dist_pixels(pi, pj, qi, qj):
    return math.sqrt(((pi - qi) ** 2) + ((pj - qj) ** 2))


def histogram_equilize_grayscale(inp_img):
    equilised_img = np.zeros((inp_img.shape[0], inp_img.shape[1]), dtype=np.uint8)
    num_pixels = inp_img.shape[0]*inp_img.shape[1]
    flattend_img = inp_img.flatten()
    hist, bins = np.histogram(flattend_img,256,[0,255])
    cdf = hist.cumsum()/float(num_pixels)
    for i in range(inp_img.shape[0]):
        for j in range(inp_img.shape[1]):
            equilised_img[i][j] = cdf[inp_img[i][j]] * 255
    return equilised_img

def histogram_equilize_color(inp_img):
    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    equilised_img = np.zeros((inp_img.shape[0], inp_img.shape[1],inp_img.shape[2]), dtype=np.uint8)
    num_pixels = inp_img.shape[0] * inp_img.shape[1]
    flattend_img = inp_img[:,:,2].flatten()
    hist, bins = np.histogram(flattend_img, 256, [0, 255])
    cdf = hist.cumsum() / float(num_pixels)
    for i in range(inp_img.shape[0]):
        for j in range(inp_img.shape[1]):
            equilised_img[i][j][0] = inp_img[i][j][0]
            equilised_img[i][j][1] = inp_img[i][j][1]
            equilised_img[i][j][2] = cdf[inp_img[i][j][2]] * 255
    return cv2.cvtColor(equilised_img,cv2.COLOR_HSV2BGR)

def lip_makeup(p, q, p_k, q_k): #p is the orignal image, and q is the reference image
    #p_k is the k_array of the image p
    #q_k is the k_array of the image q

    p = histogram_equilize_color(p)
    q = histogram_equilize_color(q)
    p = cv2.cvtColor(p,cv2.COLOR_BGR2LAB)
    q = cv2.cvtColor(q,cv2.COLOR_BGR2LAB)
    p_l = p[:,:,0]
    p_a = p[:,:,1]
    p_b = p[:,:,2]
    q_l = q[:,:,0]
    q_a = q[:,:,1]
    q_b = q[:,:,2]

    [rp, cp,_] = p.shape
    [rq, cq,_] = q.shape
    for i in range(0, rp):
        for j in range(0, cp):
            I_p = p_l[i][j]
            if p_k[i][j] != 2:	# 2 denotes a lip pixel
                continue
            max_gaussian = -1
            max_i = i
            max_j = j
            for m in range(0, rq):
                for n in range(0, cq):
                    if q_k[m][n] != 2:
                        continue
                    E_q = q_l[m][n]
                    print(E_q,I_p)
                    intensity_diff = math.fabs(E_q - I_p)
                    distance = dist_pixels(i, j,  m, n)
                    gaussian_val = gaussian(distance , 5) * gaussian(intensity_diff, 5)
                    if(gaussian_val > max_gaussian):
                        max_gaussian = gaussian_val
                        max_i = m
                        max_j = n
            resultant_image = np.zeros(p.shape)
            resultant_image[i,j,0] = q_l[max_i][max_j]
            resultant_image[i,j,1] = q_a[max_i][max_j]
            resultant_image[i,j,2] = q_b[max_i][max_j]

    print(np.max(resultant_image),np.min(resultant_image))
    return resultant_image