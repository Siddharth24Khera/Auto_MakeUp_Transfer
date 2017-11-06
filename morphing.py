import FeatureDetection
import numpy as np
import cv2
import math


def calc_area(vertex1,vertex2,vertex3):
    x1 = vertex1[0]
    x2 = vertex2[0]
    x3 = vertex3[0]
    y1 = vertex1[1]
    y2 = vertex2[1]
    y3 = vertex3[1]
    return math.fabs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2.0)

def morph_image_delta1_delta2(ToBeMorphed,FeaturesTakenFrom,delta1,delta2,sourceTuple,makeUpTuple,isOnlySkin = False):

    output_image1, feature_points1, triangulation1,K1 = sourceTuple
    output_image2, feature_points2, triangulation2,K2 = makeUpTuple
    morphed_image = np.copy(ToBeMorphed)
    #cv2.imshow("abcd",np.hstack((output_image1,output_image2)))
    count = -1
    # if isOnlySkin:
    #     abc = np.zeros((ToBeMorphed.shape[0],ToBeMorphed.shape[1]))
    for triangle in triangulation1.simplices:
        count = count +1
        vertex1Im2 = feature_points2[triangle[0]]
        vertex2Im2 = feature_points2[triangle[1]]
        vertex3Im2 = feature_points2[triangle[2]]

        vertex1Im1 = feature_points1[triangle[0]]
        vertex2Im1 = feature_points1[triangle[1]]
        vertex3Im1 = feature_points1[triangle[2]]
        area_triangle_1 = calc_area(vertex1Im1,vertex2Im1,vertex3Im1)
        # print vertex1Im2,vertex2Im2,vertex3Im2
        # print vertex1Im1, vertex2Im1, vertex3Im1
        # print('\n')
        morphed_image[vertex1Im1[1],vertex1Im1[0]] = delta2* FeaturesTakenFrom[vertex1Im2[1],vertex1Im2[0]] + delta1*ToBeMorphed[vertex1Im1[1],vertex1Im1[0]]
        morphed_image[vertex2Im1[1],vertex2Im1[0]] = delta2* FeaturesTakenFrom[vertex2Im2[1],vertex2Im2[0]] + delta1*ToBeMorphed[vertex2Im1[1],vertex2Im1[0]]
        morphed_image[vertex3Im1[1],vertex3Im1[0]] = delta2* FeaturesTakenFrom[vertex3Im2[1],vertex3Im2[0]] + delta1*ToBeMorphed[vertex3Im1[1],vertex3Im1[0]]

        # For Triangle in second image
        minX = min(vertex1Im1[0],vertex2Im1[0],vertex3Im1[0])
        maxX = max(vertex1Im1[0], vertex2Im1[0], vertex3Im1[0])
        minY = min(vertex1Im1[1], vertex2Im1[1], vertex3Im1[1])
        maxY = max(vertex1Im1[1], vertex2Im1[1], vertex3Im1[1])
        # cv2.imshow("lsa",np.hstack((output_image1,output_image2)))
        for i in range(minX,maxX+1,1):
            for j in range(minY,maxY+1,1):
                vertex = np.array([i,j])
                if isOnlySkin and not (K1[j][i] == 1 or K1[j][i] == 2) :
                    continue
                # if isOnlySkin and not K1[j][i] == 1 :
                #     continue
                triangle_num = triangulation1.find_simplex((i,j))
                # print triangle_num
                if triangle_num == count:
                    barycentric1 = calc_area(vertex,vertex3Im1,vertex2Im1)/area_triangle_1
                    barycentric2 = calc_area(vertex,vertex3Im1,vertex1Im1)/area_triangle_1
                    barycentric3 = calc_area(vertex,vertex1Im1,vertex2Im1)/area_triangle_1
                    vertex_in_im2 = vertex1Im2*barycentric1 + vertex2Im2*barycentric2 + vertex3Im2*barycentric3
                    vertex_in_im2 = vertex_in_im2.astype(np.int)
                    resultant_intensity = delta2* FeaturesTakenFrom[vertex_in_im2[1],vertex_in_im2[0]] + delta1*ToBeMorphed[vertex[1],vertex[0]]
                    if isOnlySkin and K1[j][i] == 2:
                        resultant_intensity = 1.0 * FeaturesTakenFrom[vertex_in_im2[1], vertex_in_im2[0]] + 0.0 *ToBeMorphed[vertex[1],vertex[0]]
                    morphed_image[vertex[1],vertex[0]] = resultant_intensity
    # if isOnlySkin:
    #     cv2.imshow("qq",abc)
    return morphed_image

if __name__ == "__main__":
    image_height = 500

    imPath1 = "./inputIm/face.jpg"
    image1 = cv2.imread(imPath1)
    if image1 is None:
        print "Image Not Found"
        exit()
    r = image_height * 1.0 / image1.shape[0]
    dim = (int(image1.shape[1] * r), image_height)
    resizedImage1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)

    imPath2 = "./inputIm/face_man.jpg"
    image2 = cv2.imread(imPath2)
    if image2 is None:
        print "Image Not Found"
        exit()
    r = image_height * 1.0 / image2.shape[0]
    dim = (int(image2.shape[1] * r), image_height)
    resizedImage2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)

    sourceTuple = FeatureDetection.landmark_detection(resizedImage1)
    targetTuple = FeatureDetection.landmark_detection(resizedImage2)
    alphaBlend = 0.8
    morphed_image = morph_image_delta1_delta2(resizedImage1, resizedImage2,1-alphaBlend ,alphaBlend,sourceTuple,targetTuple)
    finalPrint = np.hstack((resizedImage1, morphed_image, resizedImage2))
    cv2.imshow("Detected", finalPrint)
    key = cv2.waitKey(0)