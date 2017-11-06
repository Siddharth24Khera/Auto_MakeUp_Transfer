import FeatureDetection
import morphing
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import WLS_filter
import lipTranfer

image_height = 500


sourceImagePath = "./MakeUp/source.jpg"
sourceImage = cv2.imread(sourceImagePath)
r = image_height * 1.0 / sourceImage.shape[0]
dim = (int(sourceImage.shape[1] * r), image_height)
sourceImage = cv2.resize(sourceImage, dim, interpolation=cv2.INTER_AREA)
output_image1, feature_points1, triangulation1,K1 = FeatureDetection.landmark_detection(sourceImage)
sourceTuple = (output_image1, feature_points1, triangulation1,K1)

Kdash = (K1 - np.min(K1))/(np.max(K1)-np.min(K1))
cv2.imshow("ahfkd",Kdash)


makeUpImagePath = "./MakeUp/makeU.jpg"
makeUpImage = cv2.imread(makeUpImagePath)
r = image_height * 1.0 / makeUpImage.shape[0]
dim = (int(makeUpImage.shape[1] * r), image_height)
makeUpImage = cv2.resize(makeUpImage, dim, interpolation=cv2.INTER_AREA)
output_image2, feature_points2, triangulation2,K2 = FeatureDetection.landmark_detection(makeUpImage)
makeUpTuple = (output_image2, feature_points2, triangulation2,K2)

sourceLAB = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2LAB)
resultant_image = np.copy(sourceLAB)

sourceLightness = sourceLAB[:,:,0]
sourceColor = sourceLAB[:,:,1:3]
sourceStructureLayer,sourceTextureLayer = WLS_filter.wls_filter(sourceLightness,K1)
# cv2.imshow("h1fjs",np.hstack((sourceStructureLayer,sourceTextureLayer)))

makeUpLAB = cv2.cvtColor(makeUpImage,cv2.COLOR_BGR2LAB)
makeUpLightness = makeUpLAB[:,:,0]
makeUpColor = makeUpLAB[:,:,1:3]
makeUpStructureLayer,makeUpTextureLayer = WLS_filter.wls_filter(makeUpLightness,K2)
# cv2.imshow("h2fjs",np.hstack((makeUpStructureLayer,makeUpTextureLayer)))

deltaInput = 0
deltaMakeUp = 1
resultant_skin_detail = morphing.morph_image_delta1_delta2(sourceTextureLayer,makeUpTextureLayer,deltaInput,deltaMakeUp,sourceTuple,makeUpTuple)
# cv2.imshow("addas",resultant_skin_detail)

alphaFactor = 0.8
resultant_image[:,:,1:3] = morphing.morph_image_delta1_delta2(sourceColor,makeUpColor,1 - alphaFactor,alphaFactor,sourceTuple,makeUpTuple,True)

scaleFactor = 0.5
sourceStructureLayer_halved = cv2.resize(sourceStructureLayer,(int(sourceStructureLayer.shape[1]*scaleFactor),int(sourceStructureLayer.shape[0]*scaleFactor)),interpolation=cv2.INTER_AREA)
# sourceStructureLayer_halved_blurred = cv2.blur(sourceStructureLayer_halved,(5,5))
dim = (sourceStructureLayer.shape[1],sourceStructureLayer.shape[0])
sourceStructureLayer_retained_blurred = cv2.resize(sourceStructureLayer_halved,dim,interpolation=cv2.INTER_AREA)
# cv2.imshow("www",sourceStructureLayer_retained_blurred)


laplacian_makeUpStructure = cv2.Laplacian(makeUpStructureLayer,cv2.CV_64F)
# cv2.imshow("dads",laplacian_makeUpStructure)

resultant_structure_layer = morphing.morph_image_delta1_delta2(sourceStructureLayer_retained_blurred,laplacian_makeUpStructure,1,1,sourceTuple,makeUpTuple)
# resultant_structure_layer = morphing.morph_image_delta1_delta2(sourceStructureLayer,laplacian_makeUpStructure,1,1,sourceTuple,makeUpTuple)

# cv2.imshow("abcdefg",np.hstack((sourceStructureLayer,resultant_structure_layer,laplacian_makeUpStructure)))
# print(np.max(resultant_structure_layer),np.min(resultant_structure_layer))
# print(np.max(resultant_skin_detail),np.min(resultant_skin_detail))

sum_detail_structure = resultant_skin_detail + resultant_structure_layer
sum_detail_structure = (sum_detail_structure - np.min(sum_detail_structure))/(np.max(sum_detail_structure)-np.min(sum_detail_structure))
# print(np.max(resultant_skin_detail + resultant_structure_layer),np.min(resultant_skin_detail + resultant_structure_layer))
# print (np.max(sum_detail_structure),np.min(sum_detail_structure))

resultant_image[:,:,0] = (sum_detail_structure*255).astype(np.uint8)
# cv2.imshow("www",np.hstack((resultant_skin_detail,resultant_structure_layer)))
# cv2.imshow("gsdhgh",np.hstack((resultant_image[:,:,0],resultant_image[:,:,1],resultant_image[:,:,2])))

resultant_image = cv2.cvtColor(resultant_image,cv2.COLOR_LAB2BGR)
# resultant_image = lipTranfer.lip_makeup(resultant_image,sourceImage,K1,K2)

cv2.imshow("Output",np.hstack((sourceImage,resultant_image,makeUpImage)))
# cv2.imshow("source",np.hstack((sourceStructureLayer,sourceTextureLayer)))
# cv2.imshow("ada",sourceStructureLayer+sourceTextureLayer)
# cv2.imshow("makeUp",np.hstack((makeUpStructureLayer,makeUpTextureLayer)))
# cv2.imshow("aaa",makeUpStructureLayer+makeUpTextureLayer)

key = cv2.waitKey(0)
if key & 0xFF == ord('s'):
    cv2.imwrite('out.jpg',np.hstack((sourceImage,resultant_image,makeUpImage)))


