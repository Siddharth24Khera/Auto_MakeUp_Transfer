from scipy.spatial import Delaunay
import numpy as np
import dlib
import cv2
import tao_asari


def rect_to_bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(feature_points, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (feature_points.part(i).x, feature_points.part(i).y)

    return coords

class ForeheadCoordinateStore:
    def __init__(self,feature_points,image):
        print "Select 10 points on forehead (left to right)"
        cv2.namedWindow('Select_Forehead_Points')
        cv2.setMouseCallback('Select_Forehead_Points', self.select_point)
        self.points = np.zeros((78, 2), dtype="int")
        self.count = 0
        self.inputFeature = len(feature_points)
        self.output_marked_image = np.copy(image)
        for i in range(len(feature_points)):
            self.points[i] = feature_points[i]
        for (x, y) in feature_points:
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
        while self.count != 10:
            cv2.imshow("Select_Forehead_Points", self.output_marked_image)
            cv2.waitKey(20)
        cv2.imshow("Select_Forehead_Points", self.output_marked_image)
        cv2.waitKey(20)
        cv2.destroyWindow("Select_Forehead_Points")

    def getOutputMarkedImage(self):
        return self.output_marked_image

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print "Points selected " + str(self.count + 1)
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
            self.points[self.inputFeature + self.count] = np.array([x, y], dtype="int")
            self.count = self.count + 1
        if self.count == 10:
            print "Points successfully selected"



def landmark_detection(image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tao_asari_enhanced = tao_asari.tao_asari_enhancement(image)
    tao_gray = cv2.cvtColor(tao_asari_enhanced, cv2.COLOR_BGR2GRAY)

    rect_faces = np.copy(image)

    dets = detector(tao_gray, 1)
    # Handle case when no face found
    for rect in dets:
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(rect_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if len(dets) == 0:
        print "No face detected in input image"
        exit()

    if len(dets) > 1:
        print "More than one face detected in input image"
        exit()

    feature_points = predictor(gray_img, rect)
    feature_points = shape_to_np(feature_points)

    (x, y, w, h) = rect_to_bb(rect)



    ForeheadSelector = ForeheadCoordinateStore(feature_points, image)

    output_marked_image = ForeheadSelector.getOutputMarkedImage()
    feature_points = ForeheadSelector.points

    maxX = feature_points[0][0]
    minX = feature_points[0][0]
    maxY = feature_points[0][1]
    minY = feature_points[0][1]

    for (x, y) in feature_points:
        if x == 0 and y == 0:
            continue
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    triangulation = Delaunay(feature_points)
    triangles = triangulation.simplices

    lip_point_indices = [i for i in range(48, 68)]
    left_eye_point_indices = [i for i in range(36, 42)]
    right_eye_point_indices = [i for i in range(42, 48)]

    lip_triangles =[]
    left_eye_triangles = []
    right_eye_triangles = []

    K = np.ones((image.shape[0], image.shape[1])


                )

    for i in range(len(triangles)):
        triangle = triangles[i]
        if (triangle[0] in lip_point_indices) and (triangle[1] in lip_point_indices) and (triangle[2] in lip_point_indices):
            lip_triangles.append(i)
            continue
        if (triangle[0] in left_eye_point_indices) and (triangle[1] in left_eye_point_indices) and (triangle[2] in left_eye_point_indices):
            left_eye_triangles.append(i)
            continue
        if (triangle[0] in right_eye_point_indices) and (triangle[1] in right_eye_point_indices) and (triangle[2] in right_eye_point_indices):
            right_eye_triangles.append(i)


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < minY or i > maxY or j > maxX or j < minX:
                K[i][j]=-1
                continue
            triangle_num = triangulation.find_simplex((j, i))
            if triangle_num == -1:
                K[i][j] = -1
                continue
            if (triangle_num in lip_triangles):
                K[i][j] = 2
                continue
            if (triangle_num in left_eye_triangles) or (triangle_num in right_eye_triangles):
                K[i][j] = 0


    for triangle in triangles:
        cv2.line(output_marked_image, (feature_points[triangle[0]][0], feature_points[triangle[0]][1]),
                 (feature_points[triangle[1]][0], feature_points[triangle[1]][1]), (0, 0, 0))
        cv2.line(output_marked_image, (feature_points[triangle[0]][0], feature_points[triangle[0]][1]),
                 (feature_points[triangle[2]][0], feature_points[triangle[2]][1]), (0, 0, 0))
        cv2.line(output_marked_image, (feature_points[triangle[1]][0], feature_points[triangle[1]][1]),
                 (feature_points[triangle[2]][0], feature_points[triangle[2]][1]), (0, 0, 0))

    cv2.rectangle(output_marked_image, (minX, minY), (maxX, maxY), (255, 0, 0), 2)
    # cv2.imshow("abcksd", output_marked_image)
    # cv2.waitKey(0)

    # Right now, working under assumption of only one face in the image
    return output_marked_image,feature_points,triangulation,K



