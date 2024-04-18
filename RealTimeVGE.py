import torch
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np

import src.utils.gaze as gaze_util


import sys
sys.path.append(r'./VGE-pytorch')
from vge import VGE


height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0

def from_gaze2d(gaze, output_size, scale=1.0):
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([
        -iris_distance * sin_phi * cos_theta,
        iris_distance * sin_theta,
    ])
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset

    return iris_centre






def clip_eye_region(eye_region_landmarks, image):
    oh, ow = 36, 60

    def process_coords(coords_list):
        return np.array([(x, y) for (x, y) in coords_list])

    def process_rescale_clip(eye_landmarks):
        eye_width = 1.5 * abs(eye_landmarks[0][0] - eye_landmarks[1][0])
        eye_middle = (eye_landmarks[0] + eye_landmarks[1]) / 2

        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = -eye_middle[0] + 0.5 * eye_width
        recentre_mat[1, 2] = -eye_middle[1] + 0.5 * oh / ow * eye_width

        scale_mat = np.asmatrix(np.eye(3))
        np.fill_diagonal(scale_mat, ow / eye_width)

        transform_mat = recentre_mat * scale_mat

        eye = cv.warpAffine(image, transform_mat[:2, :3], (ow, oh))
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        return eye, np.asarray(transform_mat)

    left_eye_landmarks = process_coords(eye_region_landmarks[2:4])
    right_eye_landmarks = process_coords(eye_region_landmarks[0:2])
    left_eye_image, left_transform_mat = process_rescale_clip(left_eye_landmarks)
    right_eye_image, right_transform_mat = process_rescale_clip(right_eye_landmarks)
    
    return [left_eye_image, left_transform_mat], [right_eye_image, right_transform_mat]

def estimate_gaze(eye_image, transform_mat, model, is_left: bool):
    oh, ow = 36, 60

    eye_image = np.expand_dims(eye_image, -1)
    eye_image = np.transpose(eye_image, (2, 0, 1))
    eye_image = torch.unsqueeze(torch.Tensor(eye_image), dim=0)

    direction,_,_,_ = model((eye_image,0,1))

    direction = direction.cpu()
    predict = direction.detach().numpy()
    

    iris_center = from_gaze2d(predict[0],(oh,ow))
    if is_left:
        iris_center[0] = 60 - iris_center[0]
    iris_center = (iris_center - [transform_mat[0][2], transform_mat[1][2]]) / transform_mat[0][0]
    return predict, iris_center




def real_time_VGE(modelPath='./model256_20.pth'):
    d = ".\\src\\models\\mmod_human_face_detector.dat"
    p = ".\\src\\models\\shape_predictor_5_face_landmarks.dat"
    detector = dlib.cnn_face_detection_model_v1(d)
    predictor = dlib.shape_predictor(p)

    our_model = VGE()
    our_model = torch.nn.DataParallel(our_model)

    if torch.cuda.is_available():
        our_model.load_state_dict(torch.load(modelPath))
    else:
        our_model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    our_model.eval()

    cap = cv.VideoCapture(0)

    while True:
        # load the input image and convert it to grayscale
        _, image = cap.read()
        image = cv.flip(image, 1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # detect face in the grayscale image
        faceRects = detector(gray, 0)

        # loop over the face detections
        for (i, faceRect) in enumerate(faceRects):
            shape = predictor(gray, faceRect.rect)
            shape = face_utils.shape_to_np(shape)

            for (j, (x, y)) in enumerate(shape):
                if j in range(0, 4):
                    cv.circle(image, (x, y), 2, (0, 255, 0), -1)
            eye_region_landmarks = shape[0:4]
            left_eye, right_eye = clip_eye_region(eye_region_landmarks, gray)
            left_gaze, left_iris_center = estimate_gaze(
                cv.flip(left_eye[0], 1), 
                transform_mat=left_eye[1],
                model=our_model,
                is_left=True,
            )
            left_gaze[0][1] = -left_gaze[0][1]
            right_gaze, right_iris_center = estimate_gaze(
                right_eye[0],
                transform_mat=right_eye[1], 
                model=our_model,
                is_left=False,
            )
            image = gaze_util.draw_gaze(image, left_iris_center, left_gaze[0])
            image = gaze_util.draw_gaze(image, right_iris_center, right_gaze[0])

        cv.imshow("Output", image)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()

# Call the function
if __name__ == "__main__":
    real_time_VGE()

















