import torch
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np
from tqdm import tqdm

import src.models.gaze_modelbased as GM
import src.utils.gaze as gaze_util


import sys
sys.path.append(r'./VGE-pytorch')
from vge import VGE



def clip_eye_region(eye_region_landmarks, image):
    # Output size.
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

def estimate_gaze(eye_image, transform_mat, model, is_left: bool,ourModel):
    eye_image = np.expand_dims(eye_image, -1)
    # Change format to NCHW.
    eye_image = np.transpose(eye_image, (2, 0, 1))
    eye_image = torch.unsqueeze(torch.Tensor(eye_image), dim=0)

    direction,_,_,_ = ourModel((eye_image,0,1))

 
    if torch.cuda.is_available():
        eye_image = eye_image.cuda()
    eye_input = eye_image

    # Do predict by elg_model.
    _, ldmks_predict, _ = model(eye_input)


    # Get parameters for model_based gaze estimator.
    ldmks = ldmks_predict.cpu().detach().numpy()
    iris_center = np.array(ldmks[0][-2])
    
    predict = direction.cpu().detach().numpy()
    
    iris_center = ldmks_predict[0].cpu().detach().numpy()[16]
    if is_left:
        iris_center[0] = 60 - iris_center[0]
    iris_center = (iris_center - [transform_mat[0][2], transform_mat[1][2]]) / transform_mat[0][0]
    return predict, iris_center




def video_ELG_VGE(modelPath='./model256_20.pth'):
    # initialize dlib's face detector (mmod) and then create
    # the facial landmark predictor
    d = ".\\src\\models\\mmod_human_face_detector.dat"
    p = ".\\src\\models\\shape_predictor_5_face_landmarks.dat"
    detector = dlib.cnn_face_detection_model_v1(d)
    predictor = dlib.shape_predictor(p)
    elg_model = torch.load('./models/v0.2/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth', map_location=torch.device('cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elg_model = elg_model.to(device)    
    elg_model.eval()

    our_model = VGE()
    our_model = torch.nn.DataParallel(our_model)
    if torch.cuda.is_available():
        our_model.load_state_dict(torch.load(modelPath))
    else:
        our_model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    our_model.eval()

    # Step 1: Record a video
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('./outputVideos/outputELGandVGE.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv.imshow('Recording', frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

    # Step 2: Split video into frames
    vidcap = cv.VideoCapture('./outputVideos/outputELGandVGE.avi')
    success, image = vidcap.read()
    count = 0
    img_array = []
    while success:
        img_array.append(image)
        success, image = vidcap.read()
        count += 1

    size = (640,480)
    out = cv.VideoWriter('./outputVideos/VideoELGandVGE.avi', cv.VideoWriter_fourcc(*'XVID'), 20.0, size)

    for i in tqdm(range(count)):
        image = img_array[i]
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
                model=elg_model,
                is_left=True,
                ourModel = our_model
            )
            left_gaze[0][1] = -left_gaze[0][1]
            right_gaze, right_iris_center = estimate_gaze(
                right_eye[0],
                transform_mat=right_eye[1], 
                model=elg_model,
                is_left=False,
                ourModel = our_model
            )
            image = gaze_util.draw_gaze(image, left_iris_center, left_gaze[0])
            image = gaze_util.draw_gaze(image, right_iris_center, right_gaze[0])

            out.write(image)

    out.release()

# Call the function
if __name__ == "__main__":
    video_ELG_VGE()










