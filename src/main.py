from face_detection import FaceDetectionClass
from facial_landmarks_detection import FacialLandmarksClass
from gaze_estimation import GazeEstimationClass
from head_pose_estimation import HeadPoseEstimationClass
from mouse_controller import MouseController
from argparse import ArgumentParser
from in_fedr import InputFeeder
import math
import cv2
import os
import numpy as np
import logging as log
import time
import warnings
warnings.filterwarnings("ignore")

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model.")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file with a trained model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file with a trained model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-flgs", "--visualization_flag", required=False, nargs='+',
                        default=[],
                        help="Example: --flgs fd fl hp ge (Seperate each flgs by space)"
                             "for see the visualization of different model outputs of each frms,"
                             "fd for Face Detection Model, fl for Facial Landmark Detection Model"
                             "hp for Head Pose Estimation Model, ge for Gaze Estimation Model.")
    return parser


def draw_axes(frms, cntr_of_face, yaw, pitch, roll, scale, foc_lnth):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    c_x = int(cntr_of_face[0])
    c_y = int(cntr_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x
    cam_mtrx = build_camera_matrix(cntr_of_face, foc_lnth)
    x_axs = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    y_axs = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    z_axs = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    z_axs_1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    ot = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    ot[2] = cam_mtrx[0][0]
    x_axs = np.dot(r, x_axs) + ot
    y_axs = np.dot(r, y_axs) + ot
    z_axs = np.dot(r, z_axs) + ot
    z_axs_1 = np.dot(r, z_axs_1) + ot
    xp_2 = (x_axs[0] / x_axs[2] * cam_mtrx[0][0]) + c_x
    yp_2 = (x_axs[1] / x_axs[2] * cam_mtrx[1][1]) + c_y
    p_2 = (int(xp_2), int(yp_2))
    cv2.line(frms, (c_x, c_y), p_2, (0, 0, 255), 2)
    xp_2 = (y_axs[0] / y_axs[2] * cam_mtrx[0][0]) + c_x
    yp_2 = (y_axs[1] / y_axs[2] * cam_mtrx[1][1]) + c_y
    p_2 = (int(xp_2), int(yp_2))
    cv2.line(frms, (c_x, c_y), p_2, (0, 255, 0), 2)
    xp_1 = (z_axs_1[0] / z_axs_1[2] * cam_mtrx[0][0]) + c_x
    yp_1 = (z_axs_1[1] / z_axs_1[2] * cam_mtrx[1][1]) + c_y
    p_1 = (int(xp_1), int(yp_1))
    xp_2 = (z_axs[0] / z_axs[2] * cam_mtrx[0][0]) + c_x
    yp_2 = (z_axs[1] / z_axs[2] * cam_mtrx[1][1]) + c_y
    p_2 = (int(xp_2), int(yp_2))
    cv2.line(frms, p_1, p_2, (255, 0, 0), 2)
    cv2.circle(frms, p_2, 3, (255, 0, 0), 2)
    return frms


def build_camera_matrix(cntr_of_face, foc_lnth):
    c_x = int(cntr_of_face[0])
    c_y = int(cntr_of_face[1])
    cam_mtrx = np.zeros((3, 3), dtype='float32')
    cam_mtrx[0][0] = foc_lnth
    cam_mtrx[0][2] = c_x
    cam_mtrx[1][1] = foc_lnth
    cam_mtrx[1][2] = c_y
    cam_mtrx[2][2] = 1
    return cam_mtrx


def main():
    # command line args
    args = build_argparser().parse_args()
    in_file_pth = args.input
    log_obj = log.getLogger()
    onenene_flags = args.visualization_flag
    if in_file_pth == "CAM":
        in_fedr = InputFeeder("cam")
    else:
        if not os.path.isfile(in_file_pth):
            log_obj.error("ERROR: INPUT PATH IS NOT VALID")
            exit(1)
        in_fedr = InputFeeder("video", in_file_pth)

    model_paths = {'Face_detection_model': args.face_detection_model,
                   'Facial_landmarks_detection_model': args.facial_landmarks_model,
                   'head_pose_estimation_model': args.head_pose_model,
                   'gaze_estimation_model': args.gaze_estimation_model}

    print(model_paths['Face_detection_model'])
    face_detct_model_obj = FaceDetectionClass(model_name=model_paths['Face_detection_model'],
                                                          device=args.device, threshold=args.prob_threshold,
                                                          extensions=args.cpu_extension)

    facial_landmark_detct_model_obj = FacialLandmarksClass(
        model_name=model_paths['Facial_landmarks_detection_model'],
        device=args.device, extensions=args.cpu_extension)

    gaze_estim_model_obj = GazeEstimationClass(
        model_name=model_paths['gaze_estimation_model'], device=args.device, extensions=args.cpu_extension)
    head_pose_estim_model_obj = HeadPoseEstimationClass(
        model_name=model_paths['head_pose_estimation_model'], device=args.device, extensions=args.cpu_extension)
    mouse_ctrl_obj = MouseController('medium', 'fast')
    start_time = time.time()
    face_detct_model_obj.load_model()
    log_obj.error("Face detection model loaded: time: {:.3f} ms".format((time.time() - start_time) * 1000))
    first_mark = time.time()
    facial_landmark_detct_model_obj.load_model()
    log_obj.error(
        "Facial landmarks detection model loaded: time: {:.3f} ms".format((time.time() - first_mark) * 1000))
    sec_mark = time.time()
    head_pose_estim_model_obj.load_model()
    log_obj.error("Head pose estimation model loaded: time: {:.3f} ms".format((time.time() - sec_mark) * 1000))
    third_mark = time.time()
    gaze_estim_model_obj.load_model()
    log_obj.error("Gaze estimation model loaded: time: {:.3f} ms".format((time.time() - third_mark) * 1000))
    load_total_time = time.time() - start_time
    log_obj.error("Total loading time: time: {:.3f} ms".format(load_total_time * 1000))
    log_obj.error("All models are loaded successfully..")
    in_fedr.load_data()
    log_obj.error("Input feeder are loaded")

    counter = 0
    start_inf_time = time.time()
    log_obj.error("Start inferencing on input video.. ")
    for flgs, frms in in_fedr.next_batch():
        if not flgs:
            break
        key_in = cv2.waitKey(60)
        counter = counter + 1
        face_coord, face_img = face_detct_model_obj.predict(frms.copy())

        if face_coord == 0:
            continue

        head_pose_estim_model_out = head_pose_estim_model_obj.predict(face_img)

        l_eye_img, r_eye_img, eye_coord = facial_landmark_detct_model_obj.predict(face_img)

        mouse_coord, gaze_vec = gaze_estim_model_obj.predict(l_eye_img, r_eye_img,
                                                                             head_pose_estim_model_out)

        if len(onenene_flags) != 0:
            prev_window = frms.copy()
            if 'fd' in onenene_flags:
                if len(onenene_flags) != 1:
                    prev_window = face_img
                else:
                    cv2.rectangle(prev_window, (face_coord[0], face_coord[1]),
                                  (face_coord[2], face_coord[3]), (0, 150, 0), 3)
            if 'fl' in onenene_flags:
                if not 'fd' in onenene_flags:
                    prev_window = face_img.copy()
                cv2.rectangle(prev_window, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]),
                              (150, 0, 150))
                cv2.rectangle(prev_window, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]),
                              (150, 0, 150))
            if 'hp' in onenene_flags:
                cv2.putText(prev_window,
                            "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(head_pose_estim_model_out[0],
                                                                             head_pose_estim_model_out[1],
                                                                             head_pose_estim_model_out[2]),
                            (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
            if 'ge' in onenene_flags:

                yaw = head_pose_estim_model_out[0]
                pitch = head_pose_estim_model_out[1]
                roll = head_pose_estim_model_out[2]
                foc_lnth = 950.0
                scale = 50
                cntr_of_face = (face_img.shape[1] / 2, face_img.shape[0] / 2, 0)
                if 'fd' in onenene_flags or 'fl' in onenene_flags:
                    draw_axes(prev_window, cntr_of_face, yaw, pitch, roll, scale, foc_lnth)
                else:
                    draw_axes(frms, cntr_of_face, yaw, pitch, roll, scale, foc_lnth)

        if len(onenene_flags) != 0:
            img_horiz = np.hstack((cv2.resize(frms, (500, 500)), cv2.resize(prev_window, (500, 500))))
        else:
            img_horiz = cv2.resize(frms, (500, 500))

        cv2.imshow('Visualization', img_horiz)
        mouse_ctrl_obj.move(mouse_coord[0], mouse_coord[1])

        if key_in == 27:
            log_obj.error("exit key is pressed..")
            break
    inference_time = round(time.time() - start_inf_time, 1)
    fps = int(counter) / inference_time
    log_obj.error("counter {} seconds".format(counter))
    log_obj.error("total inference time {} seconds".format(inference_time))
    log_obj.error("fps {} frms/second".format(fps))
    log_obj.error("Video has ended")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.txt'), 'w') as f:
        f.write(str(inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(load_total_time) + '\n')

    in_fedr.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
