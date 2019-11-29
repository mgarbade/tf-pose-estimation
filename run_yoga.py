import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def draw_angle_confidence(img, teacher, student):
    # cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
    neck_id = 1
    shoulder_left_id = 5
    shoulder_right_id = 2
    elbow_left_id = 6
    elbow_right_id = 3
    wrist_left_id = 7
    wrist_right_id = 4
    hip_left_id = 11
    hip_right_id = 8
    knee_left_id = 12
    knee_right_id = 9
    ankle_left_id = 13
    ankle_right_id = 10

    angle_ids = [shoulder_left_id,
                 elbow_left_id,
                 shoulder_right_id,
                 elbow_right_id,
                 hip_left_id,
                 knee_left_id,
                 hip_right_id,
                 knee_right_id]

    # Absolute distance between angles
    angle_distances = np.abs((np.mod(np.abs(teacher[0].angles[1:]), 180) - np.mod(np.abs(student[0].angles[1:]), 180)))
    print(angle_distances)
    overlay_student = img.copy()
    img_student_accuracy = img.copy()
    alpha = 0.7
    image_h, image_w = overlay_student.shape[:2]
    counter = 0
    for i in angle_ids:
        body_part = student[0].body_parts[i]
        center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
        distance = angle_distances[counter]
        counter = counter + 1
        color = (0, 0, 255)
        if distance < 30:
            color = (255, 0, 0)
        if distance < 15:
            color = (0, 255, 0)
        cv2.circle(overlay_student, center, 10, color, -1, 8, 0)
    cv2.addWeighted(overlay_student, alpha, img_student_accuracy, 1 - alpha, 0, img_student_accuracy)
    return img_student_accuracy


keyframes = [10, 31, 35, 68, 80]  # compare teacher <-> student around +-3 seconds around these frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    pose_estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    cap_target = cv2.VideoCapture(args.target)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    if cap_target.isOpened() is False:
        print("Error opening video stream or file")

    startAtFrame = 25 * 60 * 2 + 25 * 49
    for i in range(startAtFrame):
        tmp1, tmp2 = cap.read()
        tmp3, tmp4 = cap_target.read()


    frame_rate = 25
    prev = 0
    time_start = time.time()
    counter = 0
    while cap.isOpened():
        time_elapsed = time.time() - prev

        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            res, img_student = cap.read()
            res_target, img_teacher = cap_target.read()
            counter = counter + 1

            # Rotate ccw if necessary
            # img_student = cv2.transpose(img_student)
            img_student = cv2.flip(img_student, flipCode=1)

            # Do something with your image here.
            student = pose_estimator.inference(img_student, resize_to_default=(w > 0 and h > 0), upsample_size=4, estimate_paf=False)
            img_student_angles = img_student.copy()
            img_student_angles = TfPoseEstimator.draw_humans(img_student_angles, student, imgcopy=False)

            # Detect human on target image
            teacher = pose_estimator.inference(img_teacher, resize_to_default=(w > 0 and h > 0), upsample_size=4, estimate_paf=False)
            img_teacher = TfPoseEstimator.draw_humans(img_teacher, teacher, imgcopy=False)

            img_student_accuracy = draw_angle_confidence(img_student, teacher, student)

            cv2.putText(img_student_angles, "FPS: %.1f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(img_teacher, "Time total: %.1f" % (time.time() - time_start), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(img_student_accuracy, "Frame-Number: %d" % (counter / 25), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            numpy_horizontal = np.hstack((img_student_angles, img_teacher, img_student_accuracy))
            cv2.imshow('Numpy Horizontal', numpy_horizontal)

            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()
logger.debug('finished+')
