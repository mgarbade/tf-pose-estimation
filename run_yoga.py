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
frame_rate = 25
num_student_frames = 75
startAtFrame = frame_rate * 60 * 2 + frame_rate * 49  # Start at 2:49 min
keyframes = [3 * frame_rate, 11 * frame_rate, 16 * frame_rate]  # time in seconds; compare teacher <-> student around +-3 seconds around these frames
keyframes_student = [x - frame_rate for x in keyframes]
num_keyframes = len(keyframes)
container_student_angles = np.zeros((num_keyframes, num_student_frames, 8))
container_teacher_angles = np.zeros((num_keyframes, 8))
min_distance_global = 360 * 8 * np.ones(num_keyframes)  # start from max possible distance: 360° x 8 angles
best_student_frame_ids = np.zeros(num_keyframes)

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
    angle1 = np.abs(teacher[0].angles[1:])
    angle2 = np.abs(student[0].angles[1:])
    # angle1 = angle1 if angle1 < 180 else 360 - angle1  # flip angles that are bigger than 180 degrees.
    # angle2 = angle2 if angle2 < 180 else 360 - angle2
    angle_distances = np.abs(angle1 - angle2)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video-student', type=str)
    parser.add_argument('--video-teacher', type=str)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    pose_estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap_student = cv2.VideoCapture(args.video_student)
    cap_teacher = cv2.VideoCapture(args.video_teacher)

    if cap_student.isOpened() is False:
        print("Error opening video stream or file")
    if cap_teacher.isOpened() is False:
        print("Error opening video stream or file")

    # Throw away first frames
    total_frames = cap_student.get(7)
    total_frames = cap_teacher.get(7)
    cap_student.set(1, startAtFrame)
    cap_teacher.set(1, startAtFrame)

    prev = 0
    time_start = time.time()
    counter = 0
    counter_keyframe = 0
    counter_student_frame = 0
    run_video = True
    while cap_student.isOpened():
        time_elapsed = time.time() - prev

        if time_elapsed > 1. / frame_rate and run_video:
            prev = time.time()

            res, img_student = cap_student.read()
            res_target, img_teacher = cap_teacher.read()
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

            if counter == keyframes[counter_keyframe] and counter_keyframe < 2:
                counter_keyframe = counter_keyframe + 1
                # cv2.waitKey(0)
            cv2.putText(img_teacher, "Counter %d, Keyframe %d / %d reached" %
                        (counter, counter_keyframe, keyframes[counter_keyframe]), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)


            numpy_horizontal = np.hstack((img_student_angles, img_teacher, img_student_accuracy))
            cv2.imshow('Numpy Horizontal', numpy_horizontal)

            # Record student poses
            for k in range(0, num_keyframes):
                if counter in range(keyframes_student[k], keyframes_student[k] + num_student_frames):
                    container_student_angles[k, counter_student_frame, :] = student[0].angles[1:]
                    counter_student_frame = np.mod(counter_student_frame + 1, num_student_frames)

            # Record teacher poses
            for k in range(0, num_keyframes):
                if counter == keyframes[k]:
                    container_teacher_angles[k, :] = teacher[0].angles[1:]

            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

            if counter / frame_rate == 20:
                print("End video here")
                run_video = False
        if not run_video:
            break

    cv2.destroyAllWindows()

    # Search for best student pose

    for k in range(0, num_keyframes):
        angle1 = container_teacher_angles[k, :]
        # angle1 = angle1 if angle1 < 180 else 360 - angle1
        for i in range(0, num_student_frames):
            angle2 = container_student_angles[k, i, :]
            # angle2 = angle2 if angle2 < 180 else 360 - angle2
            min_distance = np.sum(np.abs(angle1 - angle2))
            if min_distance < min_distance_global[k]:
                min_distance_global[k] = min_distance
                # save best student frame
                best_student_frame_ids[k] = i + keyframes_student[k]

    print(best_student_frame_ids)


    # Present final result:

    # Get corresponding teacher / student frames
    cap_student = cv2.VideoCapture(args.video_student)
    cap_teacher = cv2.VideoCapture(args.video_teacher)
    if cap_student.isOpened() is False:
        print("Error opening video stream or file")
    if cap_teacher.isOpened() is False:
        print("Error opening video stream or file")
    total_frames = cap_student.get(7)
    total_frames = cap_teacher.get(7)
    cap_student.set(1, startAtFrame)
    cap_teacher.set(1, startAtFrame)





    # Display corresponding frames with together with min_distance score






logger.debug('finished+')
