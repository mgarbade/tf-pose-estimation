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
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    cap_target = cv2.VideoCapture(args.target)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    if cap_target.isOpened() is False:
        print("Error opening video stream or file")

    # startAtFrame = 25 * 10 + 4
    # for i in range(startAtFrame):
    #     tmp1, tmp2 = cap.read()


    frame_rate = 25
    prev = 0
    time_start = time.time()
    while cap.isOpened():
        time_elapsed = time.time() - prev

        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            res, img_student = cap.read()
            res_target, img_teacher = cap_target.read()

            # Rotate ccw if necessary
            # image = cv2.transpose(image)
            # image = cv2.flip(image, flipCode=0)




            # Do something with your image here.
            student = e.inference(img_student, resize_to_default=(w > 0 and h > 0), upsample_size=4)
            if not args.showBG:
                img_student = np.zeros(img_student.shape)
            img_student = TfPoseEstimator.draw_humans(img_student, student, imgcopy=False)

            # Detect human on target image
            teacher = e.inference(img_teacher, resize_to_default=(w > 0 and h > 0), upsample_size=4)
            if not args.showBG:
                img_student = np.zeros(img_student.shape)
            img_teacher = TfPoseEstimator.draw_humans(img_teacher, teacher, imgcopy=False)

            cv2.putText(img_student, "FPS: %.1f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(img_teacher, "Time total: %.1f" % (time.time() - time_start), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            numpy_horizontal = np.hstack((img_student, img_teacher))
            cv2.imshow('Numpy Horizontal', numpy_horizontal)

            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()
logger.debug('finished+')
