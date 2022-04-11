import cv2

from Detector import YOLOV5_Detector

video_path = "testImage/test.mp4"
vid = cv2.VideoCapture(video_path)
detector = YOLOV5_Detector(weights='best.pt',
                           img_size=640,
                           confidence_thres=0.25,
                           iou_thresh=0.45,
                           agnostic_nms=True,
                           augment=True)
detector.detect_on_video(video_path)
