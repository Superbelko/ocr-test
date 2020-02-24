import argparse
import json
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2 as cv

from blur import laplacian_variance
from app.common import Rect
import perspective


@dataclass(frozen=True)
class ObjDetectNetConfig:
    model: str
    config: str


@dataclass
class DetectedObject:
    class_id: int
    confidence: float
    rect: Rect


# Pretrained classes in the model
CLASSES = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def detect_objects(image: np.ndarray, nnconfig: ObjDetectNetConfig, threshold: float=0.2) -> List[DetectedObject]:
    """Detects all objects in an image
    Accepts opencv compatible image and single tuple with NN weight and description file paths

    Returns array of all objects as tuple containing class id, confidence value and rect tuple
    """

    image_height, image_width, _ = image.shape

    net = cv.dnn.readNetFromTensorflow(nnconfig.model, nnconfig.config)
    net.setInput(cv.dnn.blobFromImage(image, 1, (300, 300), (127.5, 127.5, 127.5), swapRB = True, crop = False))
    
    output = net.forward()

    results = []
    for detection in output[0,0,:,:]:

        confidence = float(detection[2])

        if confidence < threshold:
            continue

        class_id = int(detection[1])

        # look only for people
        #if class_id != 1:
        #    continue

        #print('{} {:.3f} {}'.format(class_id , confidence, classname))
        left = detection[3] * image_width
        top = detection[4] * image_height
        right = detection[5] * image_width
        bottom = detection[6] * image_height

        results.append(DetectedObject(class_id, confidence, Rect(left,top,right,bottom)))

    return results


def main():
    parser = argparse.ArgumentParser(description='Object detection using TF MobileNet SSD network')
    parser.add_argument('--input', help='Path to input image or video file.', required=True)
    parser.add_argument('--thr',type=float, default=0.2,
                    help='Confidence threshold.')
    parser.add_argument('--model', default='./models/frozen_inference_graph.pb',
                    help='Path to a binary .pb file of model contains trained weights.')
    parser.add_argument('--modelcfg', default='./models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
                    help='Path to model config file.')
    parser.add_argument('--debug', action='store_true', help='Show window with debug information')
    args = parser.parse_args()


    frame = cv.imread(args.input)
    objects = detect_objects(frame, ObjDetectNetConfig(args.model, args.modelcfg), args.thr)

    round_float = lambda a: round(a, 2)

    results = []
    for classid, confidence, rect in objects:
        left, top, right, bottom = rect
        subimage = frame[int(top):int(bottom), int(left):int(right)]
        subimage = cv.cvtColor(subimage, cv.COLOR_RGB2GRAY)
        #subimage = cv.fastNlMeansDenoising(subimage,None,2,7,21)
        variance = laplacian_variance(subimage)

        results.append({
            'classid': classid, 
            'conconfidence': round_float(confidence), 
            'rect': [round_float(left), round_float(top), round_float(right), round_float(bottom)], 
            'variance': round_float(variance)
        })

    # output results to console
    print(json.dumps(results))

    # visualization for debug
    if args.debug:
        for obj in results:
            left, top, right, bottom = obj['rect']
            variance = obj['variance']
            classname = CLASSES.get(obj['classid'], 'unknown')
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv.putText(frame, '{} [{:.2f}]'.format(classname, confidence) ,(int(right), int(bottom)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            cv.putText(frame, '{:.1f} ({})'.format(variance, ('Blur' if variance < 100 else 'Ok')) ,(int(right), int(bottom)+20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0))

        cv.imshow('Debug view', frame)

        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
