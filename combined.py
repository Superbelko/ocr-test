# Import required modules
import argparse
import datetime
import math
import json
import sys
from collections import namedtuple
from pathlib import Path
from dataclasses import dataclass, astuple
from typing import List, Union

import numpy as np
import cv2 as cv
from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level

import perspective
from objdetect import DetectedObject, CLASSES, detect_objects
from lockfile import LockDummy, LockFile


WINDOW_LABEL = 'Debug view'

@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class TextRegion:
    vertices: np.ndarray
    bounding_box: Rect


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def get_file_ocr_result(image, rect):
    text = ''
    confidence = 0.0
    with PyTessBaseAPI() as api:
        api.SetImageFile(image)
        api.SetRectangle(*rect)
        text = api.GetUTF8Text()
        confidence = api.AllWordConfidences()
    return [text, confidence]


def get_blob_ocr_result(image, rect: Rect, ppi: int = 0):
    text = ''
    confidence = 0.0
    with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
        # only read numbers (doesn't seem to work)
        #api.SetVariable('tessedit_char_whitelist', '0123456789')
        api.SetImageBytes(*image)
        api.SetRectangle(*astuple(rect))
        if ppi != 0:
            api.SetSourceResolution(ppi)
        #api.Recognize()
        text = api.GetUTF8Text()
        confidence = api.AllWordConfidences()
        if not len(confidence):
            confidence = (0,)
    return [text, confidence]


def print_video_stats(cap):
    print('FrameRate: ', cap.get(cv.CAP_PROP_FPS))
    print('Dimensions: {} {}'.format(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print('TotalFrames: ', cap.get(cv.CAP_PROP_FRAME_COUNT))


def get_wait_time(cap):
    """Time in ms to wait before next frame to simulate realtime playback"""

    if cap.get(cv.CAP_PROP_FRAME_COUNT) < 2:
        return 1
    else:
        return int((1/cap.get(cv.CAP_PROP_FPS)) * 1000)


def ocr_image_region(image, region):

    # -------------------------
    # deproject image
    rect = region[0].vertices
    warped = perspective.four_point_transform(image, rect)

    # -------------------------
    # get text from image 
    ocr_image = warped
    ocr_image = cv.cvtColor(ocr_image, cv.COLOR_RGB2GRAY)
    img_w = ocr_image.shape[1]
    img_h = ocr_image.shape[0]
    tess_rect = Rect(0,0, img_w, img_h)
    img_bpp = 1
    img_bpl = int(img_bpp * img_w)

    text, textconf = get_blob_ocr_result((ocr_image.tobytes(), int(img_w), int(img_h), img_bpp, img_bpl), tess_rect)

    # do inverted 2nd pass and pick better result
    ocr_inv = cv.bitwise_not(ocr_image)
    text1, textconf1 = get_blob_ocr_result((ocr_inv.tobytes(), int(img_w), int(img_h), img_bpp, img_bpl), tess_rect)
    if textconf1[0] > textconf[0]:
        textconf = textconf1
        text = text1

    return (text, textconf[0])


def detect_text_areas(image, model, *args, **kwargs):

    results = []

    # Get frame height and width
    sourceSize = kwargs['sourceSize']
    width_ = sourceSize[0]
    height_ = sourceSize[1]
    inpWidth = image.shape[1]
    inpHeight = image.shape[0]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    net = model
    confThreshold = kwargs['confThreshold']
    nmsThreshold = kwargs['nmsThreshold']
    outNames = kwargs['outNames']

    # Create a 4D blob from frame.
    blob = cv.dnn.blobFromImage(image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Run the model
    net.setInput(blob)
    outs = net.forward(outNames)
    t, _ = net.getPerfProfile()

    # Get scores and geometry
    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)

    # Apply NMS
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            #cv.line(frame, p1, p2, (0, 255, 0), 1)

        xmin = vertices[0][0]
        xmax = 0
        ymin = vertices[0][1]
        ymax = 0
        for p in vertices:
            xmin = int(min(xmin, p[0]))
            ymin = int(min(ymin, p[1]))
            xmax = int(max(xmax, p[0]))
            ymax = int(max(ymax, p[1]))

        results.append((TextRegion(vertices, (xmin, ymin, xmax, ymax)), confidences[i[0]]))

    return results


def create_network(model):
    net = cv.dnn.readNet(model)

    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    return net, outNames


def adjust_dimension(size: float) -> int:
    """Adjusts size to satisfy EAST requirements"""

    return int(32 * math.ceil(size/32))


def set_next_frame(cap: cv.VideoCapture, frame: int) -> None:
    """Rewind capture to specified frame number if possible"""

    if frame != -1 and cap.get(cv.CAP_PROP_FRAME_COUNT) > 1:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame)


def is_too_wide_for(obj, dim):
    """Filters too wide objects in detected objects list"""

    left, top, right, bottom = obj.rect # rect
    return (right-left)/dim > 0.5


def is_rects_overlap(a: Rect, b: Rect) -> bool: 
    if int(a.xmin) > int(b.xmax) or int(a.xmax) < int(b.xmin):
        return False
    if int(b.ymin) > int(b.ymax) or int(a.ymax) < int(b.ymin):
        return False
    return True


def get_lock(dummy: bool) -> LockFile:
    if dummy:
        return LockDummy
    return LockFile

def main():
    parser = argparse.ArgumentParser(description='Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)')
    parser.add_argument('--input', help='Path to input image or video file.', required=True)
    parser.add_argument('--model', default='./models/frozen_east_text_detection.pb',
                        help='Path to a binary .pb file of model contains trained weights.')
    parser.add_argument('--width', type=int, default=320,
                        help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
    parser.add_argument('--height',type=int, default=320,
                        help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
    parser.add_argument('--thr',type=float, default=0.5,
                        help='Confidence threshold.')
    parser.add_argument('--nms',type=float, default=0.4,
                        help='Non-maximum suppression threshold.')
    parser.add_argument('--starttime',type=str, default=None,
                        help='Optional start time of the range (Video only).')
    parser.add_argument('--endtime',type=str, default=None,
                        help='Optional end time of the range (Video only).')
    parser.add_argument('--frame',type=int, default=-1,
                        help='Specify frame number to read at (Video only).')
    parser.add_argument('--step',type=int, default=1,
                        help='Video stream frame step count')
    parser.add_argument('--info', action='store_true', help='Show video file stats. Skips processing.')
    parser.add_argument('--debug', action='store_true', help='Show window with debug information')
    parser.add_argument('--nolock', action='store_true', help='Do not create lock file')
    args = parser.parse_args()

    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = adjust_dimension(args.width)
    inpHeight = adjust_dimension(args.height)
    model = args.model

    # trained model for object detection
    objdetectmodel = ('./models/frozen_inference_graph.pb', './models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # Open a video/image file or a camera stream
    imsrc = args.input

    cap = cv.VideoCapture(imsrc)

    if not cap.isOpened():
        print('Cannot open resource')
        exit(1)

    if args.info:
        print_video_stats(cap)
        exit()

    starttime = datetime.datetime.now()

    net, outNames = create_network(model)

    set_next_frame(cap, args.frame)
    
    wait_ms = get_wait_time(cap)

    lockfile = get_lock(args.nolock)(imsrc+'.lock')
    frames = []
    try:
        while cv.waitKey(wait_ms) < 0:
            # Read frame
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES)-1)

            lockfile.write(('Started: {}\nFrame: {}/{}').format(
                starttime.isoformat(), 
                current_frame, 
                int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
            )

            objects = detect_objects(frame, objdetectmodel, threshold=0.2)
            filtered_objects = list(filter(lambda obj: not is_too_wide_for(obj, frame.shape[1]), objects))

            # resize source image for text detection
            resized = cv.resize(frame, (inpWidth, inpHeight))

            text_areas = detect_text_areas(resized, net, sourceSize=(frame.shape[1], frame.shape[0]), outNames=outNames, confThreshold=confThreshold, nmsThreshold=nmsThreshold)

            ocrresults = []
            result = []
            ids = [] # only used for debug
            for i, region in enumerate(text_areas):

                overlaps = list(filter(lambda obj: is_rects_overlap(Rect(*region[0].bounding_box), Rect(*obj.rect)), 
                                filtered_objects))

                if not len(overlaps):
                    result.append(('', 0))
                    ids.append(i)
                    continue

                ocr = ocr_image_region(frame, region)
                result.append(ocr)
                
                ocrresults.append({
                    'rect': {
                        'x': region[0].bounding_box[0], 
                        'y': region[0].bounding_box[1], 
                        'x1': region[0].bounding_box[2], 
                        'y1': region[0].bounding_box[3]
                    }, 
                    'text': ocr[0], 
                    'confidence': ocr[1]
                })

            frames.append({
                'frame': current_frame,
                'time': round(cap.get(cv.CAP_PROP_POS_MSEC), 2), # .2f format
                'ocr': ocrresults
            })

            set_next_frame(cap, int(cap.get(cv.CAP_PROP_POS_FRAMES)-1) + args.step)

            if args.debug:
                for obj in filtered_objects:
                    left, top, right, bottom = obj.rect
                    cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (73, 255, 0), thickness=2)
                for i, text in enumerate(text_areas):
                    if i in ids:
                        continue
                    left, top, right, bottom = text[0].bounding_box
                    cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=1)
                    cv.putText(frame, '{} [{:.2f}%]'.format(result[i][0], result[i][1]), (int(right), int(bottom)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                cv.imshow(WINDOW_LABEL, frame)
                
    except Exception as e:
        print(e, file=sys.stderr)
        with open(imsrc+'.log') as f:
            f.write(e)

    finishtime = datetime.datetime.now()

    output = {
        'version': 1,
        'started': starttime.isoformat(),
        'finished': finishtime.isoformat(),
        'file': Path(imsrc).name,
        'frames': frames
    }

    print(json.dumps(output))

    if args.debug:
        cv.waitKey()


if __name__ == "__main__":
    main()
