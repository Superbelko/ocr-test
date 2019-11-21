# Import required modules
import argparse
import math
import json

import numpy as np
import cv2 as cv
from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level

import perspective
results = []

############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description='Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', required=True)
parser.add_argument('--model', default='./frozen_east_text_detection.pb',
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
parser.add_argument('--info', action='store_true', help='Show video file stats. Skips processing.')
parser.add_argument('--debug', action='store_true', help='Show window with debug information')
args = parser.parse_args()

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


def get_blob_ocr_result(image, rect, ppi=0):
    text = ''
    confidence = 0.0
    with PyTessBaseAPI(psm=PSM.SINGLE_LINE) as api:
        api.SetImageBytes(*image)
        api.SetRectangle(*rect)
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
    return int((1/cap.get(cv.CAP_PROP_FPS)) * 1000)

def main():
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    # Load network
    net = cv.dnn.readNet(model)

    # Create a new named window
    kWinName = "Text OCR Extractor"
    if args.debug:
        cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    # Open a video/image file or a camera stream
    imsrc = args.input

    cap = cv.VideoCapture(imsrc)

    if not cap.isOpened():
        print('Cannot open resource')
        exit(1)

    if args.info:
        print_video_stats(cap)
        exit()

    if args.frame != -1 and cap.get(cv.CAP_PROP_FRAME_COUNT) > 1:
        cap.set(cv.CAP_PROP_POS_FRAMES, args.frame)

    # Read initial frame
    #hasFrame, frame = cap.read()
    wait_ms = 1 if cap.get(cv.CAP_PROP_FRAME_COUNT) < 2 else get_wait_time(cap)
    while cv.waitKey(wait_ms) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # make grayscale for later use
        # pre-filter image
        thresh_mode = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        #thresh_mode = cv.ADAPTIVE_THRESH_MEAN_C
        gray = frame
        gray = cv.cvtColor(gray, cv.COLOR_RGB2GRAY)
        #gray = cv.adaptiveThreshold(gray, 255, thresh_mode, cv.THRESH_BINARY, 11, 2)
        #gray = cv.medianBlur(gray, 3)

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        outs = net.forward(outNames)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

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
                cv.line(frame, p1, p2, (0, 255, 0), 1)

            xmin = vertices[0][0]
            xmax = 0
            ymin = vertices[0][1]
            ymax = 0
            for p in vertices:
                xmin = int(min(xmin, p[0]))
                ymin = int(min(ymin, p[1]))
                xmax = int(max(xmax, p[0]))
                ymax = int(max(ymax, p[1]))
            #cv.line(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 1)
            tess_rect = (xmin, ymin, xmax-xmin, ymax-ymin)

            # -------------------------
            # deproject image

            rect = vertices
            warped = perspective.four_point_transform(gray, rect)

            # -------------------------
            # get text from image 
            ocr_image = warped
            img_w = ocr_image.shape[1]
            img_h = ocr_image.shape[0]
            tess_rect = (0,0, img_w, img_h)
            img_bpp = 1
            #img_bpp = int(ocr_image.size / (img_w*img_h))
            img_bpl = int(img_bpp * img_w)

            text, textconf = get_blob_ocr_result((ocr_image.tobytes(), int(img_w), int(img_h), img_bpp, img_bpl), tess_rect)

            # do inverted 2nd pass and pick better result
            ocr_inv = cv.bitwise_not(ocr_image)
            text1, textconf1 = get_blob_ocr_result((ocr_inv.tobytes(), int(img_w), int(img_h), img_bpp, img_bpl), tess_rect)
            if textconf1[0] > textconf[0]:
                textconf = textconf1
                text = text1

            outfmt = '{} [{}%]'.format(text, textconf[0])
            cv.putText(frame, outfmt, (xmax, ymax), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # TODO: write results
            results.append({'rect': {'x': xmin, 'y': ymin, 'x1': xmax, 'y1': ymax}, 'text': text, 'confidence': textconf[0]})

            #print(outfmt)
            #exit()
            #cv.imshow(kWinName,ocr_image)
            #cv.waitKey()
            # -------------------------

        # Put efficiency information
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        if args.debug:
            cv.imshow(kWinName,frame)
        
        print(json.dumps(results))

if __name__ == "__main__":
    main()
