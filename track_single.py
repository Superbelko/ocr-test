# Import required modules
import argparse
import datetime
import math
import json
import sys
import functools
import pickle
from typing import Optional
from bisect import bisect_right
from collections import namedtuple
from pathlib import Path

import numpy as np
import cv2 as cv
from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level
from fuzzywuzzy import fuzz

import app.perspective as perspective
from app.objdetect import ObjDetectNetConfig, DetectedObject, CLASSES, detect_objects
from app.lockfile import LockDummy, LockFile
from combined import *

# maximum time threshold since tracked text was seen to give up further searching
LAST_SEEN_THRESH_SECONDS = 20


class App:
    pass

def exp_traversal(frames, *, key, depth: int = 1, stepthresh: int = 1) -> Optional[int]:
    '''Exponent split broad first traversing.
    Each depth level is split up in 1+2^h equal parts and traversed recursively.
    '''

    if len(frames) == 0 or depth < 1:
        return None
    numstops = 1 + math.ceil(math.pow(2, depth))
    steplength = math.ceil(len(frames) / numstops)
    if steplength < stepthresh:
        return None
    for i in range(numstops):
        pos = i*steplength
        if key(frames[pos]):
            return frames[pos]
    return exp_traversal(frames, depth=depth+1, key=key, stepthresh=stepthresh)


@dataclass
class InterpData:
    """Holds intermediate data for interpolation"""

    frame: int
    textarea: Rect
    trackobj: Rect


class VideoFrames:
    '''(Retarded) wrapper that is used for searching'''

    # this class is just horrible. will refactor later
    def __init__(self, cap, args, key, **context):
        self.context = context
        self.cap = cap
        self.key = key
        self.args = args
        self.cached = dict()
        self.ocrresults = []
        self.forward = list()
        self.backward = list()
    def __len__(self):
        return int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
    def __getitem__(self, index):
        #self.attempts.update({index: self.attempts.get(index, 0)+1})
        #print(f'search in frame {index} ({self.attempts[index]} times)')
        set_next_frame(self.cap, index+1)
        self.detect()
        return self.ocrresults[0]

    def current_frame(self) -> int:
        return int(self.cap.get(cv.CAP_PROP_POS_FRAMES)-1)

    def framerate(self) -> float:
        return self.cap.get(cv.CAP_PROP_FPS)

    def find_last_presence(self, starting_frame, *, comparer=None):
        if not comparer:
            comparer = self.key
        last_detected = starting_frame
        up_to_frame = min(last_detected + math.ceil(LAST_SEEN_THRESH_SECONDS * self.framerate()), len(self))
        frame_step = math.ceil(self.framerate() * 0.2) # 0.x times per second for smoother detection
        #print(f'start forward detection from {last_detected} to {up_to_frame}')

        for i in range(starting_frame, up_to_frame, frame_step):
            #print(f'forward detection, current frame {i} ')
            tmp = self[i]
            if comparer(tmp):
                last_detected = self.current_frame()

        # dirty way to do full list
        self.interp_back(last_detected, starting_frame)
        self.forward = self.backward
        self.backward.clear()
    
    def interp_back(self, known_first: int, known_last: int) -> None:
        temp: List[InterpData] = []
        first = self.cached.get(known_first, None)
        # find closest object
        textarea = self._ocrresult_to_rect(first['ocrresults'][0])
        objects = sorted(first['objects'], key=lambda obj: obj.rect.center().distance(textarea.center()))
        #print(f'distance {textarea.center().distance(objects[0].rect.center())}')
        temp.append(InterpData(known_first, textarea, objects[0].rect))

        same_or_default = known_last == known_first or known_last == -1
        hi_frame = known_first
        lo_frame = min(known_first - math.ceil(LAST_SEEN_THRESH_SECONDS * self.framerate()), len(self)) if same_or_default else known_last
        frame_step = math.ceil(self.framerate() * 0.2) # 0.x times per second for smoother detection
        #print(f'start reverse detection from {known_first} to {lo_frame}')

        for i in reversed(range(lo_frame, hi_frame, frame_step)):
            _ = self[i]
            cache = self.cached.get(i, None)
            # find closest object
            prev = temp[-1]
            textarea = self._ocrresult_to_rect(cache['ocrresults'][0])
            #if self.key(cache['ocrresults'][0]):
            #    print(f'matched at {i}')
            objects = sorted(cache['objects'], key=lambda obj: obj.rect.center().distance(prev.trackobj.center()))
            #print(f'distance {textarea.center().distance(objects[0].rect.center())}')
            temp.append(InterpData(i, textarea, objects[0].rect))

            # interpolate last 3 frames
            if len(temp) > 2:
                temp[0].trackobj = self._interp_rects(temp[-2], temp[-1], prev)

            if self.args.debug:
                set_next_frame(self.cap, i+1)
                _, img = self.cap.read()
                self._draw_interp_centers(img, temp[-1])
                for pos in range(min(12, len(temp))):
                    self._draw_onion_rect(img, pos, temp[-1-pos])

                cv.namedWindow(WINDOW_LABEL, cv.WINDOW_NORMAL)
                cv.imshow(WINDOW_LABEL, img)
                keycode = cv.waitKey(500)
                if keycode == 27: # esc key
                    break

        # filter results, such as out of view one's
        # ...
        
        self.backward = temp

    def _ocrresult_to_rect(self, ocrres) -> Rect:
        r = ocrres.get('rect', None)
        return Rect(*r.values()) if r else Rect(0,0,0,0)

    def _draw_interp_centers(self, img, data: InterpData) -> None:
        start = astuple(data.trackobj.center().as_int())
        end = astuple(data.textarea.center().as_int())
        cv.rectangle(img, (int(data.trackobj.xmin), int(data.trackobj.ymin)), (int(data.trackobj.xmax), int(data.trackobj.ymax)), (0, 255, 0), thickness=2)
        cv.line(img, start, end, (0, 255, 255), 2)
        cv.ellipse(img, start, (5, 5), 0, 0, 0, (0, 255, 255), 20)
        cv.ellipse(img, end, (5, 5), 0, 0, 0, (0, 255, 255), 20)

    def _draw_onion_rect(self, img, diff: int, data: InterpData) -> None:
        gstart = 255
        gend = 50
        mul = diff/10
        gfin = gstart - (gend * mul)
        font_scale = max(1.5, min(1.5 - (0.5 * mul), 0.5))
        start = astuple(data.trackobj.center().as_int())
        end = astuple(data.textarea.center().as_int())
        cpy = img.copy()
        cv.rectangle(cpy, (int(data.trackobj.xmin), int(data.trackobj.ymin)), (int(data.trackobj.xmax), int(data.trackobj.ymax)), (0, gfin, 0), thickness=2)
        cv.putText(cpy, f'{diff}', (int(data.trackobj.xmax + 40 + (diff * 3.5)), int(data.trackobj.ymin - 20 + (diff * 2.5))), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, gfin, 0), 3)
        img[:] = cv.addWeighted(img, 0.7, cpy, 0.3, gamma=0)

    def _interp_rects(self, a: InterpData, b: InterpData, c: InterpData):
        return Rect(
            int( (a.trackobj.xmin * 0.5) + (b.trackobj.xmin * 0.3) + (c.trackobj.xmin * 0.2) ),
            int( (a.trackobj.ymin * 0.5) + (b.trackobj.ymin * 0.3) + (c.trackobj.ymin * 0.2) ),
            int( (a.trackobj.xmax * 0.5) + (b.trackobj.xmax * 0.3) + (c.trackobj.xmax * 0.2) ),
            int( (a.trackobj.ymax * 0.5) + (b.trackobj.ymax * 0.3) + (c.trackobj.ymax * 0.2) ))


    def detect(self, *, forceupdate: bool = False):

        if forceupdate:
            self.cached.pop(self.current_frame(), None)

        cacheitem = self.cached.get(self.current_frame(), None)
        if cacheitem is not None:
            #print(f'using cached frame {self.current_frame()}')
            self.ocrresults = cacheitem['ocrresults']
            self.ocrresults.sort(key=lambda a: 100-fuzz.partial_ratio(a['text'], self.args.track))
            return

        args = self.args
        # Read and store arguments
        confThreshold = args.thr
        nmsThreshold = args.nms
        inpWidth = adjust_dimension(args.width)
        inpHeight = adjust_dimension(args.height)
        #model = args.model

        # trained model for object detection
        objdetectmodel = ObjDetectNetConfig(f'{APP_DIR}/models/frozen_inference_graph.pb', f'{APP_DIR}/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

        eastmodel = f'{APP_DIR}/models/frozen_east_text_detection.pb'
        net, outNames = create_network(eastmodel)

        # --------------------------

        self.ocrresults = []
        hasFrame, frame = self.cap.read()
        
        if not hasFrame:
            self.ocrresults = list({'text': ''})
            return
        
        self.frame = frame
        objects = detect_objects(frame, objdetectmodel, threshold=0.2)
        filtered_objects = list(filter(lambda obj: not is_too_wide_for(obj, frame.shape[1]), objects))

        # resize source image for text detection
        resized = cv.resize(frame, (inpWidth, inpHeight))

        text_areas = detect_text_areas(resized, net, sourceSize=(frame.shape[1], frame.shape[0]), outNames=outNames, confThreshold=confThreshold, nmsThreshold=nmsThreshold)

        #ocrresults = []
        result = []
        ids = [] # only used for debug
        for i, region in enumerate(text_areas):

            overlaps = list(filter(lambda obj: is_rects_overlap(region[0].bounding_box, obj.rect),
                                   filtered_objects))

            if len(overlaps) == 0:
                result.append(('', 0))
                ids.append(i)
                continue

            ocr = ocr_image_region(frame, region)
            
            self.ocrresults.append({
                'rect': {
                    'x': region[0].bounding_box.xmin, 
                    'y': region[0].bounding_box.ymin, 
                    'x1': region[0].bounding_box.xmax, 
                    'y1': region[0].bounding_box.ymax
                }, 
                'text': ocr[0], 
                'confidence': ocr[1]
            })

        self.ocrresults.sort(key=lambda a: 100-fuzz.partial_ratio(a['text'], self.args.track))

        if len(self.ocrresults) < 1:
            self.ocrresults.append({'text': '', 'rect': None, 'confidence': 0})

        # cap.read() advances the frame number so we subtract back to current frame
        self.cached[self.current_frame()-1] = {'ocrresults': self.ocrresults.copy(), 'objects': filtered_objects}
        self.write_cache('cache.pkl')

    def write_cache(self, path):
        with open(path, 'wb') as f:
            f.write(pickle.dumps(self.cached))

    def load_cache(self, path):
        try:
            with open(path, 'rb') as f:
                self.cached = pickle.loads(f.read())
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description='Process video or still image, and makes list of frames where tracked text was found.')
    parser.add_argument('--input', help='Path to input image or video file.', required=True)
    parser.add_argument('--width', type=int, default=0,
                        help='Preprocess input image by resizing to a specific width.')
    parser.add_argument('--height', type=int, default=0,
                        help='Preprocess input image by resizing to a specific height.')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='Confidence threshold.')
    parser.add_argument('--nms', type=float, default=0.4,
                        help='Non-maximum suppression threshold.')
    parser.add_argument('--starttime', type=str, default=None,
                        help='Optional start time of the range (Video only).')
    parser.add_argument('--endtime', type=str, default=None,
                        help='Optional end time of the range (Video only).')
    parser.add_argument('--frame', type=int, default=-1,
                        help='Specify frame number to read at (Video only).')
    parser.add_argument('--step', type=int, default=1,
                        help='Video stream frame step count')
    parser.add_argument('--track', type=str, required=True, help='A text marker to track')
    parser.add_argument('--info', action='store_true', help='Show video file stats. Skips processing.')
    parser.add_argument('--debug', action='store_true', help='Show window with debug information')
    parser.add_argument('--nolock', action='store_true', help='Do not create lock file')
    args = parser.parse_args()

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
    frames = []

    lockfile = get_lock(args.nolock)(imsrc+'.lock')
    try:
        cache = 'cache.pkl'
        vid = VideoFrames(cap, args, lambda texts: fuzz.partial_ratio(texts['text'], args.track) > 0)
        vid.load_cache(cache)
        comparer = lambda a: fuzz.partial_ratio(a['text'], args.track) > 75
        res = exp_traversal(vid, stepthresh=args.step, key=comparer)
        anchor_frame = vid.current_frame()
        vid.write_cache(cache)
        #print(f'last detected occurence in frame {anchor_frame}')
        #print('performing forward detection to determine last known frame...')
        vid.find_last_presence(anchor_frame, comparer=comparer)
        vid.write_cache(cache)
        #print('performing reverse interpolation...')
        vid.key = comparer
        vid.interp_back(anchor_frame, anchor_frame)

        for f in sorted(vid.backward + vid.forward, key=lambda interpdata: interpdata.frame):
            ocr = vid[f.frame]
            frames.append({
                "frame": f.frame,
                "timecode": "",
                "ocr": ocr if comparer(ocr) else None,
                "people": [
                    {"rect": {
                        "x": int(f.trackobj.xmin),
                        "y": int(f.trackobj.ymin),
                        "x1": int(f.trackobj.xmax),
                        "y1": int(f.trackobj.ymax)
                    }},
                ]
            })

    except Exception as e:
        import traceback
        print(e, file=sys.stderr)
        with open(imsrc+'.log', 'w+') as f:
            f.write(str(e) + '\n\n')
            f.write(traceback.format_exc())
    finally:
        lockfile.__exit__()

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
