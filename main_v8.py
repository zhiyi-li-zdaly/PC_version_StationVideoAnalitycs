# main_v5.py for version 4, adapt to real-time cameras
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# main_v8.py for version 8
# Author: Zhiyi Li, Date: 20230919
# Adapted from detect.py, to read configuration file for IP address and virtual lines. 
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import cv2
import yaml
import requests
import argparse
import os
import platform
import sys
import math
from pathlib import Path
import numpy as np
from numpy.linalg import norm

import time
import datetime
from datetime import datetime

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Computes the direction of the three given points
# Returns a positive value if they form a counter-clockwise orientation,
# A negative value if they form a clockwise orientation,
# And zero if they are collinear 
# https://www.codingninjas.com/codestudio/library/check-if-two-line-segments-intersect
def direction(p, q, r):
    # return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

# Checks if two line segments are collinear and overlapping
def areCollinearAndOverlapping(a1, b1, a2, b2):
    # Check if the line segments are collinear
    if direction(a1, b1, a2) == 0:
        # Check if the line segments overlap
        # if a2.x <= max(a1.x, b1.x) and a2.x >= min(a1.x, b1.x) and a2.y <= max(a1.y, b1.y) and a2.y >= min(a1.y, b1.y):
        if a2[0] <= max(a1[0], b1[0]) and a2[0] >= min(a1[0], b1[0]) and a2[1] <= max(a1[1], b1[1]) and a2[1] >= min(a1[1], b1[1]):
           return True

    return False

# Checks if two line segments intersect or not
def isintersect(a1, b1, a2, b2):
    # Compute the directions of the four line segments
    d1 = direction(a1, b1, a2)
    d2 = direction(a1, b1, b2)
    d3 = direction(a2, b2, a1)
    d4 = direction(a2, b2, b1)

    # Check if the two line segments intersect
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Check if the line segments are collinear and overlapping
    if areCollinearAndOverlapping(a1, b1, a2, b2) or areCollinearAndOverlapping(a2, b2, a1, b1):
        return True

    return False

DIRECTION_MAPPING = {
    (0, 0)   : '-',     # STILL
    (0, 1)   : 'L',     # LEFT     
    (0, -1)  : 'R',     # RIGHT
    (1, 0)   : 'D',     # DOWN
    (1, 1)   : 'DL',    # DOWN LEFT
    (1, -1)  : 'DR',    # DOWN RIGHT
    (-1, 0)  : 'U',     # UP
    (-1, 1)  : 'UL',    # UP LEFT
    (-1, -1) : 'UR',    # UP RIGHT
}

# To Detect direction of Object
# Returns two ints in a Tuple the values of int could be -1, 0 or 1
# For first Int it is a Vertical Movement
# 1 -> DOWN || 0 -> Still || -1 -> UP 
# For second Int it is a Horizontal Movement
# 1 -> Left || 0 -> Still || -1 -> Right 
def determine_direction(prev_center, current_center):
    h_dis, v_dis = (current_center[0] - prev_center[0], current_center[1] - prev_center[1])
    dir_v = 1 if v_dis > 0  else -1 if v_dis < 0 else 0
    dir_h = 1 if h_dis > 0  else -1 if h_dis < 0 else 0
    return dir_v, dir_h

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 480),  # inference size (height, width)

        # Change conf_thres to 0.05 instead. 
        # conf_thres=0.25,  # confidence threshold
        conf_thres = 0.05,
        # Finish conf_thres

        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # Define virtual line
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 640
    height = 480
    # start_point = (int(3/10 * width), int(1/10 * height))
    # end_point = (int(3/10 * width), int(9/10 * height))
    color =(0, 255, 0)
    thickness = 5

    # Read IP address and virtual lines from yml configuration file
    with open('samples/camera1.yml', 'r') as file: 
        config_service = yaml.safe_load(file)

    print (config_service)

    IP_address = config_service['camera']['IP_address']
    raw_start_point = config_service['camera']['start_point']
    raw_end_point = config_service['camera']['end_point']  
        
    # Parse points from string to integer
    # tuple(map(int, test_str.split(', ')))
    # Remove parenthesis of a string
    raw_start_point = raw_start_point.replace('(','').replace(')','')
    raw_end_point = raw_end_point.replace('(','').replace(')','') 

    # Covert string to tuple, learned from https://www.geeksforgeeks.org/python-convert-string-to-tuple/

    start_point = tuple(map(int, raw_start_point.split(',')))
    end_point = tuple(map(int, raw_end_point.split(',')))
 
    print("IP_address: ", IP_address)
    print("start_point: ", start_point)
    print("end_point: ", end_point)
    print("start_point len: ", len(start_point))
    print("end_point len: ", len(end_point))
    #########################

    camera_id = "admin"
    camera_password = "AdminAdmin1"
    source = "rtsp://" + camera_id + ":" + camera_password + "@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=1"
    print (source)

    # Get the size of source
    vcap = cv2.VideoCapture(source) # 0=camera 
    if vcap.isOpened(): 
        width  = int(vcap.get(3))  # float `width`
        height = int(vcap.get(4))  # float `height`
        print ("width: ", width, "height: ", height)
        

    # out_writter = cv2.VideoWriter('samples/video_1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640, 480))
    out_writter = cv2.VideoWriter('samples/video_0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

    store_video_flag = True
                          
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("Test after model loading")
    
    # Save the video
    
    # File to save the counting information
    local_file_name = "./samples/timeRecordCountingInformationPer10mins_sample0_v1.txt" 
    with open(local_file_name, "a") as f:
        outLine = "Station_id" + "," + "Time" + "," + " counting" + "\n"
        f.write(outLine)
    
    CustomerID = "255"
    StationID = "4083408367007383835"
    NameOfMetric = "CarsNumber"
    Camera_Category = "Camera 0"
    start_time = datetime.now()
    pre_min = start_time.minute
    sum = 0
    directions = dict()

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        print ("Test for Webcam")
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        print ("Test for Screenshots") 
    else:
        print ("Test for Images")
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # The dictionary stores previous frame objects key: id, value:{center_x, center_y}
    pre_center_points = []   # List, value: [center_x, center_y]
    id = 0
    cur_center_points = []

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Test
            # im = cv2.rotate(im, cv2.ROTATE_180)
            # Finish test

            im = torch.from_numpy(im).to(model.device)

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            # Testing
            cv2.line(im0, start_point, end_point, color, thickness)
            cv2.putText(im0, "Count: " + str(int(sum)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1.75, (0, 0, 255), 1, cv2.LINE_AA)
            direction_text = ','.join([f'{DIRECTION_MAPPING.get(k,"-")}: {len(v)}' for k,v in directions.items() if len(v) > 0])
            cv2.putText(im0, direction_text, (50, height), cv2.FONT_HERSHEY_SIMPLEX , 1.75, (0, 0, 255), 1, cv2.LINE_AA)

            # Write output to results
            if store_video_flag == True:
                out_writter.write(im0)            

            now_time = datetime.now()
            cur_min = now_time.minute

            d = (now_time - start_time).total_seconds()
            print ('second: ', d)

            min = int(d / 60) 
            print ('min: ', min)
            
            if min % 10 == 0 and cur_min != pre_min:	# For 10 mins, filter out same min
                current_count = int(sum)
                timestamp = str(datetime.now())
                '''
                url_link = """ 'https://bp.zdaly.com:5010/api/Survey/SaveVideoAnalytics?'+ 'CustomerID=' + str(self.CustomerID) + '&StationID=' +       
                           str(self.StationID) + '&NameOfMetric=' + str(NameOfMetric) +  '&Value=' + str(current_count) +'&Created=' +       
                           timestamp + '&Camera_Category=' + Camera_Category """
                '''                

                url_link = 'https://bp.zdaly.com:5010/api/Survey/SaveVideoAnalytics?CustomerID=251&StationID=408367007383835&NameOfMetric=CarsNumber&Value=' + str(current_count) + '&Created=' + str(timestamp) + '&Camera_Category=' + Camera_Category

                req = requests.post(url_link)

                with open(local_file_name, "a") as f:
                    outLine = str(CustomerID) + "," +  str(StationID) + "," + str(NameOfMetric) + "," + str(current_count) + "," + str(timestamp) + "," + Camera_Category + "," + direction_text + "\n"
                    f.write(outLine)
            
            pre_min = cur_min
            # print ("Draw a line and count")
            # Finish testing

            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            cur_center_points = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # Start the testing
                        label_class = label.split(" ")[0]
                        if str(label_class) == 'car' or str(label_class) == 'motocycle' or str(label_class) == 'bus' or str(label_class) == 'truck':
                            # annotator.box_label(xyxy, label, color=colors(c, True))
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            print ("xyxy: ", xyxy)

                            xmin = xyxy[0].int()
                            ymin = xyxy[1].int()
                            xmax = xyxy[2].int()
                            ymax = xyxy[3].int()
                        
                            # Center of objects
                            center_x = xmin + (xmax - xmin) / 2
                            center_y = ymin + (ymax - ymin) / 2
                
                            # same_object_detected = False
                            print("center_x: ", center_x, "center_y: ", center_y)
                            
                            same_object_detected = False
                            min_id = -1
                            min_dist = sys.maxsize
                            min_point = []

                            # print (len(pre_center_points))
                            # print (pre_center_points)              
                            
                            for pt in pre_center_points:
                                dist = math.hypot(center_x - pt[0], center_y - pt[1])
                                # print ("dist: ", dist)
                
                                if dist < min_dist:
                                    min_dist = dist
                                    min_point = pt

                            if min_dist < 50:       # Find it
                                print ("Same vehicle")
                
                                same_object_detected = True
 
                                pre_point = (int(min_point[0]), int(min_point[1]))
                                cur_point = (int(center_x), int(center_y))

                    
                                # print ("pre_point: ", pre_point)
                                # print ("cur_point: ", cur_point)

                                # Update the center_points
                                # print (start_point, end_point, pre_point, cur_point)
                                cv2.line(im0, pre_point, cur_point, color, thickness)
                    
                                # Check whether the line is intersect with virutal line
                                if isintersect(start_point, end_point, pre_point, cur_point):
                 
                                    # print (start_point, end_point, pre_point, cur_point)
                                    sum += 1
                                    direction = determine_direction(pre_point, cur_point)
                                    directions[direction] = directions.get(direction, []) + [cur_point]
                                    print ("intersection found")
                     
                                    cv2.line(im0, start_point, end_point, color, thickness)

                            else: # No same car were found in previous frame, but  the car may be near the area of virtual wall
                                  # Calculate perpendiicular distance between cur_point to line(start_point, end_point)
                                p1 = np.asarray(start_point)
                                p2 = np.asarray(end_point)
                                
                                cur_point = (int(center_x), int(center_y))
                                
                                p3 = np.asarray(cur_point)
                                
                                d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)

                                # print ("d: ", d)
                                if d < 100: # SPecify the range. 
                                    # Check whether this car is counted before                                    

                                    sum += 0.5  # Apply  less than 0.5 value, reduce the over-counting errors
           
                            cur_center_points.append([center_x, center_y])
                                              
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            pre_center_points = cur_center_points
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
 
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
