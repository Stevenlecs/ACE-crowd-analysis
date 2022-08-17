#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE
import datetime
import time
import numpy as np
import imutils
import cv2
import os
import csv
import json
from video_process import video_process
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from re import I
import sys
import glob

sys.path.append("")
import argparse
import tensorflow as tf
import requests

from video_process import video_process

from ace import analyticservice, grpcservice


def registar_analytic():
    '''
    Simple registration, probably should make modular in future
    '''
    ace_server_url = "http://192.168.255.10:5000/api/v1/add_analytic"
    payload = json.dumps(
        {"analytic_host": "173.72.194.154", "analytic_name": "crowd-analysis"}
    )
    headers = {
        'Content-Type': 'application/json'
    }
    print("ace start")
    response = requests.request("POST", ace_server_url, headers=headers, data=payload)
    print(response)
    return response


def detect(handler, frame_override=None):
    '''
    Inputs:
        handler (ace.analytichandler.FrameHandler) - ace frame handler obj (https://github.com/usnistgov/ACE/blob/develop/lang/python/ace/analytichandler.py#L50)
    '''
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    movement_data_file = open('processed_data/movement_data.csv', 'w')
    crowd_data_file = open('processed_data/crowd_data.csv', 'w')
    # sd_violate_data_file = open('sd_violate_data.csv', 'w')
    # restricted_entry_data_file = open('restricted_entry_data.csv', 'w')

    movement_data_writer = csv.writer(movement_data_file)
    crowd_data_writer = csv.writer(crowd_data_file)
    # sd_violate_writer = csv.writer(sd_violate_data_file)
    # restricted_entry_data_writer = csv.writer(restricted_entry_data_file)

    if os.path.getsize('processed_data/movement_data.csv') == 0:
        movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
    if os.path.getsize('processed_data/crowd_data.csv') == 0:
        crowd_data_writer.writerow(
            ['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])
    frame, bboxes = video_process(handler, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, frame_override=frame_override)
    movement_data_file.close()
    crowd_data_file.close()
    if frame_override is None:
        for b in bboxes:
            # x1 ... looks like this is the format based on code, probably need to verify
            handler.add_bounding_box(
                classification=b[0],
                confidence=float(b[1]),
                x1=int(b[2][0]),
                y1=int(b[2][1]),
                x2=int(b[2][2]),
                y2=int(b[2][3]))
    return frame, bboxes

if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_on_video', default="", help="Do not use ACE, just run on a video file")
    parser.add_argument("--grpc", default=False,
                        help="If true, this analytic will set up a gRPC service instead of a REST service.",
                        action="store_true")
    parser.add_argument("--grpc_port", default=50052, help="Port the analytic will run on.")
    args = parser.parse_args()
    """
    parser.add_argument('--load', default="/data/model/model-7400", help='load a model for evaluation.', required=True)

    '''
    # TF serving would be nice to use, but would for quick ACE demo, skip for now
    # Probably come back to tensorflow serving. Might be fun to play with    
    parser.add_argument('--output-serving', help='Save a model to serving file')
    '''

    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    """
    '''
    Load Model
    '''
    # Read from video
    IS_CAM = VIDEO_CONFIG["IS_CAM"]
    #cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])

    # Load YOLOv3-tiny weights and config
    WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
    CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]

    # Load the YOLOv3-tiny pre-trained COCO dataset
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    # Set the preferable backend to CPU since we are not using GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the names of all the layers in the network
    ln = net.getLayerNames()
    # Filter out the layer names we dont need for YOLO
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Tracker parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    if IS_CAM:
        max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
    else:
        max_age = DATA_RECORD_RATE * TRACK_MAX_AGE
        if max_age > 30:
            max_age = 30
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    START_TIME = time.time()

    '''
    Registar Analytic with ACE
    '''
    if args.test_on_video != "":
        cap = cv2.VideoCapture(args.test_on_video)
        print(cap.isOpened())
        while cap.isOpened():
            ret, cvFrame = cap.read()
            img, boxes = detect(handler=None, frame_override=cvFrame)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
        exit(0)
    else:
        registar_analytic()
        '''
        Run ACE Service
        '''
        if args.grpc:
            svc = grpcservice.AnalyticServiceGRPC()
            svc.register_name("crowd-analysis")
            svc.RegisterProcessVideoFrame(detect)
            sys.exit(svc.Run(analytic_port=args.grpc_port))
        else:
            svc = analyticservice.AnalyticService(__name__, )
            svc.register_name("crowd-analysis")
            svc.RegisterProcessVideoFrame(detect)
            sys.exit(svc.Run())
