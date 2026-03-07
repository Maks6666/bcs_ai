import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch 




class Tracker:
    def __init__(self, path, device, yolo_link):
        self.device = device
        self.path = path 
        self.yolo_link = yolo_link
        self.tracker = DeepSort(max_age=5, max_iou_distance=0.4)
        self.model = self.load_model()
        self.names = self.model.names

    def load_model(self):
        model = YOLO(self.yolo_link)
        model.fuse()
        model.to(self.device)
        return model
    
    def results(self, frame):
        return self.model.predict()