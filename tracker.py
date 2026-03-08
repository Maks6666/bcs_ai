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
        self.threshold = 0.4

    def load_model(self):
        model = YOLO(self.yolo_link)
        model.fuse()
        model.to(self.device)
        return model
    
    def results(self, frame):
        return self.model.predict(frame, classes=[0], conf=0.3, verbose=False)[0]
    
    def get_results(self, results, frame):
        res_array = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                res_array.append((bbox, float(score), int(class_id)))

        
        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bboxes = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()

            results.append((bboxes, idx, class_id))
    
        return results
    
    def draw(self, results, frame):
        if results is not None:
            colour = (0, 255, 0)
            for (bboxes, idx, class_id) in results:
                x1, y1, x2, y2 = map(int, bboxes)
                name = self.names[int(class_id)]
                text = f"{idx}:{name}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

            return frame
        
    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()

            if not ret: 
                break

            results = self.results(frame)
            res_array = self.get_results(results, frame)
            upd_frmae = self.draw(res_array, frame)

            cv2.imshow('YOLO Tracker', upd_frmae)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # return frame (bytecode: jpeg) + logs
        cap.release()
        cv2.destroyAllWindows()



path = 1
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
yolo_link = "yolo12n.pt"
tracker = Tracker(path, device, yolo_link)
tracker()


