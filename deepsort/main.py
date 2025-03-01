from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import numpy as np
import cv2

class SecondMilitaryTracker:
    def __init__(self, output, device):
        self.output = output
        self.device = device
        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(max_iou_distance=0.6, max_age=100)


    def load_model(self):
        # Load YOLO model
        model = YOLO("yolo11n.pt")
        model.to(self.device)
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)[0]
        return results

    # It's better to place everything in the same function using DeepSort. Otherwise method .update returns None.
    def get_results(self, frame, results):
        res_array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.3:
                res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)- int(y1)], float(score), int(class_id)))

        print(res_array)
        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_tlbr()
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            class_id = track.get_det_class()
            class_name = self.names[class_id]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
            cv2.putText(
                frame,
                f"ID: {track_id} {class_name}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.output)
        assert cap.isOpened(), "Cannot open video file"

        while True:
            ret, frame = cap.read()
            if not ret:
                break


            results = self.predict(frame)
            frame = self.get_results(frame, results)

            cv2.imshow('2nd version', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()



file = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
tracker = SecondMilitaryTracker(file, device)
tracker()


