import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import cv2
import random


from classification_model import load_model
from tools import preprocess_img

class DeepDetector:
    def __init__(self, path, device, threshold):
        self.path = path
        self.device = device
        self.model = self.load_model()
        self.tracker = DeepSort(max_iou_distance=0.5, max_age=10)
        self.threshold = threshold
        self.names = self.model.names

    def load_model(self):
        model = YOLO("/Users/maxkucher/PycharmProjects/bcs_ai/deepsort_v2/models/single_detector_v01.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        return self.model(frame)[0]

    def get_results(self, frame, results):
        res_array = []
        if len(results) != 0:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))

                tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)
                detected_objects = []

                for track in tracks:
                    bbox = track.to_ltrb()
                    idx = track.track_id
                    class_id = track.get_det_class()
                    detected_objects.append((bbox, idx, class_id))

                return detected_objects

        else:
            return None

    def predict_class(self, frame, detected_objects):
        class_names = ["apc", "tank", "ifv"]
        names = []
        # res = 2
        # name = "Analyzing"
        if detected_objects is not None:
            for obj in detected_objects:
                bbox, _ , _  = obj

                x1, y1, x2, y2 = map(int, bbox)
                print(x1, y1, x2, y2)
                image = frame[y1:y2, x1:x2]
                image = preprocess_img(image)
                image = image.to(self.device)

                classification_model = load_model(self.device)

                res = classification_model.predict(image)
                # res = random.randint(0, 2)
                name = class_names[int(res)]
                names.append(name)

            return names
        else:
            return names




    def draw(self, frame, detected_objects, names, previous_positions, movings, moving_status="Doesn't move"):
        if detected_objects is not None:
            for i, obj in enumerate(detected_objects):
                bbox, idx, class_id = obj
                x1, y1, x2, y2 = bbox

                center = np.array([(x1+x2)/2, (y1+y2)/2])

                if idx in previous_positions:
                    prev_center = previous_positions[idx]
                    speed = np.linalg.norm(center - prev_center)
                    if speed > 0:
                        moving_status = "Moving"

                previous_positions[idx] = center
                movings[idx] = moving_status

                if len(names) != 0:
                    name = names[i]
                else:
                    name = ""

                text = f"{idx}:{self.names[int(class_id)]}:{name}:{movings[idx]}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame
        else:
            return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        previous_positions = {}
        while True:
            movings = {}
            ret, frame = cap.read()

            if not ret:
                break

            results = self.results(frame)
            detected_objects = self.get_results(frame, results)
            names = self.predict_class(frame, detected_objects)
            print(names)
            upd_frane = self.draw(frame, detected_objects, names, previous_positions, movings)

            cv2.imshow('YOLO', upd_frane)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


path = "/Users/maxkucher/PycharmProjects/bcs_ai/deepsort/videos/Ukraine drone video shows attack on Russian tanks.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print(device)
threshold = 0.4
detector = DeepDetector(path, device, threshold)
detector()




    # def predict_class(self, frame, detected_objects):
    #     for obj in detected_objects:
    #         bbox, idx, score, class_id = obj
    #         x1, y1, x2, y2 = map(int, bbox)
    #         new_frame = frame[y1:y2, x1:x2]


device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("/Users/maxkucher/PycharmProjects/bcs_ai/deepsort_v2/models/best (7).pt")


model.to(device)
print(device)
# model.fuse()
print(model.names)
