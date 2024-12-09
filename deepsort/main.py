from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import numpy as np
import cv2


class SecondMilitaryTracker:
    def __init__(self, output):
        self.output = output
        self.model = self.load_model()
        self.names = self.model.names
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.tracker = DeepSort(
            max_age=100,
            n_init=8,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None
        )

    def load_model(self):
        model = YOLO("yolo/best (16).pt")
        model.fuse()
        return model

    def predict(self, frame):
        return self.model.predict(frame, verbose=True, conf=0.4)

    def get_results(self, results):
        array = []
        for result in results[0]:
            bboxes = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()
            t_array = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], conf[0], class_id[0]]
            array.append(t_array)
        return np.array(array)


    def draw_boxes(self, frame, bboxes, ids, class_id):
        for box, idx, cls in zip(bboxes, ids, class_id):

            label = f"{idx}:{self.names[cls]}"

            cv2.rectangle(frame, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.output)
        assert cap.isOpened(), "Cannot open video file"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.predict(frame)
            array_results = self.get_results(results)

            fin_res = self.tracker.update_tracks(array_results, frame)
            print(fin_res)

            cv2.imshow('2nd version', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


file = "videos/Ukraine drone video shows attack on Russian tanks.mp4"
tracker = SecondMilitaryTracker(file)



