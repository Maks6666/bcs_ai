import cv2
import torch
from sort import Sort
from ultralytics import YOLO
import numpy as np
from time import time

from connect import Session, get_data, get_status, Vehicles
from tools import get_path

class FirstMilitaryTracker:
    def __init__(self, output, save_to_db = True):
        self.output = output
        self.model = self.load_model()
        self.names = self.model.names
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.save_to_db = save_to_db

        if self.save_to_db:
            self.session = Session()
        else:
            self.session = None


    def load_model(self):
        model = YOLO("yolo/detector.pt")
        model.fuse()
        return model

    def predict(self, model, frame):
        return model.predict(frame, verbose=True, conf = 0.3)

    def get_results(self, results):
        summa = 0
        detection_list = []

        for result in results[0]:
            summa += 1
            bbox = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            merger_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], conf[0], class_id[0]]
            detection_list.append(merger_detection)

        return np.array(detection_list), summa

    def draw_boxes(self, frame, boxes, ids, class_id):
        for box, idx, cls in zip(boxes, ids, class_id):

            name = self.names[cls]
            label = f"{idx}:{name}"

            cv2.rectangle(frame, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.output)
        assert cap.isOpened(), "Video capture failed."

        sort = Sort(max_age=80, min_hits=4, iou_threshold=0.30)

        while True:
            detected_obj = []


            start = time()
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break


            results = self.predict(self.model, frame)
            final_results, summa = self.get_results(results)

            if len(final_results) == 0:
                final_results = np.empty((0, 5))


            t_res = sort.update(final_results)

            bboxes = t_res[:, :-1]
            ids = t_res[:, -1].astype(int)
            class_id = final_results[:, -1].astype(int)


            for box, idx, cls in zip(bboxes, ids, class_id):
                detected_obj.append(idx)

                if self.save_to_db:
                    obj = self.session.query(Vehicles).filter_by(status_at_moment="detected").all()
                    get_data(self.session, idx, self.names[int(cls)], len(obj))

            if self.session:
                get_status(self.session, detected_obj)


            frame = self.draw_boxes(frame, bboxes, ids, class_id)

            end = time()
            fps = 1 / round(end - start, 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f'Total objects: {summa}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('1st version', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                if self.session:
                    self.session.close()
                break

            if cv2.waitKey(1) & 0xFF == ord("d"):
                if self.session:
                    self.session.query(Vehicles).delete()
                    self.session.commit()
                    self.session.close()
                break


        cap.release()
        cv2.destroyAllWindows()

path = get_path("videos")
tracker = FirstMilitaryTracker(path, save_to_db=False)
tracker()













