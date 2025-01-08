import cv2
import torch
from sort import Sort
from ultralytics import YOLO
import numpy as np
from time import time
import random

from connect import Session, get_data, get_status, Vehicles
from tools import get_path, process_image
from models import model

class FirstMilitaryTracker:
    def __init__(self, output, save_to_db, record = True):
        self.output = output
        self.model = self.load_model()
        self.names = ['apc', 'ifv', 'tank']
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.save_to_db = save_to_db
        self.record = record

        if self.save_to_db == True:
            self.session = Session()

        else:
            self.session = None


    def load_model(self):
        model = YOLO("classification_detection/yolo/vehicles.pt")
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

    def draw_boxes(self, frame, boxes, ids, types, movings):
        for box, idx, type, moving_status in zip(boxes, ids, types, movings):

            # name = self.names[res.item()]
            label = f"{idx}:{type}:{moving_status}"

            cv2.rectangle(frame, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def __call__(self):

        if self.save_to_db == True:
            try:
                test_session = Session()
                t_obj = test_session.query(Vehicles).first()
                print("Connected to DB!")
            except Exception as e:
                print(f"Connection failed: {e}")

        out = None
        cap = cv2.VideoCapture(self.output)
        assert cap.isOpened(), "Video capture failed."

        sort = Sort(max_age=80, min_hits=4, iou_threshold=0.30)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        number = random.randint(1, 1000)
        output_file = f"output/video_{number}.mp4"

        previouse_positions = {}

        if self.record:
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        while True:
            # type = "Analyzing..."
            #
            # moving_status = "Doesnt move"

            movings = []
            detected_obj = []
            types = []

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
                x1, y1, x2, y2 = map(int, box)

                # make sense when it filled by at least 2 center points:
                # for example you've got a video with two frames. On the first one object were detected
                # and because 'previouse_positions' dictionary is empty at the beginning of video processing,
                # so for "f idx in previouse_positions" we move to the 'else' part and add the index to the dictionary
                # as key and current center value (of first frame) as a value. But then we processing the seocnd frame and the loop:
                # "if idx in previouse_positions:" turns True (if object detected) so we take a previouse center value (from first frame) as
                # prev_center and calculate an euclidean distance between current center and previouse one. If distance is bigger tham 0,
                # them we mark an object as a moving one.


                center = np.array([x1 + x2 / 2, y1 + y2 / 2])


                if idx in previouse_positions:

                    prev_center = previouse_positions[idx]
                    speed = np.linalg.norm(center - prev_center)

                    if speed > 0:
                        moving_status = "Moving"
                    else:
                        moving_status = "Doesn't move"

                else:
                    moving_status = "Doesn't move"

                previouse_positions[idx] = center
                movings.append(moving_status)


                obj_frame = frame[y1:y2, x1:x2]

                if obj_frame.size == 0:
                    continue

                image = process_image(obj_frame)
                image = image.to(self.device)
                res = model.predict(image)
                types.append(self.names[res.item()])


                detected_obj.append(idx)

                if self.save_to_db == True:
                    obj = self.session.query(Vehicles).filter_by(status_at_moment="detected").all()
                    get_data(self.session, idx, self.names[int(res.item())], len(obj))

            if self.session:
                get_status(self.session, detected_obj)


            frame = self.draw_boxes(frame, bboxes, ids, types, movings)

            end = time()
            fps = 1 / round(end - start, 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f'Total objects: {summa}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('1st version', frame)

            if out:
                out.write(frame)


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
        if self.record:
            out.release()
        cv2.destroyAllWindows()

path = get_path("videos")
tracker = FirstMilitaryTracker(path, save_to_db=True, record=True)
tracker()













