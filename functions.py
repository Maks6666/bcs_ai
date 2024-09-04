import cv2
import numpy as np
from datetime import datetime
import pandas as pd


# looks for closest target point from the best point of fire
def total_coordinates_func(lst, frame, color, point, shells):
    if lst:


        results = []

        x_min = min(coord[0] for coord in lst) - 10
        y_min = min(coord[1] for coord in lst) - 10
        x_max = max(coord[0] for coord in lst) + 10
        y_max = max(coord[1] for coord in lst) + 10

        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 1)
        cv2.circle(frame, (x_center, y_center), 3, color, thickness=cv2.FILLED)

        # cv2.circle(frame, (x_center - (x_center // 6), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center + (x_center // 6), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center - (y_center // 6)), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center + (y_center // 6)), 3, color, thickness=cv2.FILLED)
        #
        # cv2.circle(frame, (x_center-(x_center//3), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center+(x_center//3), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center - (y_center//3)), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center + (y_center//3)), 3, color, thickness=cv2.FILLED)
        #
        #
        # cv2.circle(frame, (x_center - (x_center // 2), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center + (x_center // 2), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center - (y_center // 2)), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center + (y_center // 2)), 3, color, thickness=cv2.FILLED)
        #
        #
        # cv2.circle(frame, (int(x_center - (x_center // 1.5)), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center + int((x_center // 1.5)), y_center), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center - int((y_center // 1.5))), 3, color, thickness=cv2.FILLED)
        # cv2.circle(frame, (x_center, y_center + int((y_center // 1.5))), 3, color, thickness=cv2.FILLED)

        if shells == True:
            coordinates = [
                        (x_center, y_center), (x_center - (x_center // 6), y_center), (x_center + (x_center // 6), y_center),(x_center, y_center - (y_center // 6)), (x_center, y_center + (y_center // 6)),
                        (x_center-(x_center//3), y_center), (x_center+(x_center//3), y_center), (x_center, y_center - (y_center//3)), (x_center, y_center + (y_center//3)),
                        (x_center - (x_center // 2), y_center), (x_center + (x_center // 2), y_center), (x_center, y_center - (y_center // 2)), (x_center, y_center + (y_center // 2)),
                        (x_center - int((x_center // 1.5)), y_center), (x_center + int((x_center // 1.5)), y_center), (x_center, y_center - int((y_center // 1.5))), (x_center, y_center + int((y_center // 1.5)))

                        ]

            for coordinate in coordinates:
                t_res = np.sqrt((coordinate[0] - point[0]) ** 2 + (coordinate[1] - point[1]) ** 2)
                results.append(t_res)
            res = np.argmin(results)
            point_coordinates = coordinates[res]
            # target drawing
            cv2.circle(frame, (point_coordinates[0], point_coordinates[1]), 10, (0, 0, 255), thickness=1)
            cv2.circle(frame, (point_coordinates[0], point_coordinates[1]), 8, (0, 0, 255), thickness=1)
            cv2.circle(frame, (point_coordinates[0], point_coordinates[1]), 6, (0, 0, 255), thickness=1)

            cv2.line(frame, (point_coordinates[0] - 6, point_coordinates[1]), (point_coordinates[0] + 6,  point_coordinates[1]), color, thickness=1)
            cv2.line(frame, (point_coordinates[0], point_coordinates[1] - 6), (point_coordinates[0], point_coordinates[1] + 6), color, thickness=1)


            return point_coordinates

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def find_closest_object(center, objects_dict, threshold=5):
    min_distance = float('inf')
    closest_obj_id = None
    for obj_id, obj_center in objects_dict.items():
        distance = np.linalg.norm(np.array(center) - np.array(obj_center))
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_obj_id = obj_id


    return closest_obj_id


def time_of_detection():
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_second = now.second

    return f"{current_hour}:{current_minute}:{current_second}"


def choose_object(dataframe, frame):
    if len(dataframe) == 0:
        print("DataFrame is empty")
        return


    min_idx = dataframe["obj_id"].idxmin()
    t_data = dataframe.loc[min_idx]

    required_columns = ['obj_id', 'name', 'time', 'x1', 'y1', 'x2', 'y2']
    if not set(required_columns).issubset(t_data.index):
        print("Missing required columns in the data")
        return

    obj_id, name, time, x1, y1, x2, y2 = t_data[['obj_id', 'name', 'time', 'x1', 'y1', 'x2', 'y2']].tolist()

    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


    cv2.circle(frame, (x_center, y_center), 8, (0, 255, 0), thickness=1)
    cv2.circle(frame, (x_center, y_center), 6, (0, 255, 0), thickness=1)
    cv2.circle(frame, (x_center, y_center), 4, (0, 255, 0), thickness=1)

    cv2.line(frame, (x_center - 6, y_center),
             (x_center + 6, y_center), (0, 255, 0), thickness=1)
    cv2.line(frame, (x_center, y_center - 6),
             (x_center, y_center + 6), (0, 255, 0), thickness=1)


    return obj_id, name, time, x1, y1, x2, y2


def choose_object_4_shelling(dataframe, frame):
    if len(dataframe) == 0:
        print("DataFrame is empty")
        return


    min_idx = dataframe["obj_id"].idxmin()
    t_data = dataframe.loc[min_idx]

    required_columns = ['obj_id', 'name', 'time', 'x1', 'y1', 'x2', 'y2']
    if not set(required_columns).issubset(t_data.index):
        print("Missing required columns in the data")
        return

    obj_id, name, time, x1, y1, x2, y2 = t_data[['obj_id', 'name', 'time', 'x1', 'y1', 'x2', 'y2']].tolist()

    # x_center = int((x1 + x2) / 2)
    # y_center = int((y1 + y2) / 2)

    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #
    #
    # cv2.circle(frame, (x_center, y_center), 8, (0, 255, 0), thickness=1)
    # cv2.circle(frame, (x_center, y_center), 6, (0, 255, 0), thickness=1)
    # cv2.circle(frame, (x_center, y_center), 4, (0, 255, 0), thickness=1)
    #
    # cv2.line(frame, (x_center - 6, y_center),
    #          (x_center + 6, y_center), (0, 255, 0), thickness=1)
    # cv2.line(frame, (x_center, y_center - 6),
    #          (x_center, y_center + 6), (0, 255, 0), thickness=1)


    return obj_id, name, time, x1, y1, x2, y2


def choose_area():
    ...





def vehicles_window(apc_ammount, tank_ammount, ifv_ammount, total_vehicles_ammount, descision):
    text_window = np.zeros((400, 500, 3), dtype=np.uint8)

    cv2.putText(text_window, "Assault vehicles info:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Total Vehicles: {total_vehicles_ammount}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Amount of APC: {apc_ammount}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(text_window, f"Amount of Tanks: {tank_ammount}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(text_window, f"Amount of IFV: {ifv_ammount}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.putText(text_window, f"Descision: {descision}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                2,
                cv2.LINE_AA)

    return text_window



def special_window(ev_ammount, aa_ammount, total_specials_ammount, descision):
    text_window = np.zeros((400, 400, 3), dtype=np.uint8)

    cv2.putText(text_window, "Special vehicles info", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Total Special vehicles: {total_specials_ammount}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Amount of EV: {ev_ammount}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.putText(text_window, f"Amount of AA systems: {aa_ammount}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Descision: {descision}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    return text_window



def aviation_window(aap_ammount, ah_ammount, th_ammount, ta_ammount, total_flying_units, descision):
    text_window = np.zeros((400, 450, 3), dtype=np.uint8)

    cv2.putText(text_window, "Aviation info", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Total flying units: {total_flying_units}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Attack planes: {aap_ammount}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.putText(text_window, f"Attack helicopters: {ah_ammount}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Transport helicopter: {th_ammount}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(text_window, f"Transport plane: {ta_ammount}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.putText(text_window, f"Descision: {descision}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    return text_window


def artillery_window(tart_ammount, spart_ammount, total_artillery_units, descision):
    text_window = np.zeros((400, 400, 3), dtype=np.uint8)

    cv2.putText(text_window, "Artillery info", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Total artillery units: {total_artillery_units}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Towed artillery: {tart_ammount}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.putText(text_window, f"Self propelled artillery: {spart_ammount}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)


    cv2.putText(text_window, f"Descision: {descision}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    return text_window


def troops_window(total_troops, descision):
    text_window = np.zeros((200, 500, 3), dtype=np.uint8)

    cv2.putText(text_window, "Infantry info", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(text_window, f"Total troops: {total_troops}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(text_window, f"Descision: {descision}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)

    return text_window


def troops_data(model, frame, threshold):
    results = model.predict(frame, imgsz=736, conf=threshold, classes=[0])[0]

    if not results.boxes.data.tolist():
        return None

    data_list = []
    for result in results.boxes.data.tolist():
        data_list.append(result)

    return data_list


def detection(x1, y1, x2, y2, score, class_id, names, objects_dict, next_object_id, objects, frame, current_frame_objects):
            center = get_center(x1, y1, x2, y2)
            time = time_of_detection()
            name = names[int(class_id)]

            obj_id = find_closest_object(center, objects_dict)

            if obj_id is None:
                print("None")
                obj_id = next_object_id
                next_object_id += 1

            current_frame_objects[obj_id] = center

            new_row = pd.DataFrame({
                "obj_id": [obj_id],
                "name": [name],
                "time": [time],
                "x1": [x1],
                "y1": [y1],
                "x2": [x2],
                "y2": [y2],
                "score": [score]
            })
            objects = pd.concat([objects, new_row], ignore_index=True)


            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{names[int(class_id)].upper()} {obj_id}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (255, 0, 0), 3, cv2.LINE_AA)

            # if 0 < ammount <= 1:
            #     cv2.circle(frame, (center[0], center[1]), 8, (255, 0, 0), thickness=1)
            #     cv2.circle(frame, (center[0], center[1]), 6, (255, 0, 0), thickness=1)
            #     cv2.circle(frame, (center[0], center[1]), 4, (255, 0, 0), thickness=1)
            #
            #     cv2.line(frame, (center[0] - 6, center[1]),
            #              (center[0] + 6, center[1]), (255, 0, 0), thickness=1)
            #     cv2.line(frame, (center[0], center[1] - 6),
            #              (center[0], center[1] + 6), (255, 0, 0), thickness=1)



            return next_object_id, objects, current_frame_objects


def encoder(*amounts):
    def encode_amount(amount):
        if 1 <= amount < 3:
            return 1
        elif 4 <= amount < 6:
            return 2
        elif amount >= 6:
            return 3
        else:
            return 0

    return tuple(encode_amount(amount) for amount in amounts)


def troops_encoder(amount):
    if 1 <= amount < 5:
        return 1
    elif 5 <= amount < 10:
         return 2
    elif amount >= 10:
        return 3
    else:
        return 0

# how many objects in radius of specific point
def objects_in_radius(centers, point, radius):
    count = 0
    for x, y in centers:
        distance = np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if distance <= radius:
            count += 1
    return count


# find the best point in radius of which are the most objects
def find_best_point(centers, radius):
    max_count = 0
    best_point = None

    for i in range(len(centers)):
        current_point = centers[i]

        count = objects_in_radius(centers, current_point, radius)

        if count > max_count:
            max_count = count
            best_point = current_point

    return best_point, max_count

# which type of targeting (depends on weapons) it will use
def classify_descisions(descision, dataframe, frame):
    if descision in ["ATGM", "FPV-drones", "Machine gun"]:
        obj_id, name, time, x1, y1, x2, y2 = choose_object(dataframe, frame)
        shells = False
        return shells,  obj_id, name, time, x1, y1, x2, y2

    elif descision in ["Clusster shells", "Unitar shells"]:
        obj_id, name, time, x1, y1, x2, y2 = choose_object_4_shelling(dataframe, frame)
        shells = True
        return shells, obj_id, name, time, x1, y1, x2, y2


    elif descision == "Rest of amunition":
        choose_object(dataframe, frame)
        return True




