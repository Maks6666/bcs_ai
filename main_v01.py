# information display with open-cv


from ultralytics import YOLO
import cv2
import pandas as pd
from functions import total_coordinates_func, get_center, find_closest_object, time_of_detection, choose_object, vehicles_window, special_window, aviation_window, artillery_window, troops_window, troops_data, encoder, troops_encoder, detection, find_best_point, classify_descisions
from models_building import vehicles_model, special_model, aviation_model, artillery_model, troops_model
from interface import ground_weapons, ground_weapons_encoder





def main():

    global apc_ammount


    t_atgm, t_cl_shells, t_u_shells, t_fpv_drones, t_manpads, t_sam = ground_weapons()

    atgm, cl_shells, u_shells, fpv_drones, manpads, sam = ground_weapons_encoder(t_atgm, t_cl_shells, t_u_shells, t_fpv_drones, t_manpads, t_sam)


    vehicles_commands = ["ATGM", "Clusster shells", "Unitar shells", "FPV-drones", "Machine gun", "ATGM/Guns", "FPV/Guns", "Rest of amunition"]
    special_commaands = ["ATGM", "Clusster shells", "Unitar shells", "FPV-drones", "Machine gun"]
    flying_commands  = ["MANPADS", "SAM", "AAG", "Wait for reloading", "SAM/MANPADS", "SAM/AAG"]
    artillery_commands = ["Clusster shells", "Unitar shells", "FPV-drones", "Wait for reloading"]
    troops_commands = ["Clusster shells", "Unitar shells", "FPV-drones", "Machine gun"]


    vehicles = ["TANK", "IFV", "APC"]
    flying_units = ["AH", "TH", "AAP", "TA"]
    artillery = ["TART", "SPART"]
    specials = ["EV", "AA"]


    model = YOLO("yolo/best (16).pt")
    # vehicles_model = "models/vehicles.pt"
    names = model.names


    troops_detection = YOLO("yolov9c")
    yolo_names = troops_detection.names

    link = "Ukraine drone video shows attack on Russian tanks.mp4"
    # link = "videoplayback (1).mp4"
    # link = "Self-propelled artillery installation 2S3 'Akatsiya' 152 mm and MLRS 'Grad' fire.mp4"
    # link = ("South Korean KH-179 155mm Field Howitzer Artillery In Action.mp4")
    # link = "Holdfast - Cavalry charge vs. Infantry square.mp4"


    # link = "Cross of Iron - Russian Infantry Attack.mp4"
    video = f"videos/{link}"
    threshold = 0.3
    print(names)

    cap = cv2.VideoCapture(video)

    objects_dict = {}
    next_object_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = pd.DataFrame(columns=["obj_id", "name", "time", "x1", "y1", "x2", "y2", "score"])
        current_frame_objects = {}

        total_object_ammount = 0
        total_troops = 0
        total_troops_coordinates = []
        troops_descision = "Wait"

        total_vehicles_ammount = 0
        vehicles_total_coordinates = []
        apc_ammount = 0
        tank_ammount = 0
        ifv_ammount = 0
        vehicles_descision = "Wait"

        total_specials_ammount = 0
        specials_total_coordinates = []
        ev_ammount = 0
        aa_ammount = 0
        specials_descision = "Wait"

        total_flying_units = 0
        flying_total_coordinates = []
        ah_ammount = 0
        th_ammount = 0
        aap_ammount = 0
        ta_ammount = 0
        flying_descision = "Wait"

        total_artillery_units = 0
        total_artillery_coordinates = []
        tart_ammount = 0
        spart_ammount = 0
        artillery_descision = "Wait"

        # ------------------------------------------------------------------------------------------------------------------------------------------

        troops_centers = []

        troops_results = troops_data(troops_detection, frame, threshold)

        if troops_results is not None:
            for result in troops_results:
                x1, y1, x2, y2, score, class_id = result
                name = yolo_names[int(class_id)]
                if score > threshold:
                    center = get_center(x1, y1, x2, y2)
                    time = time_of_detection()

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

                    troops_centers.append(get_center(x1, y1, x2, y2))
                    if troops_centers:
                        radius = 50
                        best_point, count = find_best_point(troops_centers, radius)


                    total_troops += 1

                    total_troops_coordinates.append((x1, y1))
                    total_troops_coordinates.append((x2, y2))


                    troops = troops_encoder(total_troops)

                    data = [troops, cl_shells, u_shells, fpv_drones]
                    result = troops_model(data)
                    troops_descision = troops_commands[result]

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    cv2.putText(frame, f"{name.upper()} {obj_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                (255, 0, 0), 3, cv2.LINE_AA)

        # ------------------------------------------------------------------------------------------------------------------------------------------
        results = model.predict(frame, imgsz=1024, conf=threshold)[0]

        vehicles_centers = []
        artillery_centers = []
        specials_cenetrs = []

        for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                name = names[int(class_id)]

                if score > threshold and name in vehicles:

                    next_object_id, objects, current_frame_objects = detection(
                        x1, y1, x2, y2, score, class_id, names, objects_dict, next_object_id, objects, frame, current_frame_objects
                    )

                    vehicles_centers.append(get_center(x1, y1, x2, y2))
                    if vehicles_centers:
                        radius = 50
                        best_point, count = find_best_point(vehicles_centers, radius)



                    total_object_ammount += 1
                    total_vehicles_ammount += 1
                    vehicles_total_coordinates.append((x1, y1))
                    vehicles_total_coordinates.append((x2, y2))


                    if name == "APC":
                        apc_ammount += 1
                    elif name == "TANK":
                        tank_ammount += 1
                    elif name == "IFV":
                        ifv_ammount += 1

                    apc, tanks, ifv = encoder(apc_ammount, tank_ammount, ifv_ammount)
                    data = [apc, tanks, ifv, atgm, cl_shells, u_shells, fpv_drones]
                    result = vehicles_model(data)
                    vehicles_descision = vehicles_commands[result]

                # ------------------------------------------------------------------------------------------------------------------------------------------

                if score > threshold and name in specials:

                    next_object_id, objects, current_frame_objects = detection(
                        x1, y1, x2, y2, score, class_id, names, objects_dict, next_object_id, objects, frame, current_frame_objects
                    )

                    specials_cenetrs.append(get_center(x1, y1, x2, y2))
                    if specials_cenetrs:
                        radius = 50
                        best_point, count = find_best_point(specials_cenetrs, radius)


                    total_object_ammount += 1
                    total_specials_ammount += 1
                    specials_total_coordinates.append((x1, y1))
                    specials_total_coordinates.append((x2, y2))



                    if name == "EV":
                        ev_ammount += 1
                    elif name == "AA":
                        aa_ammount += 1

                    ev, aa = encoder(ev_ammount, aa_ammount)

                    data = [ev, aa, atgm, cl_shells, u_shells, fpv_drones]
                    result = special_model(data)
                    specials_descision = special_commaands[result]


            # ------------------------------------------------------------------------------------------------------------------------------------------

                if score > threshold and name in flying_units:

                    next_object_id, objects, current_frame_objects = detection(
                        x1, y1, x2, y2, score, class_id, names, objects_dict, next_object_id, objects, frame, current_frame_objects
                    )

                    total_object_ammount += 1
                    total_flying_units += 1

                    flying_total_coordinates.append((x1, y1))
                    flying_total_coordinates.append((x2, y2))

                    if name == "AH":
                        ah_ammount += 1
                    elif name == "TH":
                        th_ammount += 1
                    elif name == "AAP":
                        aap_ammount += 1
                    elif name == "TA":
                        ta_ammount += 1

                    aap, ah, th, ta = encoder(aap_ammount, ah_ammount, th_ammount, ta_ammount)

                    data = [aap, ah, th, ta, manpads, sam]
                    result = aviation_model(data)
                    flying_descision = flying_commands[result]
            # ------------------------------------------------------------------------------------------------------------------------------------------

                if score > threshold and name in artillery:

                    next_object_id, objects, current_frame_objects = detection(
                        x1, y1, x2, y2, score, class_id, names, objects_dict, next_object_id, objects, frame, current_frame_objects
                    )

                    artillery_centers.append(get_center(x1, y1, x2, y2))
                    if artillery_centers:
                        radius = 450
                        best_point, count = find_best_point(artillery_centers, radius)

                    total_object_ammount += 1
                    total_artillery_units += 1


                    total_artillery_coordinates.append((x1, y1))
                    total_artillery_coordinates.append((x2, y2))


                    if name == "TART":
                        tart_ammount += 1
                    elif name == "SPART":
                        spart_ammount += 1


                    tart, spart = encoder(tart_ammount, spart_ammount)

                    data = [tart, spart, cl_shells, u_shells, fpv_drones]
                    result = artillery_model(data)
                    artillery_descision = artillery_commands[result]




        if total_specials_ammount > 1:
            shells = classify_descisions(specials_descision, objects, frame)
            total_coordinates_func(specials_total_coordinates, frame, (0, 0, 0), best_point, shells=shells)

        shells = classify_descisions(vehicles_descision, objects, frame)
        total_coordinates_func(vehicles_total_coordinates, frame, (0, 0, 255), best_point, shells=shells)

        if total_flying_units > 1:
            total_coordinates_func(flying_total_coordinates, frame, (0, 0, 255), None, shells=False)
            choose_object(objects, frame)

        if total_artillery_units > 1:
            total_coordinates_func(total_artillery_coordinates, frame, (0, 0, 255), best_point, shells=True)


        if total_troops > 1:
            shells = classify_descisions(troops_descision, objects, frame)
            total_coordinates_func(total_troops_coordinates, frame, (0, 0, 255),  best_point, shells=shells)
            if shells is False:
                choose_object(objects, frame)





        cv2.putText(frame, f"Amount of objects: {total_object_ammount}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 0, 0), 3, cv2.LINE_AA)




        objects_dict = current_frame_objects


        text_window_1 = vehicles_window(apc_ammount, tank_ammount, ifv_ammount, total_vehicles_ammount, vehicles_descision)
        text_window_2 = special_window(ev_ammount, aa_ammount, total_specials_ammount, specials_descision)
        text_window_3 = aviation_window(aap_ammount, ah_ammount, th_ammount, ta_ammount, total_flying_units, flying_descision)
        text_window_4 = artillery_window(tart_ammount, spart_ammount, total_artillery_units, artillery_descision)
        text_window_5 = troops_window(total_troops, troops_descision)









        cv2.imshow("Video", frame)
        cv2.imshow("Assault vehicles", text_window_1)
        cv2.imshow("Special vehicles", text_window_2)
        cv2.imshow("Aviation", text_window_3)
        cv2.imshow("Artillery", text_window_4)
        cv2.imshow("Troops", text_window_5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(objects)
            break

    cap.release()
    cv2.destroyAllWindows()

    print(names)
    print(objects_dict)





main()
