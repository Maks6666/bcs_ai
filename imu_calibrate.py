import time
import math
import requests

CAMERA_IP = "192.168.1.142"  
URL = f"http://{CAMERA_IP}/imu"
SAMPLES = 30
DELAY = 0.1  


def main():
    print(f"Снимаю {SAMPLES} показаний с {URL}...")
    print("Держи камеру неподвижно!")

    ax_list, ay_list, az_list = [], [], []

    for i in range(SAMPLES):
        try:
            resp = requests.get(URL, timeout=1.0)
            data = resp.json()
            if data.get("ok"):
                a = data["accel"]
                ax_list.append(a["x"])
                ay_list.append(a["y"])
                az_list.append(a["z"])
        except Exception as e:
            print(f"Ошибка запроса: {e}")
        time.sleep(DELAY)

    if not ax_list:
        print("Не удалось получить ни одного показания.")
        return

    ax_avg = sum(ax_list) / len(ax_list)
    ay_avg = sum(ay_list) / len(ay_list)
    az_avg = sum(az_list) / len(az_list)

    magnitude = math.sqrt(ax_avg**2 + ay_avg**2 + az_avg**2)

    print("\n--- Результат ---")
    print(f"accel.x (среднее): {ax_avg:.3f}")
    print(f"accel.y (среднее): {ay_avg:.3f}")
    print(f"accel.z (среднее): {az_avg:.3f}")
    print(f"Величина вектора: {magnitude:.3f}g  (в покое должно быть ~1.0)")

    dominant = max(("x", abs(ax_avg)), ("y", abs(ay_avg)), ("z", abs(az_avg)), key=lambda p: p[1])
    print(f"\nДоминирующая ось (ближе всего к гравитации сейчас): {dominant[0]}")


if __name__ == "__main__":
    main()