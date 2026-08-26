import math
import time
import threading

import requests


class IMUReader(threading.Thread):
    """
    Читает сырые данные accel/gyro с ESP32-CAM (/imu) в фоновом потоке,
    вычисляет yaw и pitch и вызывает camera.update_from_imu().

    Важно про дрейф:
      - MEMS-гироскоп всегда имеет небольшое постоянное смещение (bias) —
        даже в полном покое gyro.x/y/z не равны точно 0. При интегрировании
        (yaw = сумма gyro.z * dt) эта ошибка накапливается линейно со временем,
        поэтому при старте выполняется калибровка bias: несколько секунд
        показаний в покое усредняются и вычитаются из всех дальнейших данных.
      - pitch считается из акселерометра (вектор гравитации) — стабилен,
        не дрейфует со временем (в отличие от чистого yaw).
      - yaw всё равно будет медленно "уплывать" в долгосрочной перспективе
        (нет магнитометра для абсолютной привязки) — калибровка bias убирает
        основной линейный дрейф, но не устраняет его полностью.

    Использование:
        imu = IMUReader(camera=cameras[0], host="192.168.1.104")
        imu.start()
    """

    def __init__(self, camera, host: str, interval: float = 0.1,
                 yaw_offset: float = None, gyro_z_sign: int = 1,
                 complementary_alpha: float = 0.98,
                 pitch_offset: float = 0.0,
                 calibration_samples: int = 100):
        super().__init__(daemon=True)
        self.camera = camera
        self.url = f"http://{host}/imu"
        self.interval = interval
        self._running = True

        self._yaw = yaw_offset if yaw_offset is not None else getattr(camera, "yaw_deg", 0.0)
        self._pitch = getattr(camera, "cam_pitch", 0.0)
        self._last_time = None

        self.gyro_z_sign = gyro_z_sign
        self.alpha = complementary_alpha
        self.pitch_offset = pitch_offset
        self.calibration_samples = calibration_samples

        # Bias гироскопа — заполняется при калибровке в начале run()
        self._gyro_bias = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _pitch_from_accel(self, ax, ay, az):
        raw = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return raw - self.pitch_offset

    def _calibrate_gyro_bias(self):
        """
        Снимает N показаний гироскопа (камера должна быть неподвижна!)
        и усредняет их — это и есть bias, который нужно вычитать из
        всех последующих измерений, чтобы устранить линейный дрейф.

        MEMS-гироскопы дрейфуют по bias при нагреве чипа после включения —
        калибровка сразу после power-on может "устареть" за несколько минут
        работы. Поэтому небольшая пауза перед калибровкой (дать датчику
        немного прогреться на рабочей температуре) и больше отсчётов дают
        более стабильный результат на протяжении сессии.
        """
        print(f"[IMU] Warming up sensor for {self.camera.camera_id}...")
        time.sleep(3.0)  # даём датчику немного прогреться перед калибровкой

        print(f"[IMU] Calibrating gyro bias for {self.camera.camera_id} "
              f"({self.calibration_samples} samples, держи камеру неподвижно)...")

        sx, sy, sz, n = 0.0, 0.0, 0.0, 0

        for _ in range(self.calibration_samples):
            try:
                response = requests.get(self.url, timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok", False):
                        gyro = data["gyro"]
                        sx += gyro["x"]
                        sy += gyro["y"]
                        sz += gyro["z"]
                        n += 1
            except Exception as e:
                print(f"[IMU] Calibration sample error: {e}")
            time.sleep(self.interval)

        if n > 0:
            self._gyro_bias = {"x": sx / n, "y": sy / n, "z": sz / n}
            print(f"[IMU] Gyro bias for {self.camera.camera_id}: {self._gyro_bias}")
        else:
            print(f"[IMU] Calibration failed for {self.camera.camera_id}, bias=0")

    def run(self):
        self._calibrate_gyro_bias()

        # Инициализируем pitch сразу из первого реального замера акселерометра,
        # а не из произвольного camera.cam_pitch — иначе комплементарный фильтр
        # (alpha=0.98) будет МЕДЛЕННО (несколько секунд) сходиться от стартового
        # значения к истинному, и pitch будет казаться "неправильным" первые
        # секунды работы, хотя на самом деле это просто ещё не сошедшийся фильтр.
        try:
            response = requests.get(self.url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok", False):
                    accel = data["accel"]
                    self._pitch = self._pitch_from_accel(accel["x"], accel["y"], accel["z"])
                    print(f"[IMU] Initial pitch for {self.camera.camera_id}: {self._pitch:.2f}")
        except Exception as e:
            print(f"[IMU] Could not get initial pitch: {e}")

        print(f"[IMU] Started for camera {self.camera.camera_id} -> {self.url}")
        while self._running:
            try:
                response = requests.get(self.url, timeout=1.0)
                if response.status_code == 200:
                    data = response.json()

                    if not data.get("ok", False):
                        print(f"[IMU] Sensor not ready for {self.camera.camera_id}")
                        time.sleep(self.interval)
                        continue

                    accel = data["accel"]
                    gyro = data["gyro"]

                    # Вычитаем bias, откалиброванный на старте
                    gyro_x = gyro["x"] - self._gyro_bias["x"]
                    gyro_y = gyro["y"] - self._gyro_bias["y"]
                    gyro_z = gyro["z"] - self._gyro_bias["z"]

                    now = time.time()
                    dt = (now - self._last_time) if self._last_time else self.interval
                    self._last_time = now

                    accel_pitch = self._pitch_from_accel(accel["x"], accel["y"], accel["z"])
                    self._pitch = (
                        self.alpha * (self._pitch + gyro_y * dt)
                        + (1 - self.alpha) * accel_pitch
                    )

                    self._yaw += self.gyro_z_sign * gyro_z * dt
                    self._yaw = self._yaw % 360.0

                    self.camera.update_from_imu(self._yaw, self._pitch)
                    print(f"[NEW YAW] -> {self._yaw}; [NEW PITCH] -> {self._pitch}")

            except requests.exceptions.Timeout:
                print(f"[IMU] Timeout {self.camera.camera_id} - waiting...")
            except requests.exceptions.ConnectionError:
                print(f"[IMU] No connection {self.camera.camera_id} - waiting...")
            except (KeyError, TypeError) as e:
                print(f"[IMU] Bad data format {self.camera.camera_id}: {e}")
            except Exception as e:
                print(f"[IMU] Error {self.camera.camera_id}: {e}")

            time.sleep(self.interval)

    def stop(self):
        self._running = False