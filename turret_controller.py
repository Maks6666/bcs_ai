import socket
import json

import board
import busio
from adafruit_pca9685 import PCA9685

HOST = "0.0.0.0"
PORT = 5000

yaw_home = 90.0
pitch_home = 60.0

current_yaw = yaw_home
current_pitch = pitch_home

YAW_MIN = 0
YAW_MAX = 180

PITCH_MIN = 20
PITCH_MAX = 120

SERVO_FREQ = 50
PULSE_MIN_US = 500
PULSE_MAX_US = 2500

YAW_CHANNEL = 0
PITCH_CHANNEL = 15


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def update_turret(yaw_target, pitch_target):
    global current_yaw, current_pitch

    current_yaw = clamp(yaw_target, YAW_MIN, YAW_MAX)
    current_pitch = clamp(pitch_target, PITCH_MIN, PITCH_MAX)

    print(f"[RPI] updated -> yaw={current_yaw:.2f}, pitch={current_pitch:.2f}")


def set_angle(pca, channel, angle):
    angle = clamp(angle, 0, 180)
    period_us = 1_000_000 / SERVO_FREQ
    pulse_us = PULSE_MIN_US + (angle / 180.0) * (PULSE_MAX_US - PULSE_MIN_US)
    duty_cycle = int((pulse_us / period_us) * 65535)
    pca.channels[channel].duty_cycle = duty_cycle


def start_server(pca):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[RPI] Waiting for connection on {HOST}:{PORT}...")

    conn, addr = server.accept()
    print(f"[RPI] Connected by {addr}")

    buffer = ""

    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                print("[RPI] Client disconnected")
                break

            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                print("[RPI] raw:", line)

                try:
                    message = json.loads(line)

                    yaw_target = float(message.get("yaw_target", yaw_home))
                    pitch_target = float(message.get("pitch_target", pitch_home))

                    update_turret(yaw_target, pitch_target)

                    set_angle(pca, YAW_CHANNEL, current_yaw)
                    set_angle(pca, PITCH_CHANNEL, current_pitch)

                except Exception as e:
                    print("[RPI] JSON parse error:", e)


if __name__ == "__main__":
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = SERVO_FREQ

    set_angle(pca, YAW_CHANNEL, yaw_home)
    set_angle(pca, PITCH_CHANNEL, pitch_home)

    start_server(pca)