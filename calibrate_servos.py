import board
import busio
from adafruit_pca9685 import PCA9685

# source ~/servo-env/bin/activate


# frequency of signal - so PCA sends impuls 50 times per second 
SERVO_FREQ = 50
# minimum lenght of impuls 
PULSE_MIN_US = 500
# maximum lenght of impuls 
PULSE_MAX_US = 2500

# channels on PCA 
YAW_CHANNEL = 0
PITCH_CHANNEL = 1

yaw = 90
pitch = 90

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def set_angle(pca, channel, angle):
    # prevent 'strange values' -> (-20 -> 0) / (200 -> 180)
    angle = clamp(angle, 0, 180)

    # to calculate one PWM (Pulse Width Modulation - one full cycle of signal) period
    period_us = 1_000_000 / SERVO_FREQ

    # this sting turns angle into impuls lenght
    pulse_us = PULSE_MIN_US + (angle / 180.0) * (PULSE_MAX_US - PULSE_MIN_US)

    duty_cycle = int((pulse_us / period_us) * 65535)
    pca.channels[channel].duty_cycle = duty_cycle

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQ

set_angle(pca, YAW_CHANNEL, yaw)
set_angle(pca, PITCH_CHANNEL, pitch)

print("Controls:")
print("a/d -> yaw left/right")
print("w/s -> pitch up/down")
print("q -> quit")

while True:
    print(f"\nyaw={yaw}, pitch={pitch}")
    cmd = input("cmd: ").strip().lower()

    if cmd == "a":
        yaw -= 1
    elif cmd == "d":
        yaw += 1
    elif cmd == "w":
        pitch += 1
    elif cmd == "s":
        pitch -= 1
    elif cmd == "q":
        break
    else:
        continue

    yaw = clamp(yaw, 0, 180)
    pitch = clamp(pitch, 0, 180)

    set_angle(pca, YAW_CHANNEL, yaw)
    set_angle(pca, PITCH_CHANNEL, pitch)

print(f"Final home: yaw_home={yaw}, pitch_home={pitch}")