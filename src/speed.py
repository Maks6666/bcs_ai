import time 
import math 

class Velocity:
    def __init__(self, prev_positions, prev_time, velocities):
        self.prev_positions = prev_positions
        self.prev_time = prev_time
        self.velocities = velocities
    

    def calculate(self, idx, X, Z):
        t_now = time.time()

        # for the first itteration 
        if idx not in self.prev_positions:
            self.prev_positions[idx] = (X, Z)
            self.prev_time[idx] = t_now
            self.velocities[idx] = (0.0, 0.0, 0.0)
            return (0.0, 0.0, 0.0)

        X_prev, Z_prev = self.prev_positions[idx]
        t_prev = self.prev_time[idx]

        d_t = t_now - t_prev
        
        if d_t < 0.1:
            return self.velocities.get(idx, (0.0, 0.0, 0.0))

        v_x = (X - X_prev) / d_t
        v_z = (Z - Z_prev) / d_t
        speed = math.sqrt(v_x ** 2 + v_z ** 2)

        vx_old, vz_old, speed_old = self.velocities.get(idx, (0.0, 0.0, 0.0))


        alpha = 0.5
        v_x = alpha * v_x + (1 - alpha) * vx_old
        v_z = alpha * v_z + (1 - alpha) * vz_old
        speed = alpha * speed + (1 - alpha) * speed_old

        if speed < 0.1:
            speed *= 0.8

        if speed > 10:
            speed = speed_old * 0.7


        self.prev_positions[idx] = (X, Z)
        self.prev_time[idx] = t_now
        self.velocities[idx] = (v_x, v_z, speed)

        return v_x, v_z, speed