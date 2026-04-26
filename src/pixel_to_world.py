import math 


class Pixel2World:
    def __init__(self, fov_horizontal, fov_vertical):
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
    
    def calculcate(self, cx, cy, frame_w, frame_h, distance):
        center_x = frame_w / 2
        center_y = frame_h / 2

        # turns hfov / vfov into radians 
        hfov = math.radians(self.fov_horizontal)
        vfov = math.radians(self.fov_vertical)

        # calculates focal length in pixels

        # fx, fy = parameters that allow you to convert pixel coordinates into 
        # the direction of the ray from the camera

        # fx, fy - parameters of camera 
        fx = frame_w / (2 * math.tan(hfov / 2))
        fy = frame_h / (2 * math.tan(vfov / 2))

        # calculates direction vector in camera space
        # this is a ray from the camera center through the pixel (cx, cy)
        x = (cx - center_x) / fx
        y = (cy - center_y) / fy
        # z always points forward in camera space, so we can set it to 1
        z = 1.0

        # normalize the direction vector
        norm = math.sqrt(x*x + y*y + z*z)

        x /= norm
        y /= norm
        z /= norm

        # scale the direction vector by the distance to get world coordinates
        X = distance * x
        Y = distance * y
        Z = distance * z

        return X, Y, Z