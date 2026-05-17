import math

class Distance:
    def __init__(self, v_fov, cam_height, cam_pitch):
        self.v_fov = math.radians(v_fov)
        self.cam_height = cam_height         
        self.cam_pitch = math.radians(cam_pitch)  

    def estimate(self, bbox, frame_height):
        _, _, _, y2 = bbox

        y_foot = y2 

        # alpha - by how many degrees is the beam to the feet deflected downwards from the center of the frame.

        #           [ CAMERA ]
        #             /    |    
        #            /)alpha|     
        #. ray to   /      | optiсal axis
        #   feet   /       |     
        #         /        |            
        #        /         |            
        #       /__________|
        #    [FEET]      ( frame center
        #                  
        #      )


        alpha = ((y_foot - frame_height / 2) / frame_height) * self.v_fov


        # камера 
        #   | \ 
        #   |  \
        #   |   \
        #   |    \
        #   |)beta\
        #   |      \     
        #   |       \
        #   |        \
        #   |         \
        #   |          \
        #   |           \
        #   |            \
        #   |             \
        #   |              \
        #   |               \
        #   |                \
        #   | cam_height      \  ← ray to feet
        #   |                  \
        #   |___________________\
        #    ground             feet (object)
        #       ←      D      →


        beta = self.cam_pitch + alpha

        if beta <= 0:
            return None  # луч не пересекает плоскость земли

        D = self.cam_height / math.tan(beta)

        return round(D, 2)
