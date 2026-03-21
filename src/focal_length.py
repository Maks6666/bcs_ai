class FocalLength:
    def __init__(self):

        self.person_size = 1.8

        self.f_mm = 8.0
        self.sensor_width = 6.4
        self.frame_width = 1920

    def estimate(self, bbox):
        x1, y1, x2, y2 = bbox

        P = y2 - y1
        if P <= 0:
            return None
        
        W = self.person_size

        f_mm = self.f_mm
        sensor_width = self.sensor_width
        frame_width = self.frame_width

        f = (f_mm * frame_width) / sensor_width

        D = (W * f) / P

        return round(D, 2)



