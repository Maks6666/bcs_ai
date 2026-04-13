import math 

class Local2GPS:
    def __init__(self, lat, lon, heading):
        self.lat = lat
        self.lon = lon
        self.heading = math.radians(heading)
    
    def convert(self, X, Y):
        
        # here we convert (X, Y) local(!!!) coordinates into real world ones via formulas:

        # X_world ​= X ⋅ cos(θ) − Y ⋅ sin(θ) -> for X
        X_w = X * math.cos(self.heading) - Y * math.sin(self.heading)

        # Y_world ​= X ⋅ sin(θ) + Y ⋅ cos(θ) -> for Y 
        Y_w = X * math.sin(self.heading) + Y * math.cos(self.heading)

        # where θ - is a heading of camera 


        # put meters into GPS:

        dlat = Y_w / 111111
        dlon = X_w / (111111 * math.cos(math.radians(self.lat)))

        lat = self.lat + dlat
        lon = self.lon + dlon

        return lat, lon