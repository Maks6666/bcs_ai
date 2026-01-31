class VehiclesCounter:
    def __init__(self):
        ...
    def count(self, vehicles: dict):
        tanks = 0
        apc = 0
        ifv = 0
        if len(vehicles) != 0:
            for key, _ in vehicles.items():
                if vehicles[key] == "TANK":
                    tanks += 1
                elif vehicles[key] == "APC":
                    apc += 1
                elif vehicles[key] == "IFV":
                    ifv += 1
        
        return tanks, ifv, apc
                