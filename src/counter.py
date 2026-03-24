class Counter:
    def __init__(self):
        ...
    
    def count_statuses(self, statuses: dict):
        moving_forward = 0
        from_left_flank = 0
        from_right_flank = 0
        moving_back = 0

        if len(statuses) != 0:
            for key, _ in statuses.items():

                if statuses[key] == "moving_back":
                    moving_back += 1
                elif statuses[key] == "from_left_flank":
                    from_left_flank += 1
                elif statuses[key] == "from_right_flank":
                    from_right_flank += 1
                elif statuses[key] == "center_flank":
                    moving_forward += 1

        text = f"Moves back: {moving_back} | Moves from left flank: {from_left_flank} | Moves from right flank: {from_right_flank} | Moves from central flank: {moving_forward}"
        return text, (moving_forward, from_left_flank, from_right_flank, moving_back)
    
    def count_vehicles(self, vehicles: dict):
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
    

    def count_flanks(self, positions: dict, scale: int, map_size: int, flank_threshold: int, flank_position: dict):

        center = map_size // 2
        
        for idx, (X, _) in positions.items():
            cx = int(center + X * scale)

            left_borded = center - flank_threshold * scale
            right_border = center + flank_threshold * scale

            if cx < left_borded:
                flank_position['left_flank'].append(idx)
            
            elif left_borded < cx < right_border:
                flank_position['center'].append(idx)
            
            elif cx > right_border:
                flank_position['right_flank'].append(idx)



            

