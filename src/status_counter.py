class StatusCounter:
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
