import time

class WeaponCounter:
    def __init__(self, weapons: dict, interval: int = 10):
        self.weapons = weapons
        self.interval = interval
        self.last_used = {}

        self.command_map = {
            "ATGM": "atgm",
            "Cluster shells": "cluster_shells",
            "Unitary shells": "unitary_shells",
            "FPV-drones": "fpv_drones"
        }

    def fire(self, command: str) -> bool:
        now = time.time()
        last = self.last_used.get(command, 0)

        if command not in self.command_map:
            return False
        
        if now - last < self.interval:
            return False

        key = self.command_map[command]

        if self.weapons[key] <= 0:
            return False
        
        self.weapons[key] -= 1
        self.last_used[command] = now
        return True
    

