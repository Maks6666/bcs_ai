import numpy as np

class ItemsEncoder:
    def __init__(self, weapons):
        self.weapons = weapons
    
    def encode(self, tanks, ifv, apc):
        tanks_value = 0
        ifv_value = 0
        apc_value = 0

        if 0 < tanks < 5:
            tanks_value = 1
        elif 5 <= tanks < 10:
            tanks_value = 2
        elif tanks >= 10:
            tanks_value = 3


        if 0 < ifv < 5:
            ifv_value = 1
        elif 5 <= ifv < 10:
            ifv_value = 2
        elif ifv >= 10:
            ifv_value = 3


        if 0 < apc < 5:
            apc_value = 1
        elif 5 <= apc < 10:
            apc_value = 2
        elif apc >= 10:
            apc_value = 3


        atgm = self.weapons["atgm"]
        if 0 < atgm < 10:
            atgm_value = 1
        elif 10 <= atgm < 30:
            atgm_value = 2
        elif atgm >= 30:
            atgm_value = 3
        elif atgm <= 0:
            atgm_value = 0

        cl_shells = self.weapons["cluster_shells"]
        if 0 < cl_shells < 10:
            cluster_shells_value = 1
        elif 10 <= cl_shells < 30:
            cluster_shells_value = 2
        elif cl_shells >= 30:
            cluster_shells_value = 3
        elif cl_shells <= 0:
            cluster_shells_value = 0

        u_shells = self.weapons["unitary_shells"]
        if 0 < u_shells < 10:
            unitar_shells_value = 1
        elif 10 <= u_shells < 30:
            unitar_shells_value = 2
        elif u_shells >= 30:
            unitar_shells_value = 3
        elif u_shells <= 0:
            unitar_shells_value = 0

        fpv = self.weapons["fpv_drones"]
        if 0 < fpv < 10:
            fpv_value = 1
        elif 10 <= fpv < 30:
            fpv_value = 2
        elif fpv >= 30:
            fpv_value = 3
        elif fpv <= 0:
            fpv_value = 0

        array = np.array([[tanks_value, ifv_value, apc_value, atgm_value, cluster_shells_value, unitar_shells_value, fpv_value]])

        return array