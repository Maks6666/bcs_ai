class Position:
    def __init__(self):
        ...
    
    def predict_position(self, idx, t=2):
        if idx not in self.positions or idx not in self.velocities:
            return None 
        
        X, Y = self.positions[idx]
        v_x, v_y, _ = self.velocities[idx]

        X_f = X + v_x * t
        Y_f = Y + v_y * t

        return (X_f, Y_f)