from weapon_model.weapon_model import weapon_model


class CommandPredictor:
    def __init__(self, commands):
        self.commands = commands
    def predict_command(self, array, vehicles: dict):
        if len(vehicles) > 0:
            res = weapon_model.predict(array)
            command = self.commands[int(res)]
            return command
        else:
            return None
