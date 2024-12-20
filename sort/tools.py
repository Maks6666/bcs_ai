import os
import cv2
import torch

def get_path(dir_name):
    dir_list = os.listdir(dir_name)
    for i, file in enumerate(dir_list):
        print(f'{i + 1}: {file}')

    n = int(input("Choose file number: "))
    if 0 < n <= len(dir_list):
        file = dir_list[n - 1]
        path = os.path.join(dir_name, file)

        return path


def get_password():
    def_file = "password.txt"
    if os.path.isfile(def_file):
        with open(def_file, "r") as file:
            for line in file:
                data = line.split()
                return int(data[0])



def process_image(img):
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0

        image = torch.tensor(img, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        return image



