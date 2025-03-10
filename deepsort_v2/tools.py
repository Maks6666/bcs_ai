import torch
import cv2


def preprocess_img(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0

    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1).unsqueeze(0)

    return image
