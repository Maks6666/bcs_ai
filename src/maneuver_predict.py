import cv2 
import torch 
import numpy as np
from conv_lstm_model import model

class ManeuverPredictor:
    def __init__(self, tactics, device):
        self.tactic_names = ['moving_back', 'center_flank', 'from_left_flank', 'from_right_flank']
        self.tactics = tactics
        self.device = device
        
    def compute_dense_optical_flow(self, frames):
        assert frames.ndim == 4, "frames must be [T, H, W, 3]"
        assert frames.shape[-1] == 3, "frames must be RGB"

        flows = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for t in range(1, frames.shape[0]):
            curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            flows.append(flow)
            prev_gray = curr_gray

        return np.stack(flows, axis=0)
    
    def prediction(self, frames, idx) -> None:
        """
        frames: list of 16 RGB frames [H, W, 3]
        """

        #  stack -> numpy
        video = np.stack(frames, axis=0).astype(np.uint8)  # [16, H, W, 3]

        # optical flow
        flows = self.compute_dense_optical_flow(video)  # [15, H, W, 2]

        
        dx = flows[..., 0]
        dy = flows[..., 1]
        mag = np.sqrt(dx**2 + dy**2)

        flow_seq = np.stack([
            dx.mean(axis=(1, 2)),
            dy.mean(axis=(1, 2)),
            mag.mean(axis=(1, 2))
        ], axis=1)  # [15, 3]

        flow_seq = np.clip(flow_seq, -20, 20) / 20.0

        # torch
        x = torch.tensor(flow_seq, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)   # [1, 15, 3]

        pred = model.predict(x)

        self.tactics[idx] = self.tactic_names[pred]

    