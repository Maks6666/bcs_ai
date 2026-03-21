import cv2
import numpy as np
import torch 

class OpticalFlow:
    def __init__(self):
        ...
    
    def compute_dense_optical_flow(self, frames):
        flows = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for t in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
            )
            
            flows.append(flow)
            prev_gray = curr_gray
        
        return np.stack(flows, axis=0)
    
    def extract_flow_features(self, flows):
        dx = flows[..., 0]
        dy = flows[..., 1]
        mag = np.sqrt(dx**2 + dy**2)

        T, H, W = dx.shape

        features = []

        for t in range(T):
            frame_features = []

            for i in range(2):
                for j in range(2):

                    y0 = i * H // 2
                    y1 = (i+1) * H // 2
                    x0 = j * W // 2
                    x1 = (j+1) * W // 2

                    dx_patch = dx[t,y0:y1,x0:x1]
                    dy_patch = dy[t,y0:y1,x0:x1]
                    mag_patch = mag[t,y0:y1,x0:x1]

                    frame_features.extend([
                        dx_patch.mean(),
                        dy_patch.mean(),
                        mag_patch.mean()
                    ])
            features.append(frame_features)

        flow_seq = np.array(features)

        flow_seq = np.clip(flow_seq, -20, 20) / 20.0

        return torch.tensor(flow_seq, dtype=torch.float32)
