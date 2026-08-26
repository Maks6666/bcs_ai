import numpy as np
import cv2


class ThreatField:
    def __init__(self, map_size: int, scale: int, center: int):
        self.map_size = map_size
        self.scale = scale
        self.center = center

        self.current_field = np.zeros((map_size, map_size), dtype=np.float32)
        self.future_field = np.zeros((map_size, map_size), dtype=np.float32)

        # Precomputed coordinate grids — not recreated every frame
        self._xs = np.arange(map_size, dtype=np.float32)
        self._ys = np.arange(map_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, positions: dict, threat_scores: dict, velocities: dict = None, predict_t: float = 5.0):
        """Recalculate both fields. Call once per frame."""
        self.current_field[:] = 0.0
        self.future_field[:] = 0.0

        for idx, (X, _, Z) in positions.items():
            score = threat_scores.get(idx, 0.0)
            if score < 0.05:
                continue

            px, py = self.world_to_pixel(X, Z)
            sigma = self.sigma(score)
            self.add_gaussian(self.current_field, px, py, sigma, score)

            if velocities and idx in velocities:
                vx, vz, _ = velocities[idx]
                fx = X + vx * predict_t
                fz = Z + vz * predict_t
                fpx, fpy = self.world_to_pixel(fx, fz)
                fsigma = sigma * (1.0 + predict_t * 0.05)           # spread grows over time
                fscore = score * max(0.3, 1.0 - predict_t * 0.08)   # confidence fades over time
                self.add_gaussian(self.future_field, fpx, fpy, fsigma, fscore)

        self.normalize(self.current_field)
        self.normalize(self.future_field)

    def draw(self, map_img: np.ndarray,
             show_future: bool = True,
             current_alpha: float = 0.4,
             future_alpha: float = 0.25,
             contour_levels: tuple = (0.4, 0.7, 0.9)) -> np.ndarray:
        """Overlay fields onto the map. Returns modified map_img."""
        if self.current_field.max() < 0.01:
            return map_img

        map_img = self.draw_overlay(map_img, self.current_field, current_alpha, future=False)
        map_img = self.draw_contours(map_img, self.current_field, contour_levels)

        if show_future and self.future_field.max() > 0.01:
            map_img = self.draw_overlay(map_img, self.future_field, future_alpha, future=True)

        return map_img

    def danger_at(self, X: float, Z: float, use_future: bool = False) -> float:
        """Threat level at world point (X, Z). Useful for external queries."""
        px, py = self.world_to_pixel(X, Z)
        px = np.clip(px, 0, self.map_size - 1)
        py = np.clip(py, 0, self.map_size - 1)
        field = self.future_field if use_future else self.current_field
        return float(field[py, px])

    def hotspot(self, use_future: bool = False):
        """Pixel coordinates of the field maximum."""
        field = self.future_field if use_future else self.current_field
        idx = np.argmax(field)
        return (int(idx % self.map_size), int(idx // self.map_size))





        

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def world_to_pixel(self, X: float, Z: float):
        px = int(self.center + X * self.scale)
        py = int(self.center - Z * self.scale)
        return px, py

    def sigma(self, score: float) -> float:
        return 20.0 + score * 30.0

    def add_gaussian(self, field: np.ndarray, px: int, py: int, sigma: float, score: float):
        gx = np.exp(-((self._xs - px) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((self._ys - py) ** 2) / (2 * sigma ** 2))
        field += score * np.outer(gy, gx)

    def normalize(self, field: np.ndarray):
        m = field.max()
        if m > 0:
            field /= m

    def draw_overlay(self, map_img: np.ndarray, field: np.ndarray,
                      alpha: float, future: bool) -> np.ndarray:
        field_u8 = (field * 255).astype(np.uint8)

        if future:
            heatmap = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            heatmap[:, :, 0] = field_u8
            heatmap[:, :, 2] = (field_u8 * 0.3).astype(np.uint8)
        else:
            heatmap = cv2.applyColorMap(field_u8, cv2.COLORMAP_JET)

        mask = field > 0.08
        overlay = np.zeros_like(heatmap)
        overlay[mask] = heatmap[mask]

        cv2.addWeighted(overlay, alpha, map_img, 1.0, 0, map_img)
        return map_img

    def draw_contours(self, map_img: np.ndarray, field: np.ndarray, levels: tuple) -> np.ndarray:
        colours = {
            levels[0]: (0, 200, 80),
            levels[1]: (0, 120, 255),
            levels[2]: (0, 0, 200),
        }
        field_u8 = (field * 255).astype(np.uint8)

        for level, colour in colours.items():
            _, binary = cv2.threshold(field_u8, int(level * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(map_img, contours, -1, colour, 1)

        return map_img