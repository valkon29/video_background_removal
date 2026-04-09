import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Optional


class BackgroundRemover:
    def __init__(self, model_selection: int = 1):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.selfie_segmentation.process(image_rgb)

        if results.segmentation_mask is not None:
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            return mask
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def remove_background(self, image: np.ndarray, background_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        mask = self.get_mask(image)

        background = np.full_like(image, background_color, dtype=np.uint8)

        result = np.where(mask[:, :, None] == 255, image, background)

        return result

    def blur_background(self, image: np.ndarray, blur_strength: int = 20) -> np.ndarray:
        mask = self.get_mask(image)

        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        mask_3ch = mask[:, :, None] / 255.0
        result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

        return result

    def replace_background_with_image(self, image: np.ndarray, bg_image: np.ndarray) -> np.ndarray:
        mask = self.get_mask(image)

        bg_resized = cv2.resize(bg_image, (image.shape[1], image.shape[0]))

        mask_3ch = mask[:, :, None] / 255.0
        result = (image * mask_3ch + bg_resized * (1 - mask_3ch)).astype(np.uint8)

        return result


class VideoProcessor:
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera resolution: {actual_width}x{actual_height} "
                  f"(requested: {resolution[0]}x{resolution[1]})")
        except Exception as e:
            print(f"Warning: Could not set resolution: {e}")
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Using default camera resolution: {actual_width}x{actual_height}")

        self.resolution = resolution
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.time()

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def calculate_fps(self) -> float:
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            self.frame_count = 0
            self.start_time = time.time()
            return fps

        return self.fps_history[-1] if self.fps_history else 0.0

    def get_average_fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def release(self):
        self.cap.release()


def create_top_bottom(original, processed):
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]

    if w1 != w2:
        scale = w1 / w2
        new_h = int(h2 * scale)
        processed = cv2.resize(processed, (w1, new_h))
        h2, w2 = processed.shape[:2]

    width = max(w1, w2)
    height = h1 + h2 + 20

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas[:h1, :w1] = original
    canvas[h1 + 20:h1 + 20 + h2, :w2] = processed

    cv2.line(canvas, (0, h1 + 10), (width, h1 + 10), (100, 100, 100), 2)

    return canvas