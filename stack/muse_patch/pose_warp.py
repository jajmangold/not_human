"""Motion-compensated interpolation for spatial AdvancedLivePortrait poses."""

import cv2
import numpy as np


def backward_pose_flow(base_frame, target_frame):
    """Return target-to-base flow suitable for inverse image sampling."""
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        target_gray, base_gray, None,
        0.5, 4, 21, 5, 7, 1.5, 0,
    )


def warp_with_pose_flow(frame, flow, amount):
    """Move one coherent frame along pose flow without alpha ghosting."""
    amount = min(1.0, max(0.0, float(amount)))
    if amount <= 0:
        return frame
    height, width = frame.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(
        frame,
        grid_x + flow[..., 0] * amount,
        grid_y + flow[..., 1] * amount,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
