import cv2

def draw_segmentation_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
):
    """
    Draws a semi-transparent segmentation mask over a frame.
    
    Parameters:
        frame: numpy.ndarray (H W C, BGR)
        mask: numpy.ndarray (H W, boolean or binary)
        color: Tuple[int, int, int] (RGB, but OpenCV uses BGR natively if drawn directly. 
               Ensure alignment with how your box colors are defined.)
        alpha: float, transparency level
    """
    colored_overlay = np.zeros_like(frame, dtype=np.uint8)
    colored_overlay[mask > 0] = color
    
    # Blend the overlay with the original frame
    cv2.addWeighted(colored_overlay, alpha, frame, 1.0, 0, frame)