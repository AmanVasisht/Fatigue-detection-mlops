import cv2


def get_video_source(source_type="camera", source_id=0):
    if source_type == "camera":
        return cv2.VideoCapture(source_id)
    else:
        raise ValueError("Only camera source is supported in live mode")
