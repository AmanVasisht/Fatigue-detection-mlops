import base64
import cv2


def encode_frame(frame):
    _, img_encoded = cv2.imencode(".png", frame)
    base64_encoded_image = base64.b64encode(img_encoded)
    return base64_encoded_image.decode("utf-8")
