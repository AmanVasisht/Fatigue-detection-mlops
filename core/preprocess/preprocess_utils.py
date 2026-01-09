import cv2
import numpy as np
import time
import base64
from scipy.spatial import distance as dist

from core.model.model_manager import (
    get_shape_predictor_model,
    get_detector_model,
)

# Lazy-loaded models (safe for Docker)
_predictor = None
_detector = None


def _load_models():
    global _predictor, _detector
    if _predictor is None or _detector is None:
        _predictor = get_shape_predictor_model()
        _detector = get_detector_model()
    return _predictor, _detector


EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 5
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

mouth_open_threshold = 0.3
YAWN_THRESHOLD = 0.5


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def blink_detectionn(
    base64_image,
    blink_start_time,
    COUNTER,
    drowsy_blink_count,
    blink_threshold,
    blink_durations,
):
    predictor, detector = _load_models()

    image_data = base64.b64decode(base64_image)
    image_np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        landmarks = np.array(
            [[p.x, p.y] for p in predictor(frame, rect).parts()]
        )

        left_eye = landmarks[LEFT_EYE_POINTS]
        right_eye = landmarks[RIGHT_EYE_POINTS]

        ear_left = eye_aspect_ratio(left_eye.reshape(-1, 2))
        ear_right = eye_aspect_ratio(right_eye.reshape(-1, 2))
        ear_avg = (ear_left + ear_right) / 2.0

        cv2.rectangle(
            frame,
            (left_eye[0][0], left_eye[0][1]),
            (left_eye[3][0], left_eye[3][1]),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            frame,
            (right_eye[0][0], right_eye[0][1]),
            (right_eye[3][0], right_eye[3][1]),
            (0, 255, 0),
            2,
        )

        if ear_avg < EYE_AR_THRESH:
            if blink_start_time is None:
                blink_start_time = time.time()
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                blink_end_time = time.time()
                blink_duration = blink_end_time - blink_start_time
                blink_durations.append(blink_duration)

                if blink_duration > blink_threshold:
                    drowsy_blink_count += 1
                    print(
                        f"Drowsy blink detected: {blink_duration:.2f}s",
                        drowsy_blink_count,
                    )

            COUNTER = 0
            blink_start_time = None

    cv2.putText(
        frame,
        f"Blink count: {drowsy_blink_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return COUNTER, blink_start_time, blink_durations, frame, drowsy_blink_count


def detect_yawnssss(base64_image, yawn_count, yawn_start_time, yawn_end_time):
    predictor, detector = _load_models()
    yawn_rect = None

    image_data = base64.b64decode(base64_image)
    image_np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        y_diff = landmarks.part(62).y - landmarks.part(66).y
        x_diff = landmarks.part(51).y - landmarks.part(57).y

        if x_diff != 0:
            mouth_openness = abs(y_diff) / abs(x_diff)

            if mouth_openness > mouth_open_threshold:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                yawn_end_time = time.time()

                point_48 = np.array(
                    [landmarks.part(48).x, landmarks.part(48).y]
                )
                point_54 = np.array(
                    [landmarks.part(54).x, landmarks.part(54).y]
                )

                x, y, w, h = cv2.boundingRect(
                    np.array([point_48, point_54])
                )
                yawn_rect = (x, y, w, h)

            elif yawn_start_time is not None:
                yawn_duration = yawn_end_time - yawn_start_time
                if yawn_duration >= YAWN_THRESHOLD:
                    yawn_count += 1
                    print(
                        f"Drowsy yawn detected: {yawn_duration:.2f}s",
                        yawn_count,
                    )
                yawn_start_time = None

    if yawn_rect is not None:
        x, y, w, h = yawn_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(
        frame,
        f"Yawn count: {yawn_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return yawn_count, yawn_start_time, yawn_end_time, frame
