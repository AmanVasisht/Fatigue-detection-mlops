import cv2
from core.utils.video_utils import encode_frame
from core.model.operation_manager import blink_process, yawness_process


def run_inference(cap):
    if not cap.isOpened():
        raise RuntimeError("❌ Camera could not be opened")

    # Blink detection variables
    blink_start_time = None
    COUNTER = 0
    drowsy_blink_count = 0
    blink_threshold = 0.25
    blink_durations = []

    # Yawn detection variables
    yawn_count = 0
    yawn_start_time = None
    yawn_end_time = None

    print("✅ Live camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            break

        base64_image = encode_frame(frame)

        COUNTER, blink_start_time, blink_durations, frame_blink, drowsy_blink_count = blink_process(
            base64_image,
            blink_start_time,
            COUNTER,
            drowsy_blink_count,
            blink_threshold,
            blink_durations,
        )

        base64_image_blink = encode_frame(frame_blink)

        yawn_count, yawn_start_time, yawn_end_time, frame = yawness_process(
            base64_image_blink,
            yawn_count,
            yawn_start_time,
            yawn_end_time,
        )

        cv2.imshow("Drowsiness Detection (Live)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
