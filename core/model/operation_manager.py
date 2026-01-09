from core.preprocess.preprocess_utils import (
    detect_yawnssss,
    blink_detectionn,
)


def yawness_process(base64_image, yawn_count, yawn_start_time, yawn_end_time):
    try:
        yawn_count, yawn_start_time, yawn_end_time, frame = detect_yawnssss(
            base64_image,
            yawn_count,
            yawn_start_time,
            yawn_end_time,
        )
        return yawn_count, yawn_start_time, yawn_end_time, frame

    except Exception as e:
        raise e


def blink_process(
    base64_image,
    blink_start_time,
    COUNTER,
    drowsy_blink_count,
    blink_threshold,
    blink_durations,
):
    try:
        (
            COUNTER,
            blink_start_time,
            blink_durations,
            frame,
            drowsy_blink_count,
        ) = blink_detectionn(
            base64_image,
            blink_start_time,
            COUNTER,
            drowsy_blink_count,
            blink_threshold,
            blink_durations,
        )

        return COUNTER, blink_start_time, blink_durations, frame, drowsy_blink_count

    except Exception as e:
        raise e
