from core.preprocess.preprocess_utils import (
    detect_yawnssss,
    blink_detectionn,
)


def yawness_process(frame, yawn_count, yawn_start_time, yawn_end_time):
    return detect_yawnssss(frame, yawn_count, yawn_start_time, yawn_end_time)

def blink_process(frame, blink_start_time, COUNTER, drowsy_blink_count, blink_threshold, blink_durations):
    return blink_detectionn(frame, blink_start_time, COUNTER, drowsy_blink_count, blink_threshold, blink_durations)