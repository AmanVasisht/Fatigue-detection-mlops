import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.camera import get_video_source
from core.utils.video_utils import encode_frame
from core.model.operation_manager import blink_process, yawness_process
import os

app = FastAPI()

# Move your variables here so the API can track them across frames
state = {
    "blink_start_time": None,
    "COUNTER": 0,
    "drowsy_blink_count": 0,
    "blink_threshold": 0.25,
    "blink_durations": [],
    "yawn_count": 0,
    "yawn_start_time": None,
    "yawn_end_time": None,
}

def generate_frames():
    # Load source from environment variable (Defaults to 0 for local)
    import os
    video_source = os.getenv("VIDEO_SOURCE", "0")
    
    cap = get_video_source(source_id=video_source)
    
    while True:
        success, frame = cap.read()
        if not success:
            # If it's a video file, restart it; if it's a camera, stop.
            if isinstance(video_source, str) and not video_source.isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Pass raw frame directly (No Base64 encoding here!)
        # 1. Blink Process
        state["COUNTER"], state["blink_start_time"], state["blink_durations"], frame, state["drowsy_blink_count"] = blink_process(
            frame, state["blink_start_time"], state["COUNTER"], 
            state["drowsy_blink_count"], state["blink_threshold"], state["blink_durations"]
        )

        # 2. Yawn Process
        state["yawn_count"], state["yawn_start_time"], state["yawn_end_time"], frame = yawness_process(
            frame, state["yawn_count"], state["yawn_start_time"], state["yawn_end_time"]
        )

        # 3. Only encode once for the Browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)