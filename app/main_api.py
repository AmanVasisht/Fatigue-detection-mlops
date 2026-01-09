import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from tempfile import NamedTemporaryFile
import uvicorn

# We still need these imports from your project structure
from core.model.operation_manager import blink_process, yawness_process

app = FastAPI()

def get_initial_state():
    """Returns a fresh state for every new video upload"""
    return {
        "blink_start_time": None, 
        "COUNTER": 0, 
        "drowsy_blink_count": 0,
        "blink_threshold": 0.25, 
        "blink_durations": [], 
        "yawn_count": 0,
        "yawn_start_time": None, 
        "yawn_end_time": None,
    }

@app.get("/")
def home():
    content = """
    <html>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h2>Drowsiness Detection System</h2>
            <p>Upload a video file (.mp4, .avi) to start detection</p>
            <form action="/upload_video" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="video/*" required>
                <br><br>
                <button type="submit" style="padding: 10px 20px;">Upload and Predict</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

def process_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get a fresh state for THIS specific video
    current_state = get_initial_state()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. Blink Process
        current_state["COUNTER"], current_state["blink_start_time"], current_state["blink_durations"], frame, current_state["drowsy_blink_count"] = blink_process(
            frame, current_state["blink_start_time"], current_state["COUNTER"], 
            current_state["drowsy_blink_count"], current_state["blink_threshold"], current_state["blink_durations"]
        )

        # 2. Yawn Process
        current_state["yawn_count"], current_state["yawn_start_time"], current_state["yawn_end_time"], frame = yawness_process(
            frame, current_state["yawn_count"], current_state["yawn_start_time"], current_state["yawn_end_time"]
        )

        # 3. Encode for browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    # Create temp file
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    contents = await file.read()
    with open(temp.name, "wb") as f:
        f.write(contents)
    
    return StreamingResponse(process_uploaded_video(temp.name), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ADD THIS so you can run 'python main_api.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)