import cv2


def get_video_source(source_type="camera", source_id=0):
    # If source_id is a string that looks like a number (from env vars), convert it
    if isinstance(source_id, str) and source_id.isdigit():
        source_id = int(source_id)
        
    cap = cv2.VideoCapture(source_id)
    
    # Optimization for network streams (EC2)
    if isinstance(source_id, str):
        # Reduces lag for video URLs
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
        
    return cap
