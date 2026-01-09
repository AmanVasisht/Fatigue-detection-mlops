from app.camera import get_video_source
from app.inference import run_inference


def main():
    cap = get_video_source(source_type="camera", source_id=0)
    run_inference(cap)


if __name__ == "__main__":
    main()
