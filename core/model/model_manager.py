import os
import dlib

# Module-level cached models
_shape_predictor = None
_face_detector = None


def _get_model_path():
    """
    Returns absolute path to shape predictor model
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(
        base_dir,
        "assets",
        "models",
        "shape_predictor_68_face_landmarks.dat",
    )


def load_shape_predictor():
    global _shape_predictor
    if _shape_predictor is None:
        predictor_path = _get_model_path()
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Shape predictor model not found at {predictor_path}"
            )
        _shape_predictor = dlib.shape_predictor(predictor_path)
    return _shape_predictor


def load_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    return _face_detector


def get_shape_predictor_model():
    return load_shape_predictor()


def get_detector_model():
    return load_face_detector()
