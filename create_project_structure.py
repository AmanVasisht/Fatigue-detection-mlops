import os

PROJECT_NAME = "fatigue-detection"

STRUCTURE = {
    "app": {
        "files": ["__init__.py", "main.py", "camera.py", "inference.py"],
        "folders": {}
    },

    "core": {
        "files": ["__init__.py"],
        "folders": {
            "model": {
                "files": ["__init__.py", "face_landmarks.py", "drowsiness_model.py"],
                "folders": {}
            },
            "preprocess": {
                "files": ["__init__.py", "preprocess_utils.py"],
                "folders": {}
            },
            "utils": {
                "files": ["__init__.py", "video_utils.py", "draw_utils.py"],
                "folders": {}
            }
        }
    },

    "assets": {
        "files": [],
        "folders": {
            "models": {"files": [], "folders": {}},
            "sample_videos": {"files": [], "folders": {}}
        }
    },

    "configs": {
        "files": ["config.yaml"],
        "folders": {}
    },

    "scripts": {
        "files": ["run_video.py", "run_camera.py"],
        "folders": {}
    },

    "tests": {
        "files": ["test_inference.py"],
        "folders": {}
    },

    "docker": {
        "files": ["Dockerfile"],
        "folders": {}
    }
}

ROOT_FILES = [
    ".gitignore",
    "requirements.txt",
    "README.md",
    "LICENSE"
]

def create_tree(base_path, tree):
    for folder_name, content in tree.items():
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Create files
        for file_name in content["files"]:
            open(os.path.join(folder_path, file_name), "a").close()

        # Create subfolders recursively
        create_tree(folder_path, content["folders"])

def main():
    project_path = os.path.join(os.getcwd(), PROJECT_NAME)
    os.makedirs(project_path, exist_ok=True)

    create_tree(project_path, STRUCTURE)

    for file in ROOT_FILES:
        open(os.path.join(project_path, file), "a").close()

    print(f"\n✅ Professional project structure created successfully at:\n{project_path}\n")

if __name__ == "__main__":
    main()
