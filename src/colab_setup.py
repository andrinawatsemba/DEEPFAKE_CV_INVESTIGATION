"""One call that gets every notebook ready: mounts Drive, finds the shared folder, fixes paths."""

import os

REPO_URL = "https://github.com/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION.git"


def setup(local_root="./project_data"):
    """Returns (ROOT, REPO). ROOT = the shared data folder, REPO = this code folder."""
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    try:                                                    # inside Colab
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        from pull_data import find_drive_root
        root = find_drive_root()
        if root is None:
            raise SystemExit(
                "I cannot find the shared folder in your Drive.\n"
                "1) Open drive.google.com, find the folder DEEPFAKE_CV_DATA (under 'Shared with me').\n"
                "2) Right-click it > Organize > Add shortcut > My Drive.\n"
                "3) Run this cell again.")
    except ImportError:                                     # a normal laptop
        root = os.path.abspath(local_root)
        os.makedirs(root, exist_ok=True)
        print("Not in Colab: using a local folder.")
    print("Project data folder:", root)
    print("Code folder:        ", repo)
    return root, repo
