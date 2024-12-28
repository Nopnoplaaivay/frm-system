import os

class CommonConsts:

    ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 2 * "../"))
    IMG_FOLDER = os.path.join(ROOT_FOLDER, "src\imgs")
    ticker_model = [
        "DXG",
        "TCH",
        "HDG",
        "HDC",
        "KDH",
        "IDC",
        "KBC",
        "DPG",
        "VHM",
        "NLG",
    ]
