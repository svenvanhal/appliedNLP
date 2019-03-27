try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os

# Make sure that Tesseract is defined even if it is not in $PATH variable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


class ImageHelper:
    @staticmethod
    def get_text(image_path):
        # Check if path is not a list from pandas
        if not isinstance(image_path, str):
            image_path = image_path[0]

        # RPath is parent of currentpath (so the appliedNLP folder)
        rpath = os.path.realpath("..")

        # Join paths
        # TODO: this is very ugly and surely can be done easier
        path = os.path.join(rpath, "data", "clickbait17-train-170331", image_path.split("/")[0],
                            image_path.split("/")[1])

        return pytesseract.image_to_string(Image.open(path))

