try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pathlib import Path

# Make sure that Tesseract is defined even if it is not in $PATH variable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


class ImageHelper:
    def __init__(self, data_path):
        project_path = Path(__file__).parent.parent
        data_path = (project_path / data_path)

        self.data_path = data_path

    def get_text(self, image_path):
        # Check if input is not a list from pandas
        if not isinstance(image_path, str):
            if len(image_path) == 0:
                image_path = ""
            elif len(image_path) == 1:
                image_path = image_path[0]
            else:
                # Article has multiple pictures: this should not occur
                image_path = ""

        # Return empty string if image path was undefined
        if image_path == "":
            return ""

        # Construct path and do OCR on the image
        file_path = (self.data_path / image_path).resolve()
        return pytesseract.image_to_string(Image.open(file_path))
