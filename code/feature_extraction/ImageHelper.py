try:
    from PIL import Image
except ImportError:
    import Image

import os
import pytesseract


class ImageHelper:

    def __init__(self, data_path, tesseract_path=None):
        """
        :param data_path: Relative path to images directory.
        :param tesseract_path: Absolute path to the Tesseract-OCR installation directory.
        """

        # Set path to dataset directory
        self.data_path = os.path.expandvars(data_path)

        # Update Tesseract path if needed
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = os.path.expandvars(tesseract_path)

    def get_text(self, image_path):
        """
        Runs OCR on image.
        Input format: iterable with -one- element (as in the clickbait datasets).
        """

        # Check if post has media
        if not image_path or not image_path[0]:
            return ""

        # Get full image path
        image_path = os.path.join(self.data_path, image_path[0])

        # Load image
        img = Image.open(image_path)

        # Perform OCR
        return pytesseract.image_to_string(img)
