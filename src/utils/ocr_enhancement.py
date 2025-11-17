"""
Enhanced OCR preprocessing utilities for improving Tesseract accuracy
Includes image preprocessing, enhancement, and multi-config testing
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


class OCREnhancer:
    """
    Enhanced OCR preprocessing class with multiple enhancement strategies
    for improving text recognition accuracy on financial forms
    """

    # Tesseract PSM (Page Segmentation Mode) options for different text layouts
    PSM_MODES = {
        "single_word": 8,  # Treat image as a single word
        "single_line": 7,  # Treat image as a single text line
        "sparse_text": 11,  # Sparse text with OSD (Orientation and Script Detection)
        "single_block": 6,  # Uniform block of text
        "auto": 3,  # Fully automatic page segmentation
    }

    @staticmethod
    def preprocess_image(
        image: Image.Image,
        method: str = "adaptive",
        scale_factor: float = 2.0,
        denoise: bool = True,
    ) -> Image.Image:
        """
        Preprocess image for better OCR results

        Args:
            image: PIL Image to preprocess
            method: Preprocessing method ('adaptive', 'otsu', 'gaussian', 'sharpen')
            scale_factor: Factor to upscale image (larger = better for small text)
            denoise: Whether to apply denoising

        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)

        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Upscale for better recognition of small text
        if scale_factor > 1.0:
            new_width = int(gray.shape[1] * scale_factor)
            new_height = int(gray.shape[0] * scale_factor)
            gray = cv2.resize(
                gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        # Apply denoising if requested
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Apply different preprocessing methods
        if method == "adaptive":
            # Adaptive thresholding - works well with varying lighting
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            # Otsu's thresholding - automatic threshold selection
            _, processed = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "gaussian":
            # Gaussian blur + thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, processed = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "sharpen":
            # Sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            processed = cv2.filter2D(gray, -1, kernel)
        else:
            processed = gray

        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        # Convert back to PIL Image
        return Image.fromarray(processed)

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """
        Enhance image contrast using PIL

        Args:
            image: PIL Image
            factor: Contrast enhancement factor (1.0 = no change, >1.0 = more contrast)

        Returns:
            Enhanced PIL Image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def enhance_sharpness(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """
        Enhance image sharpness using PIL

        Args:
            image: PIL Image
            factor: Sharpness factor (1.0 = no change, >1.0 = sharper)

        Returns:
            Enhanced PIL Image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def extract_text_with_config(
        image: Image.Image,
        psm_mode: str = "single_line",
        oem: int = 3,
        custom_config: str = "",
    ) -> Dict[str, any]:
        """
        Extract text using specified Tesseract configuration

        Args:
            image: PIL Image to extract text from
            psm_mode: Page segmentation mode name (see PSM_MODES)
            oem: OCR Engine Mode (0=Legacy, 1=Neural nets LSTM, 2=Legacy+LSTM, 3=Default)
            custom_config: Additional Tesseract config string

        Returns:
            Dictionary with text, confidence, and config used
        """
        psm = OCREnhancer.PSM_MODES.get(psm_mode, 6)

        # Build config string
        config = f"--oem {oem} --psm {psm}"
        if custom_config:
            config += f" {custom_config}"

        try:
            # Extract text
            text = pytesseract.image_to_string(image, config=config).strip()

            # Get confidence scores
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            confidences = [
                int(conf) for conf in data["conf"] if conf != "-1" and int(conf) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": text,
                "confidence": avg_confidence,
                "config": config,
                "psm_mode": psm_mode,
                "word_count": len([w for w in text.split() if w]),
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": 0,
                "config": config,
                "psm_mode": psm_mode,
                "word_count": 0,
                "error": str(e),
            }

    @classmethod
    def extract_text_multi_method(
        cls,
        image: Image.Image,
        preprocessing_methods: List[str] = None,
        psm_modes: List[str] = None,
        padding: int = 10,
    ) -> Dict[str, any]:
        """
        Try multiple preprocessing methods and PSM modes, return best result

        Args:
            image: PIL Image to extract text from
            preprocessing_methods: List of preprocessing methods to try
            psm_modes: List of PSM modes to try
            padding: Padding to add around image

        Returns:
            Dictionary with best OCR result
        """
        if preprocessing_methods is None:
            preprocessing_methods = ["adaptive", "otsu", "gaussian"]
        if psm_modes is None:
            psm_modes = ["single_line", "single_word", "sparse_text"]

        # Add padding
        padded = Image.new(
            "RGB",
            (image.width + 2 * padding, image.height + 2 * padding),
            color="white",
        )
        padded.paste(image, (padding, padding))

        best_result = {"text": "", "confidence": 0, "method": None, "psm_mode": None}

        # Try different combinations
        for method in preprocessing_methods:
            try:
                # Preprocess image
                processed = cls.preprocess_image(padded, method=method)

                # Enhance contrast and sharpness
                processed = cls.enhance_contrast(processed, factor=1.5)
                processed = cls.enhance_sharpness(processed, factor=1.5)

                # Try different PSM modes
                for psm_mode in psm_modes:
                    result = cls.extract_text_with_config(processed, psm_mode=psm_mode)

                    # Update best result if this one is better
                    if result["confidence"] > best_result["confidence"] or (
                        result["confidence"] >= best_result["confidence"]
                        and len(result["text"]) > len(best_result["text"])
                    ):
                        best_result = {
                            "text": result["text"],
                            "confidence": result["confidence"],
                            "word_count": result["word_count"],
                            "method": method,
                            "psm_mode": psm_mode,
                            "config": result["config"],
                        }
            except Exception as e:
                continue

        return best_result

    @classmethod
    def extract_text_enhanced(
        cls,
        image: Image.Image,
        bbox: Tuple[float, float, float, float] = None,
        use_multi_method: bool = True,
        padding: int = 10,
    ) -> Dict[str, any]:
        """
        Enhanced text extraction with automatic method selection

        Args:
            image: PIL Image (or full image if bbox provided)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop before OCR
            use_multi_method: Whether to try multiple methods and select best
            padding: Padding to add around cropped region

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Crop if bbox provided
            if bbox:
                x1, y1, x2, y2 = bbox
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(image.width, int(x2) + padding)
                y2 = min(image.height, int(y2) + padding)
                cropped = image.crop((x1, y1, x2, y2))
            else:
                cropped = image

            # Use multi-method approach for best results
            if use_multi_method:
                return cls.extract_text_multi_method(cropped, padding=padding)
            else:
                # Single method (faster but potentially less accurate)
                processed = cls.preprocess_image(cropped, method="adaptive")
                processed = cls.enhance_contrast(processed, factor=1.5)
                result = cls.extract_text_with_config(processed, psm_mode="single_line")
                result["method"] = "adaptive"
                return result

        except Exception as e:
            return {
                "text": "",
                "confidence": 0,
                "word_count": 0,
                "method": None,
                "psm_mode": None,
                "error": str(e),
            }


# Convenience functions for direct use
def extract_text_from_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    enhanced: bool = True,
) -> Dict[str, any]:
    """
    Convenience function to extract text from bounding box with optional enhancement

    Args:
        image: PIL Image
        bbox: Bounding box (x1, y1, x2, y2)
        enhanced: Whether to use enhanced multi-method OCR (slower but more accurate)

    Returns:
        Dictionary with extracted text and metadata
    """
    enhancer = OCREnhancer()
    return enhancer.extract_text_enhanced(
        image, bbox=bbox, use_multi_method=enhanced, padding=15
    )


def preprocess_for_ocr(
    image: Image.Image, method: str = "adaptive", scale: float = 2.0
) -> Image.Image:
    """
    Convenience function to preprocess image for OCR

    Args:
        image: PIL Image
        method: Preprocessing method
        scale: Scale factor for upsampling

    Returns:
        Preprocessed PIL Image
    """
    return OCREnhancer.preprocess_image(image, method=method, scale_factor=scale)
