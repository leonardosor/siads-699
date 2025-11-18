"""
Enhanced OCR preprocessing utilities for improving Tesseract accuracy
Includes image preprocessing, enhancement, and multi-config testing
"""

import re
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
        "raw_line": 13,  # Raw line without OSD or OCR
    }

    # Common character whitelist for financial forms
    FINANCIAL_FORM_CHARS = (
        "0123456789"  # Numbers
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Uppercase letters
        "abcdefghijklmnopqrstuvwxyz"  # Lowercase letters
        ".,/$%-:#() "  # Common financial symbols
    )

    @staticmethod
    def deskew_image(img_array: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew/rotation in image

        Args:
            img_array: Input image as numpy array (grayscale)

        Returns:
            Deskewed image as numpy array
        """
        try:
            # Detect edges
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = (theta * 180 / np.pi) - 90
                    angles.append(angle)

                median_angle = np.median(angles)

                # Only rotate if angle is significant (> 0.5 degrees)
                if abs(median_angle) > 0.5:
                    (h, w) = img_array.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        img_array,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    return rotated

            return img_array
        except Exception:
            return img_array

    @staticmethod
    def remove_borders(img_array: np.ndarray, border_size: int = 5) -> np.ndarray:
        """
        Remove borders from image that might confuse OCR

        Args:
            img_array: Input image as numpy array
            border_size: Size of border to remove (pixels)

        Returns:
            Image with borders removed
        """
        try:
            h, w = img_array.shape[:2]
            if h > border_size * 2 and w > border_size * 2:
                return img_array[
                    border_size : h - border_size, border_size : w - border_size
                ]
            return img_array
        except Exception:
            return img_array

    @staticmethod
    def apply_clahe(img_array: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        for better local contrast

        Args:
            img_array: Input grayscale image as numpy array

        Returns:
            Image with enhanced contrast
        """
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img_array)
        except Exception:
            return img_array

    @staticmethod
    def detect_and_invert(img_array: np.ndarray) -> np.ndarray:
        """
        Detect if image has white text on dark background and invert if needed
        Tesseract works better with dark text on light background

        Args:
            img_array: Input grayscale image as numpy array

        Returns:
            Corrected image (inverted if needed)
        """
        try:
            # Calculate mean brightness
            mean_brightness = np.mean(img_array)

            # If image is mostly dark (mean < 127), likely white-on-black
            if mean_brightness < 127:
                return cv2.bitwise_not(img_array)
            return img_array
        except Exception:
            return img_array

    @staticmethod
    def bilateral_filter(img_array: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving noise reduction

        Args:
            img_array: Input image as numpy array

        Returns:
            Filtered image
        """
        try:
            return cv2.bilateralFilter(img_array, 9, 75, 75)
        except Exception:
            return img_array

    @staticmethod
    def aggressive_morphology(
        img_array: np.ndarray, operation: str = "close"
    ) -> np.ndarray:
        """
        Apply aggressive morphological operations to connect broken characters

        Args:
            img_array: Input binary image as numpy array
            operation: 'close', 'dilate', or 'open'

        Returns:
            Morphologically processed image
        """
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            if operation == "close":
                return cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            elif operation == "dilate":
                return cv2.dilate(img_array, kernel, iterations=1)
            elif operation == "open":
                return cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)

            return img_array
        except Exception:
            return img_array

    @staticmethod
    def preprocess_image(
        image: Image.Image,
        method: str = "adaptive",
        scale_factor: float = 3.0,
        denoise: bool = True,
        deskew: bool = True,
        remove_border: bool = True,
        use_clahe: bool = True,
    ) -> Image.Image:
        """
        Preprocess image for better OCR results with advanced techniques

        Args:
            image: PIL Image to preprocess
            method: Preprocessing method ('adaptive', 'otsu', 'gaussian', 'sharpen', 'bilateral', 'clahe')
            scale_factor: Factor to upscale image (3.0 recommended for small text)
            denoise: Whether to apply denoising
            deskew: Whether to correct image rotation
            remove_border: Whether to remove borders
            use_clahe: Whether to use CLAHE for contrast enhancement

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

        # Remove borders that might confuse OCR
        if remove_border:
            gray = OCREnhancer.remove_borders(gray, border_size=3)

        # Detect and invert if white text on dark background
        gray = OCREnhancer.detect_and_invert(gray)

        # Apply CLAHE for better local contrast (before upscaling)
        if use_clahe:
            gray = OCREnhancer.apply_clahe(gray)

        # Upscale for better recognition of small text (increased to 3x)
        if scale_factor > 1.0:
            new_width = int(gray.shape[1] * scale_factor)
            new_height = int(gray.shape[0] * scale_factor)
            gray = cv2.resize(
                gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        # Deskew if requested
        if deskew:
            gray = OCREnhancer.deskew_image(gray)

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
        elif method == "bilateral":
            # Bilateral filter for edge-preserving smoothing
            filtered = OCREnhancer.bilateral_filter(gray)
            _, processed = cv2.threshold(
                filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "clahe":
            # Extra CLAHE pass
            enhanced = OCREnhancer.apply_clahe(gray)
            _, processed = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "sharpen":
            # Sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            processed = cv2.filter2D(gray, -1, kernel)
        else:
            processed = gray

        # Apply denoising after thresholding if requested
        if denoise and method in ["adaptive", "otsu", "gaussian", "bilateral", "clahe"]:
            processed = cv2.fastNlMeansDenoising(processed, h=10)

        # Apply morphological operations to connect broken characters and clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

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
    def post_process_text(text: str, apply_corrections: bool = False) -> str:
        """
        Post-process OCR text to fix common errors

        Args:
            text: Raw OCR text
            apply_corrections: Whether to apply character corrections (disabled by default)

        Returns:
            Cleaned text
        """
        if not text:
            return text

        result = text

        # Only apply corrections if explicitly enabled
        if apply_corrections:
            # Fix common OCR mistakes
            corrections = {
                "l": "1",  # lowercase L often confused with 1
                "O": "0",  # uppercase O often confused with 0
                "|": "I",  # pipe often confused with I
                "`": "'",  # backtick to apostrophe
                """: '"',  # Smart quotes to regular quotes
                """: '"',
                "'": "'",
                "'": "'",
            }

            # Apply corrections only in numeric contexts
            # If the text looks like it should be numeric (has mostly digits)
            digit_count = sum(c.isdigit() for c in text)
            if digit_count > len(text) * 0.3:  # More than 30% digits
                for old, new in corrections.items():
                    result = result.replace(old, new)

        # Remove extra whitespace (always apply this)
        result = " ".join(result.split())

        return result

    @staticmethod
    def extract_text_with_config(
        image: Image.Image,
        psm_mode: str = "single_line",
        oem: int = 3,
        custom_config: str = "",
        use_whitelist: bool = False,
    ) -> Dict[str, any]:
        """
        Extract text using specified Tesseract configuration

        Args:
            image: PIL Image to extract text from
            psm_mode: Page segmentation mode name (see PSM_MODES)
            oem: OCR Engine Mode (0=Legacy, 1=Neural nets LSTM, 2=Legacy+LSTM, 3=Default)
            custom_config: Additional Tesseract config string
            use_whitelist: Whether to constrain to common financial form characters

        Returns:
            Dictionary with text, confidence, and config used
        """
        psm = OCREnhancer.PSM_MODES.get(psm_mode, 6)

        # Build config string
        config = f"--oem {oem} --psm {psm}"
        if custom_config:
            config += f" {custom_config}"

        # Add character whitelist for financial forms if requested
        if use_whitelist:
            # Escape special characters for tesseract
            whitelist = OCREnhancer.FINANCIAL_FORM_CHARS.replace("\\", "\\\\")
            config += f" -c tessedit_char_whitelist='{whitelist}'"

        try:
            # Extract text
            text = pytesseract.image_to_string(image, config=config).strip()

            # Post-process to clean whitespace only (no character corrections)
            text = OCREnhancer.post_process_text(text, apply_corrections=False)

            # Get confidence scores with detailed word-level data
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )

            # Calculate confidence filtering out low-confidence words
            confidences = []
            high_conf_words = []
            for i, conf in enumerate(data["conf"]):
                if conf != "-1" and int(conf) > 0:
                    conf_val = int(conf)
                    confidences.append(conf_val)
                    # Only include words with confidence > 60
                    if conf_val > 60 and data["text"][i].strip():
                        high_conf_words.append(data["text"][i])

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": text,
                "confidence": avg_confidence,
                "config": config,
                "psm_mode": psm_mode,
                "word_count": len([w for w in text.split() if w]),
                "high_conf_word_count": len(high_conf_words),
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": 0,
                "config": config,
                "psm_mode": psm_mode,
                "word_count": 0,
                "high_conf_word_count": 0,
                "error": str(e),
            }

    @classmethod
    def extract_text_multi_method(
        cls,
        image: Image.Image,
        preprocessing_methods: List[str] = None,
        psm_modes: List[str] = None,
        oem_modes: List[int] = None,
        padding: int = 10,
        try_whitelist: bool = False,
        conservative: bool = True,
    ) -> Dict[str, any]:
        """
        Try multiple preprocessing methods, PSM modes, and OEM modes, return best result

        Args:
            image: PIL Image to extract text from
            preprocessing_methods: List of preprocessing methods to try
            psm_modes: List of PSM modes to try
            oem_modes: List of OCR Engine Modes to try
            padding: Padding to add around image
            try_whitelist: Whether to try with character whitelist
            conservative: If True, use gentler preprocessing (recommended)

        Returns:
            Dictionary with best OCR result
        """
        if preprocessing_methods is None:
            # Use only proven methods - fewer but better
            preprocessing_methods = ["adaptive", "otsu"]
        if psm_modes is None:
            # Auto-detect based on image size
            # Large images likely contain multiple lines/blocks
            if image.width * image.height > 100000:  # Large region
                psm_modes = ["auto", "single_block", "single_line"]
            else:
                # Small regions: likely form fields
                psm_modes = ["single_line", "single_word"]
        if oem_modes is None:
            # Just use default OEM (best overall)
            oem_modes = [3]

        # Add generous padding
        padded = Image.new(
            "RGB",
            (image.width + 2 * padding, image.height + 2 * padding),
            color="white",
        )
        padded.paste(image, (padding, padding))

        best_result = {
            "text": "",
            "confidence": 0,
            "method": None,
            "psm_mode": None,
            "oem": None,
        }

        # Also try with NO preprocessing first as baseline
        for psm_mode in psm_modes:
            for oem in oem_modes:
                result = cls.extract_text_with_config(
                    padded, psm_mode=psm_mode, oem=oem, use_whitelist=False
                )

                if result["confidence"] > best_result["confidence"] or (
                    result["confidence"] >= best_result["confidence"] - 5
                    and len(result["text"]) > len(best_result["text"])
                ):
                    best_result = {
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "word_count": result["word_count"],
                        "high_conf_word_count": result.get("high_conf_word_count", 0),
                        "method": "none",
                        "psm_mode": psm_mode,
                        "oem": oem,
                        "config": result["config"],
                    }

        # Try different combinations with preprocessing
        for method in preprocessing_methods:
            try:
                # Preprocess image - use CONSERVATIVE settings
                if conservative:
                    processed = cls.preprocess_image(
                        padded,
                        method=method,
                        scale_factor=2.0,  # Moderate scaling (was 3.0)
                        denoise=False,  # Skip denoising - can remove details
                        deskew=False,  # Skip deskewing - can distort
                        remove_border=False,  # Keep borders
                        use_clahe=False,  # Skip CLAHE initially
                    )
                else:
                    # Aggressive mode
                    processed = cls.preprocess_image(
                        padded,
                        method=method,
                        scale_factor=3.0,
                        denoise=True,
                        deskew=True,
                        remove_border=True,
                        use_clahe=True,
                    )

                # Light enhancement only
                processed = cls.enhance_contrast(
                    processed, factor=1.3
                )  # Reduced from 2.0
                processed = cls.enhance_sharpness(
                    processed, factor=1.3
                )  # Reduced from 2.0

                # Try different PSM modes
                for psm_mode in psm_modes:
                    # Try different OEM modes
                    for oem in oem_modes:
                        # Try without whitelist first
                        result = cls.extract_text_with_config(
                            processed, psm_mode=psm_mode, oem=oem, use_whitelist=False
                        )

                        # Update best result if this one is better
                        # Prioritize: 1) higher confidence, 2) more text
                        is_better = False
                        if result["confidence"] > best_result["confidence"] + 5:
                            is_better = True
                        elif abs(
                            result["confidence"] - best_result["confidence"]
                        ) <= 5 and len(result["text"]) > len(best_result["text"]):
                            is_better = True

                        if is_better:
                            best_result = {
                                "text": result["text"],
                                "confidence": result["confidence"],
                                "word_count": result["word_count"],
                                "high_conf_word_count": result.get(
                                    "high_conf_word_count", 0
                                ),
                                "method": method,
                                "psm_mode": psm_mode,
                                "oem": oem,
                                "config": result["config"],
                            }

            except Exception:
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
        image, bbox=bbox, use_multi_method=enhanced, padding=15  # Moderate padding
    )


def preprocess_for_ocr(
    image: Image.Image,
    method: str = "adaptive",
    scale: float = 3.0,
    advanced: bool = True,
) -> Image.Image:
    """
    Convenience function to preprocess image for OCR

    Args:
        image: PIL Image
        method: Preprocessing method
        scale: Scale factor for upsampling (3.0 recommended)
        advanced: Whether to use advanced preprocessing (deskew, CLAHE, etc.)

    Returns:
        Preprocessed PIL Image
    """
    if advanced:
        return OCREnhancer.preprocess_image(
            image,
            method=method,
            scale_factor=scale,
            denoise=True,
            deskew=True,
            remove_border=True,
            use_clahe=True,
        )
    else:
        return OCREnhancer.preprocess_image(
            image,
            method=method,
            scale_factor=scale,
            denoise=False,
            deskew=False,
            remove_border=False,
            use_clahe=False,
        )
