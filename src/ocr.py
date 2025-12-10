"""
OCR and document text extraction module.

Supports:
- PDF files (native text extraction + OCR fallback for scanned documents)
- Image files (PNG, JPG, TIFF) via OCR

Uses PyMuPDF for PDF processing and Tesseract for OCR.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from a PDF file.
    
    First attempts native text extraction (for digital PDFs).
    Falls back to OCR if the PDF appears to be scanned/image-based.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
    
    text_parts = []
    
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=file_content, filetype="pdf")
        
        for page_num, page in enumerate(doc):
            # Try native text extraction first
            page_text = page.get_text().strip()
            
            if page_text:
                text_parts.append(page_text)
                logger.debug(f"Page {page_num + 1}: Extracted {len(page_text)} chars (native)")
            else:
                # Page has no text - likely scanned, use OCR
                logger.debug(f"Page {page_num + 1}: No native text, falling back to OCR")
                ocr_text = _ocr_pdf_page(page)
                if ocr_text:
                    text_parts.append(ocr_text)
        
        doc.close()
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")
    
    combined_text = "\n\n".join(text_parts)
    
    if not combined_text.strip():
        raise ValueError("No text could be extracted from the PDF")
    
    return combined_text


def _ocr_pdf_page(page) -> str:
    """
    Perform OCR on a single PDF page by converting it to an image.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        OCR-extracted text
    """
    try:
        from PIL import Image
        import pytesseract
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "Pillow and pytesseract are required for OCR. "
            "Install with: pip install Pillow pytesseract"
        )
    
    # Render page to image at 300 DPI for good OCR quality
    mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    
    # Run OCR
    text = pytesseract.image_to_string(image, lang='eng')
    
    return text.strip()


def extract_text_from_image(file_content: bytes) -> str:
    """
    Extract text from an image file using OCR.
    
    Args:
        file_content: Raw bytes of the image file
        
    Returns:
        OCR-extracted text
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ImportError(
            "Pillow and pytesseract are required for OCR. "
            "Install with: pip install Pillow pytesseract"
        )
    
    try:
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Run OCR with English language
        text = pytesseract.image_to_string(image, lang='eng')
        
        if not text.strip():
            raise ValueError("No text could be extracted from the image")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        raise ValueError(f"Failed to extract text from image: {e}")


def extract_text(file_content: bytes, filename: str) -> str:
    """
    Extract text from a document file (PDF or image).
    
    Automatically detects file type and uses the appropriate extraction method.
    
    Args:
        file_content: Raw bytes of the file
        filename: Original filename (used to determine file type)
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is unsupported or extraction fails
    """
    suffix = Path(filename).suffix.lower()
    
    logger.info(f"Extracting text from {filename} (type: {suffix})")
    
    if suffix == '.pdf':
        return extract_text_from_pdf(file_content)
    
    elif suffix in ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'):
        return extract_text_from_image(file_content)
    
    elif suffix == '.txt':
        # Plain text file - just decode
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('latin-1')
    
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported types: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .txt"
        )


def is_supported_file(filename: str) -> bool:
    """Check if the file type is supported for text extraction."""
    suffix = Path(filename).suffix.lower()
    return suffix in ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.txt')


def get_supported_extensions() -> list:
    """Return list of supported file extensions."""
    return ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.txt']

