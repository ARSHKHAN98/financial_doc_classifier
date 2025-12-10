"""
Tests for the OCR and document text extraction module.

Run with: pytest tests/test_ocr.py -v

Note: Some tests require pytesseract/Tesseract to be installed.
"""

import pytest
from pathlib import Path

from src.ocr import (
    is_supported_file,
    get_supported_extensions,
    extract_text,
)


class TestFileTypeValidation:
    """Tests for file type validation."""
    
    def test_supported_pdf(self):
        assert is_supported_file("document.pdf") is True
        assert is_supported_file("DOCUMENT.PDF") is True
    
    def test_supported_images(self):
        assert is_supported_file("scan.png") is True
        assert is_supported_file("photo.jpg") is True
        assert is_supported_file("image.jpeg") is True
        assert is_supported_file("scan.tiff") is True
        assert is_supported_file("scan.tif") is True
        assert is_supported_file("image.bmp") is True
    
    def test_supported_text(self):
        assert is_supported_file("document.txt") is True
    
    def test_unsupported_types(self):
        assert is_supported_file("document.docx") is False
        assert is_supported_file("spreadsheet.xlsx") is False
        assert is_supported_file("archive.zip") is False
        assert is_supported_file("script.py") is False
    
    def test_no_extension(self):
        assert is_supported_file("noextension") is False


class TestGetSupportedExtensions:
    """Tests for supported extensions list."""
    
    def test_returns_list(self):
        extensions = get_supported_extensions()
        assert isinstance(extensions, list)
        assert len(extensions) > 0
    
    def test_includes_common_types(self):
        extensions = get_supported_extensions()
        assert '.pdf' in extensions
        assert '.png' in extensions
        assert '.jpg' in extensions
        assert '.txt' in extensions


class TestExtractTextFromPlainText:
    """Tests for plain text file extraction."""
    
    def test_extract_utf8_text(self):
        """Should extract text from UTF-8 encoded content."""
        content = "This is an invoice for consulting services.".encode('utf-8')
        text = extract_text(content, "document.txt")
        assert text == "This is an invoice for consulting services."
    
    def test_extract_latin1_text(self):
        """Should handle Latin-1 encoded content."""
        content = "Invoice with special chars: café résumé".encode('latin-1')
        text = extract_text(content, "document.txt")
        assert "Invoice" in text


class TestExtractTextValidation:
    """Tests for input validation."""
    
    def test_unsupported_file_raises_error(self):
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError) as exc_info:
            extract_text(b"content", "document.docx")
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_empty_filename_extension(self):
        """Should raise ValueError for files without extension."""
        with pytest.raises(ValueError):
            extract_text(b"content", "noextension")


class TestPDFExtraction:
    """Tests for PDF text extraction."""
    
    @pytest.fixture
    def sample_pdf_bytes(self):
        """Create a minimal PDF with text for testing."""
        try:
            import fitz
            
            # Create a simple PDF in memory
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Invoice #12345 for consulting services")
            
            pdf_bytes = doc.tobytes()
            doc.close()
            return pdf_bytes
        except ImportError:
            pytest.skip("PyMuPDF not installed")
    
    def test_extract_text_from_digital_pdf(self, sample_pdf_bytes):
        """Should extract text from a digital PDF."""
        text = extract_text(sample_pdf_bytes, "test.pdf")
        assert "Invoice" in text
        assert "12345" in text
    
    def test_empty_pdf_raises_error(self):
        """Should raise error for PDF with no extractable text."""
        try:
            import fitz
            
            # Create empty PDF
            doc = fitz.open()
            doc.new_page()  # Empty page
            pdf_bytes = doc.tobytes()
            doc.close()
            
            with pytest.raises(ValueError) as exc_info:
                extract_text(pdf_bytes, "empty.pdf")
            assert "No text" in str(exc_info.value)
            
        except ImportError:
            pytest.skip("PyMuPDF not installed")


class TestImageOCR:
    """Tests for image OCR extraction."""
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a simple image with text for testing."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create image with text
            img = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 40), "Invoice Number 98765", fill='black')
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        except ImportError:
            pytest.skip("Pillow not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("pytesseract", reason="pytesseract not installed"),
        reason="pytesseract not installed"
    )
    def test_extract_text_from_image(self, sample_image_bytes):
        """Should extract text from an image using OCR."""
        try:
            text = extract_text(sample_image_bytes, "scan.png")
            # OCR might not be perfect, but should get key words
            assert "Invoice" in text or "98765" in text
        except Exception as e:
            if "tesseract" in str(e).lower():
                pytest.skip("Tesseract not installed on system")
            raise


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_small_file(self):
        """Should handle very small text files."""
        content = b"Hi"
        text = extract_text(content, "tiny.txt")
        assert text == "Hi"
    
    def test_unicode_filename(self):
        """Should handle unicode in filename."""
        content = b"Test content"
        text = extract_text(content, "documento_español.txt")
        assert text == "Test content"
    
    def test_mixed_case_extension(self):
        """Should handle mixed case extensions."""
        content = b"Content"
        text = extract_text(content, "file.TxT")
        assert text == "Content"

