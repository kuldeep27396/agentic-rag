from app.services.pdf import validate_pdf_upload
from app.services.vector_store import build_document_filter


def test_rejects_non_pdf_upload():
    try:
        validate_pdf_upload("notes.txt", "text/plain", 100)
    except ValueError as exc:
        assert "Only PDF" in str(exc)
    else:
        raise AssertionError("Expected non-PDF upload to be rejected")


def test_rejects_large_pdf_upload():
    try:
        validate_pdf_upload("paper.pdf", "application/pdf", 26 * 1024 * 1024)
    except ValueError as exc:
        assert "25MB" in str(exc)
    else:
        raise AssertionError("Expected oversized PDF to be rejected")


def test_vector_filter_is_scoped_to_document():
    assert build_document_filter("doc-123") == 'document_id == "doc-123"'
