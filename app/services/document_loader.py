from pathlib import Path

import pymupdf


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8-sig")

    elif suffix == ".pdf":
        with pymupdf.open(file_path) as document:
            text = "\n".join(
                page.get_text() for page in document
            )

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not text.strip():
        raise ValueError(
            "No extractable text found in the document."
        )

    return text