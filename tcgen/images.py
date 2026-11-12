"""Read the text out of images (photographed BRD pages, screenshots, whiteboard shots, wireframes).

Each image is tidied up with Pillow (rotated the right way up, flattened, shrunk to a sensible size)
and sent to a Groq vision model, which transcribes the text and keeps headings, numbered lists and
tables. For a screen or wireframe it also lists the fields and buttons it shows, which is what
UI test cases are built from. The texts are joined, page by page, and used as the BRD.
"""

from __future__ import annotations

import base64
import io
import re

from . import config


class ImageError(ValueError):
    """Raised when an image can't be read or no text is found in it."""


PROMPT = """Transcribe ALL the text in this image exactly as written, top to bottom. It is part of a requirements document, a screenshot of an application, a wireframe or a whiteboard.

Rules:
- Keep the structure: headings on their own lines, numbered and bulleted lists as lists, requirement IDs (FR-01, REQ-3 ...) exactly as shown, tables as Markdown table rows.
- Do not summarise, translate, correct or add anything. Mark unreadable words as [unreadable].
- If the image shows an application screen, form or wireframe, after the text add a section
  "Screen elements:" listing each field, dropdown, checkbox, button and link with its label, and any
  validation or error messages shown.
- If there is no text at all, answer exactly: NO TEXT
Return only the transcription."""


def is_image(filename: str) -> bool:
    return filename.lower().endswith(tuple(config.IMAGE_TYPES))


def prepare(data: bytes, filename: str) -> str:
    """The image as a JPEG data URL: upright, RGB, longest side at most IMAGE_MAX_SIDE pixels."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if len(data) > config.MAX_IMAGE_BYTES:
        raise ImageError(f"{filename} is larger than {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"{filename} isn't an image that can be opened.") from exc
    img = ImageOps.exif_transpose(img)  # phone photos are often stored sideways
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")  # transparent screenshots: white, not black, background
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((config.IMAGE_MAX_SIDE, config.IMAGE_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # reasoning models may think aloud
    text = re.sub(r"^```\w*\n|\n```$", "", text.strip())
    return text.strip()


def transcribe_image(data_url: str, client, model: str = config.VISION_MODEL) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "invalid_api_key" in msg.lower():
            raise ImageError("Groq rejected the API key.") from exc
        if "429" in msg or "rate limit" in msg.lower():
            raise ImageError("Groq's rate limit was reached. Wait a minute and try again.") from exc
        if "model" in msg.lower() and ("not found" in msg.lower() or "decommissioned" in msg.lower()):
            raise ImageError(f"Groq's vision model {model} isn't available. Update VISION_MODEL in tcgen/config.py.") from exc
        raise ImageError(f"Reading the image failed: {msg[:300]}") from exc
    return _clean(r.choices[0].message.content)


def read_images(files: list[tuple[str, bytes]], api_key: str = "", client=None, progress=None) -> str:
    """Text of one or more images, in the given order, as one document."""
    if not files:
        raise ImageError("No images were uploaded.")
    if len(files) > config.MAX_IMAGES:
        raise ImageError(f"Upload at most {config.MAX_IMAGES} images at a time.")
    prepared = [(name, prepare(data, name)) for name, data in files]  # fail fast on a bad file
    if client is None:
        from groq import Groq

        client = Groq(api_key=api_key, max_retries=4, timeout=120)
    parts = []
    for i, (name, url) in enumerate(prepared):
        if progress:
            progress(i / len(prepared), f"Reading image {i + 1} of {len(prepared)}: {name}…")
        text = transcribe_image(url, client)
        if text and text.strip().upper() != "NO TEXT":
            parts.append(f"--- Image {i + 1}: {name} ---\n{text}" if len(prepared) > 1 else text)
    if progress:
        progress(1.0, "Text extracted")
    doc = "\n\n".join(parts).strip()
    if len(doc) < config.MIN_IMAGE_TEXT_CHARS:
        raise ImageError("No readable text was found in the image(s). Try a sharper, well-lit photo or a screenshot.")
    return doc[: config.MAX_DOCUMENT_CHARS]
