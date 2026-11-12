"""Image source: real Pillow processing, Groq vision calls mocked at the HTTP layer."""

import base64
import io
import json

import httpx
import pytest
from groq import Groq
from PIL import Image

from tcgen import config
from tcgen.images import ImageError, is_image, prepare, read_images

BRD_PAGE = """# Checkout requirements
FR-01 The cart total must include GST at 18%.
FR-02 A coupon applies only to orders above 500 rupees.
| Field | Rule |
| Pincode | 6 digits |"""


def png(size=(400, 300), mode="RGB", color="white", exif_orientation=None):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(buf, "JPEG", exif=exif)
    else:
        img.save(buf, "PNG")
    return buf.getvalue()


def client(handler):
    return Groq(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def reply(text):
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": text}}]})


def decode(data_url):
    assert data_url.startswith("data:image/jpeg;base64,")
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def test_is_image():
    assert is_image("Page 1.JPG") and is_image("screen.png") and is_image("w.webp")
    assert not is_image("brd.pdf") and not is_image("notes.md")


def test_prepare_shrinks_rotates_and_flattens():
    big = decode(prepare(png((5000, 2500)), "big.png"))
    assert big.size == (config.IMAGE_MAX_SIDE, 1000) and big.mode == "RGB"
    # EXIF orientation 6 = stored sideways: the result is upright (portrait)
    assert decode(prepare(png((600, 400), exif_orientation=6), "phone.jpg")).size == (400, 600)
    # Transparent screenshots get a white background, not black
    assert decode(prepare(png(mode="RGBA", color=(0, 0, 0, 0)), "t.png")).getpixel((5, 5)) == (255, 255, 255)


def test_bad_files_are_rejected():
    with pytest.raises(ImageError, match="isn't an image"):
        prepare(b"not an image", "x.png")
    with pytest.raises(ImageError, match="at most"):
        read_images([("a.png", png())] * (config.MAX_IMAGES + 1), client=client(lambda r: reply("x")))
    with pytest.raises(ImageError, match="No images"):
        read_images([])


def test_reads_each_image_with_the_vision_model_in_order():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((req.url.path, body))
        return reply(f"<think>hmm</think>\n```\nPage {len(seen)}\n{BRD_PAGE}\n```")

    text = read_images([("p1.jpg", png()), ("p2.png", png())], client=client(handler))
    path, body = seen[0]
    assert path == "/openai/v1/chat/completions" and body["model"] == config.VISION_MODEL
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "text" and "Transcribe ALL the text" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert text.startswith("--- Image 1: p1.jpg ---\nPage 1\n# Checkout requirements")
    assert "--- Image 2: p2.png ---\nPage 2" in text
    assert "<think>" not in text and "```" not in text


def test_single_image_has_no_header_and_empty_images_are_explained():
    assert read_images([("s.png", png())], client=client(lambda r: reply(BRD_PAGE))) == BRD_PAGE
    with pytest.raises(ImageError, match="No readable text"):
        read_images([("blank.png", png())], client=client(lambda r: reply("NO TEXT")))


def test_groq_errors_are_explained():
    with pytest.raises(ImageError, match="rejected the API key"):
        read_images([("a.png", png())], client=client(
            lambda r: httpx.Response(401, json={"error": {"message": "invalid_api_key"}})))
    with pytest.raises(ImageError, match="isn't available"):
        read_images([("a.png", png())], client=client(
            lambda r: httpx.Response(404, json={"error": {"message": "The model `x` does not exist or not found"}})))


def test_image_text_feeds_requirement_extraction():
    """The text read from an image goes through the same pipeline as a typed BRD."""
    from fakes import FakeLLM
    from tcgen.pipeline import extract_requirements

    text = read_images([("brd.jpg", png())], client=client(lambda r: reply(BRD_PAGE * 3)))
    reqs = extract_requirements(text, FakeLLM())
    assert reqs
