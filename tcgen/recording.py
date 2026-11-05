"""Turn a recording (meeting, requirements walkthrough, demo video) into a timestamped transcript.

Steps: ffmpeg pulls the audio track out of the video and squeezes it into small mono MP3 chunks,
then Groq's Whisper transcribes each chunk. The transcript lines keep [hh:mm:ss] timestamps so
every extracted requirement can point back to the moment it was said.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config


class RecordingError(ValueError):
    """Raised when a recording can't be read or transcribed."""


@dataclass
class Segment:
    start: float  # seconds from the start of the recording
    end: float
    text: str


def ffmpeg_exe() -> str:
    """The ffmpeg binary: the one bundled with imageio-ffmpeg (works on Mac/Windows/Cloud), else PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RecordingError("ffmpeg isn't available, so the audio can't be read.")
        return exe


def _duration(stderr: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def extract_audio(data: bytes, filename: str, chunk_seconds: int = config.AUDIO_CHUNK_SECONDS,
                  max_seconds: int = config.MAX_RECORDING_SECONDS) -> tuple[list[bytes], float]:
    """Audio track as small mono 16 kHz MP3 chunks (≈2.4 MB per 10 minutes) and the duration in seconds."""
    suffix = Path(filename).suffix.lower() or ".bin"
    if suffix not in config.RECORDING_TYPES:
        raise RecordingError("Unsupported recording type. Use " + ", ".join(config.RECORDING_TYPES) + ".")
    exe = ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(data)
        probe = subprocess.run([exe, "-hide_banner", "-i", str(src)], capture_output=True, text=True)
        duration = _duration(probe.stderr)
        if "Audio:" not in probe.stderr:
            raise RecordingError("This file has no audio track, so there is nothing to transcribe.")
        if duration > max_seconds + 1:
            raise RecordingError(f"The recording is {duration / 60:.0f} minutes long; the limit is "
                                 f"{max_seconds // 60} minutes. Trim it, or split it into parts.")
        out = Path(tmp) / "chunk%03d.mp3"
        r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                            "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "segment",
                            "-segment_time", str(chunk_seconds), "-reset_timestamps", "1", str(out)],
                           capture_output=True, text=True)
        chunks = [p.read_bytes() for p in sorted(Path(tmp).glob("chunk*.mp3"))]
        if r.returncode != 0 or not chunks:
            raise RecordingError("Couldn't read the audio: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
    return chunks, duration


def _segments_of(response, offset: float) -> list[Segment]:
    data = response if isinstance(response, dict) else (
        response.to_dict() if hasattr(response, "to_dict") else response.model_dump())
    segs = data.get("segments") or []
    if not segs and data.get("text"):
        segs = [{"start": 0.0, "end": 0.0, "text": data["text"]}]
    return [Segment(offset + float(s.get("start", 0)), offset + float(s.get("end", 0)), str(s.get("text", "")).strip())
            for s in segs if str(s.get("text", "")).strip()]


def transcribe(chunks: list[bytes], api_key: str = "", translate: bool = False, language: str = "",
               chunk_seconds: int = config.AUDIO_CHUNK_SECONDS, client=None, progress=None) -> list[Segment]:
    """Transcribe each chunk with Groq Whisper; with translate=True the text comes back in English."""
    if client is None:
        from groq import Groq

        client = Groq(api_key=api_key, max_retries=4, timeout=300)
    out: list[Segment] = []
    for i, chunk in enumerate(chunks):
        if progress:
            progress(i / len(chunks), f"Transcribing part {i + 1} of {len(chunks)}…")
        args = {"file": (f"part{i + 1}.mp3", chunk), "response_format": "verbose_json", "temperature": 0.0}
        try:
            if translate:
                r = client.audio.translations.create(model=config.TRANSLATE_MODEL, **args)
            else:
                r = client.audio.transcriptions.create(model=config.TRANSCRIBE_MODEL,
                                                       **({"language": language} if language else {}), **args)
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "invalid_api_key" in msg.lower():
                raise RecordingError("Groq rejected the API key.") from exc
            if "429" in msg or "rate" in msg.lower():
                raise RecordingError("Groq's audio rate limit was reached. Wait a few minutes and try again.") from exc
            raise RecordingError(f"Transcription failed: {msg[:300]}") from exc
        out.extend(_segments_of(r, i * chunk_seconds))
    if progress:
        progress(1.0, "Transcript ready")
    return out


def _ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def to_transcript(segments: list[Segment], name: str, duration: float, min_chars: int = 240) -> str:
    """Readable transcript: segments merged into short paragraphs, each starting with [hh:mm:ss]."""
    lines, cur, start = [], "", None
    for s in segments:
        if start is None:
            start = s.start
        cur = f"{cur} {s.text}".strip()
        if len(cur) >= min_chars or cur.endswith(("?", ".", "!")) and len(cur) >= min_chars // 2:
            lines.append(f"[{_ts(start)}] {cur}")
            cur, start = "", None
    if cur:
        lines.append(f"[{_ts(start or 0)}] {cur}")
    head = f"TRANSCRIPT OF RECORDING: {name} (duration {_ts(duration)})"
    return head + "\n\n" + "\n\n".join(lines)


def read_recording(data: bytes, filename: str, api_key: str, translate: bool = False, language: str = "",
                   client=None, progress=None) -> str:
    chunks, duration = extract_audio(data, filename)
    segments = transcribe(chunks, api_key, translate=translate, language=language, client=client, progress=progress)
    text = to_transcript(segments, filename, duration)
    if len(text) < config.MIN_DOCUMENT_CHARS:
        raise RecordingError("Hardly any speech was found in the recording, so there are no requirements to extract.")
    return text[: config.MAX_DOCUMENT_CHARS]
