"""Recording source: real ffmpeg audio extraction, Groq Whisper calls mocked at the HTTP layer."""

import json
import subprocess

import httpx
import pytest
from groq import Groq

from tcgen import recording
from tcgen.pipeline import EXTRACT_SYSTEM
from tcgen.recording import RecordingError, Segment, extract_audio, read_recording, to_transcript, transcribe

FF = recording.ffmpeg_exe()


def make_media(path, seconds=12, audio=True):
    cmd = [FF, "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=64x64:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return path.read_bytes()


def groq_client(handler):
    return Groq(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_extracts_small_mono_chunks_from_video(tmp_path):
    chunks, duration = extract_audio(make_media(tmp_path / "meeting.mp4"), "meeting.mp4", chunk_seconds=5)
    assert 11.5 < duration < 12.5
    assert len(chunks) == 3 and all(c[:3] == b"ID3" or c[:2] == b"\xff\xfb" for c in chunks)
    assert sum(map(len, chunks)) < 80_000  # 32 kbps mono: about 4 KB per second


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(RecordingError, match="Unsupported"):
        extract_audio(b"x", "notes.exe")
    with pytest.raises(RecordingError, match="no audio track"):
        extract_audio(make_media(tmp_path / "silent.mp4", audio=False), "silent.mp4")
    with pytest.raises(RecordingError, match="limit is 0 minutes|limit is"):
        extract_audio(make_media(tmp_path / "long.mp4"), "long.mp4", max_seconds=5)


def test_transcribe_calls_groq_whisper_and_offsets_chunks():
    seen = []

    def handler(req):
        body = req.content.decode("latin-1")
        seen.append((req.url.path, body))
        n = len(seen)
        return httpx.Response(200, json={"text": "x", "segments": [
            {"start": 1.0, "end": 4.0, "text": f" Part {n} first."},
            {"start": 4.0, "end": 9.0, "text": f" Part {n} second."}]})

    segs = transcribe([b"a", b"b"], client=groq_client(handler), chunk_seconds=600)
    assert [(s.start, s.text) for s in segs] == [(1.0, "Part 1 first."), (4.0, "Part 1 second."),
                                                  (601.0, "Part 2 first."), (604.0, "Part 2 second.")]
    path, body = seen[0]
    assert path == "/openai/v1/audio/transcriptions"
    assert 'name="model"' in body and "whisper-large-v3-turbo" in body and "verbose_json" in body

    seen.clear()
    transcribe([b"a"], client=groq_client(handler), translate=True)
    assert seen[0][0] == "/openai/v1/audio/translations" and "whisper-large-v3\r\n" in seen[0][1]


def test_groq_errors_are_explained():
    with pytest.raises(RecordingError, match="rejected the API key"):
        transcribe([b"a"], client=groq_client(lambda r: httpx.Response(401, json={"error": {"message": "invalid_api_key"}})))


def test_transcript_has_timestamps_and_feeds_extraction(tmp_path):
    segs = [Segment(0, 5, "Welcome everyone."), Segment(65, 70, "The coupon applies only above 500 rupees."),
            Segment(3725, 3730, "Delivery slots are two hours long.")]
    t = to_transcript(segs, "kickoff.mp4", 3735, min_chars=10)
    assert t.splitlines()[0] == "TRANSCRIPT OF RECORDING: kickoff.mp4 (duration 01:02:15)"
    assert "[00:01:05] The coupon applies only above 500 rupees." in t and "[01:02:05] Delivery slots" in t
    assert "TRANSCRIPT" in EXTRACT_SYSTEM and "[hh:mm:ss]" in EXTRACT_SYSTEM

    def handler(req):
        return httpx.Response(200, json={"text": "", "segments": [
            {"start": 2.0, "end": 8.0, "text": " The customer must be able to apply one coupon per order, and the "
             "coupon only applies when the basket total is above five hundred rupees. If the coupon has expired, show "
             "an error message saying the coupon is no longer valid."}]})

    text = read_recording(make_media(tmp_path / "m.mp4"), "m.mp4", "k", client=groq_client(handler))
    assert text.startswith("TRANSCRIPT OF RECORDING: m.mp4") and "[00:00:02] The customer must" in text
