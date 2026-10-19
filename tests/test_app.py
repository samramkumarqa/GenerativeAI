"""Drive the Streamlit UI end to end with the fakes swapped in for Groq and the embedder."""

from streamlit.testing.v1 import AppTest

from conftest import ROOT
from fakes import FakeLLM, HashEmbedder


def _app(monkeypatch):
    import tcgen.llm
    import tcgen.rag

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(tcgen.llm, "GroqLLM", lambda api_key, model: FakeLLM())
    monkeypatch.setattr(tcgen.rag, "FastEmbedder", HashEmbedder)
    return AppTest.from_file(str(ROOT / "main.py"), default_timeout=30)


def _button(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def test_full_flow_with_sample(monkeypatch):
    at = _app(monkeypatch).run()
    assert not at.exception

    _button(at, "Use sample BRD").click().run()
    assert any("sample_brd.pdf" in i.value for i in at.info)

    _button(at, "Extract requirements").click().run()
    assert not at.exception
    assert any("14 requirements found" in c.value for c in at.caption)

    _button(at, "Generate for 14 requirements").click().run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric] == ["0", "0", "14", "0%"]  # approved, rejected, pending, coverage

    # Download is locked until everything is reviewed and a reviewer signs off.
    assert _download(at).disabled
    assert any("14 test case(s) still pending" in i.value for i in at.info)

    _button(at, "Approve all pending").click().run()
    assert [m.value for m in at.metric] == ["14", "0", "0", "100%"]
    assert _download(at).disabled
    assert any("Enter the reviewer's name" in i.value for i in at.info)

    at.text_input(key="reviewer").input("Ramkumar").run()
    assert not at.exception, at.exception
    assert not _download(at).disabled
    assert any("signed off by Ramkumar" in s.value for s in at.success)


def _download(at):
    return next(b.proto for b in at.get("download_button") if "approved test pack" in b.proto.label)


def test_without_key_asks_for_one(monkeypatch):
    at = _app(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY")
    at.run()
    _button(at, "Use sample BRD").click().run()
    assert any("Groq API key" in w.value for w in at.warning)


def test_reject_and_regenerate_with_feedback(monkeypatch):
    at = _app(monkeypatch).run()
    _button(at, "Use sample BRD").click().run()
    _button(at, "Extract requirements").click().run()
    _button(at, "Generate for 14 requirements").click().run()

    # Simulate the reviewer rejecting TC-001 with a comment in the review table.
    tcs = at.session_state["test_cases"]
    tcs[0] = tcs[0].model_copy(update={"review_status": "Rejected", "reviewer_comment": "Cover INR 198"})
    at.session_state["test_cases"] = tcs
    at.session_state["review_version"] += 1
    at.run()
    assert [m.value for m in at.metric][:3] == ["0", "1", "13"]

    _button(at, "Regenerate 1 rejected").click().run()
    assert not at.exception, at.exception
    assert any("1 test case(s) rewritten" in s.value for s in at.success)

    tcs = at.session_state["test_cases"]
    new = tcs[0]
    assert (new.id, new.revision, new.review_status) == ("TC-015", 2, "Pending")
    assert "Cover INR 198" in new.title  # the fake model echoes the feedback it was given
    assert [m.value for m in at.metric][:3] == ["0", "0", "14"]

    # The rejected original is kept for the audit sheet.
    assert [tc.id for tc in at.session_state["superseded"]] == ["TC-001"]

    _button(at, "Approve all pending").click().run()
    at.text_input(key="reviewer").input("Ramkumar").run()
    assert not _download(at).disabled
