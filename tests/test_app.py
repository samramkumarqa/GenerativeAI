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


def _click_in_review(at, label):
    """Click a button inside the review fragment.

    AppTest doesn't re-render a fragment after a fragment-scoped st.rerun, so do a
    full run afterwards, as the browser would show it."""
    _button(at, label).click().run()
    return at.run()


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
    assert [m.value for m in at.metric] == ["0", "14", "0%"]  # approved, not approved, coverage

    # Download is locked until something is approved and a reviewer signs off.
    assert _download(at).disabled
    assert any("Tick Approve on at least one" in i.value for i in at.info)

    _click_in_review(at, "☑ Select all")
    assert [m.value for m in at.metric] == ["14", "0", "100%"]
    assert _download(at).disabled
    assert any("Enter the reviewer's name" in i.value for i in at.info)

    at.text_input(key="reviewer").input("Ramkumar").run()
    assert not at.exception, at.exception
    assert not _download(at).disabled
    assert any("signed off by Ramkumar" in s.value for s in at.success)

    _click_in_review(at, "☐ Clear all")
    assert [m.value for m in at.metric] == ["0", "14", "0%"]
    assert _download(at).disabled


def _download(at):
    return next(b.proto for b in at.get("download_button") if "approved test cases (Excel)" in b.proto.label)


def test_without_key_asks_for_one(monkeypatch):
    at = _app(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY")
    at.run()
    _button(at, "Use sample BRD").click().run()
    assert any("Groq API key" in w.value for w in at.warning)


def test_regenerate_unticked_with_feedback(monkeypatch):
    at = _app(monkeypatch).run()
    _button(at, "Use sample BRD").click().run()
    _button(at, "Extract requirements").click().run()
    _button(at, "Generate for 14 requirements").click().run()

    # Reviewer approves everything except TC-001.
    tcs = at.session_state["test_cases"]
    at.session_state["test_cases"] = [
        tc.model_copy(update={"review_status": "Pending" if tc.id == "TC-001" else "Approved"}) for tc in tcs
    ]
    at.session_state["review_version"] += 1
    at.run()
    assert [m.value for m in at.metric][:2] == ["13", "1"]

    at.text_area(key="feedback").input("Cover INR 198").run()
    _click_in_review(at, "🔁 Regenerate 1 unticked")
    assert not at.exception, at.exception
    assert any("1 test case(s) rewritten" in s.value for s in at.success)

    new = at.session_state["test_cases"][0]
    assert (new.id, new.revision, new.review_status) == ("TC-015", 2, "Pending")
    assert "Cover INR 198" in new.title  # the fake model echoes the feedback it was given
    assert [m.value for m in at.metric][:2] == ["13", "1"]

    # The original is kept, with the feedback, for the Not Approved audit sheet.
    old = at.session_state["superseded"]
    assert [(tc.id, tc.reviewer_comment) for tc in old] == [("TC-001", "Cover INR 198")]

    at.text_input(key="reviewer").input("Ramkumar").run()
    assert not _download(at).disabled  # 13 approved is enough; the unticked one is left out
