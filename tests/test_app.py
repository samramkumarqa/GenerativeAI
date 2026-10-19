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
    assert [m.value for m in at.metric] == ["14", "14", "100%"]
    assert at.get("download_button")


def test_without_key_asks_for_one(monkeypatch):
    at = _app(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY")
    at.run()
    _button(at, "Use sample BRD").click().run()
    assert any("Groq API key" in w.value for w in at.warning)
