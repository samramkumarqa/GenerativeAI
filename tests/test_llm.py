"""GroqLLM retry behaviour, with the Groq client replaced by a scripted stand-in."""

import pytest

import tcgen.llm as llm_mod
from tcgen.llm import GroqLLM, is_bad_generation
from tcgen.schemas import RequirementList


class ScriptedRunnable:
    def __init__(self, outcome):
        self.outcome = outcome

    def invoke(self, _messages):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ScriptedChat:
    """Pops one outcome per call and records which method was used."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.methods = []

    def with_structured_output(self, schema, method, **_):
        self.methods.append(method)
        return ScriptedRunnable(self.outcomes.pop(0))


BAD = Exception("Error code: 400 - {'error': {'code': 'json_validate_failed', 'failed_generation': ''}}")
OK = RequirementList(requirements=[])


@pytest.fixture
def make_llm(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)

    def _make(outcomes):
        llm = GroqLLM(api_key="test")
        llm._chat = ScriptedChat(outcomes)
        return llm

    return _make


def test_retries_bad_generation_then_succeeds(make_llm):
    llm = make_llm([BAD, OK])
    assert llm.generate("s", "u", RequirementList) == OK
    assert llm._chat.methods == ["json_schema", "json_schema"]


def test_last_attempt_falls_back_to_tool_calling(make_llm):
    llm = make_llm([BAD, BAD, OK])
    assert llm.generate("s", "u", RequirementList) == OK
    assert llm._chat.methods == ["json_schema", "json_schema", "function_calling"]


def test_empty_result_counts_as_bad_generation(make_llm):
    llm = make_llm([None, OK])
    assert llm.generate("s", "u", RequirementList) == OK


def test_other_errors_are_not_retried(make_llm):
    llm = make_llm([Exception("Error code: 401 - invalid_api_key"), OK])
    with pytest.raises(Exception, match="401"):
        llm.generate("s", "u", RequirementList)
    assert len(llm._chat.methods) == 1


def test_gives_up_after_three_bad_generations(make_llm):
    llm = make_llm([BAD, BAD, BAD])
    with pytest.raises(Exception, match="json_validate_failed"):
        llm.generate("s", "u", RequirementList)


def test_is_bad_generation():
    assert is_bad_generation(BAD)
    assert not is_bad_generation(Exception("rate limit 429"))
