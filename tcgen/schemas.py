"""Pydantic models for everything the LLM returns and the app passes around.

Groq strict JSON-schema mode requires every field to be required, so the
LLM-facing models have no defaults. Empty strings stand in for "not present".
"""

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["High", "Medium", "Low"]


# ---------- LLM output: requirement extraction ----------

class ExtractedRequirement(BaseModel):
    source_id: str = Field(
        description="The requirement's ID exactly as written in the document "
        "(e.g. 'FR-03', 'BR 2.1'). Empty string if the document gives none."
    )
    title: str = Field(description="Short name of the requirement, max ~10 words.")
    description: str = Field(
        description="One or two sentences stating what the system must do, "
        "including any rules, limits or values the document specifies."
    )
    category: Literal["Functional", "Non-Functional", "Business Rule"]
    priority: Priority = Field(
        description="Priority stated in the document; otherwise your best judgement."
    )
    source_excerpt: str = Field(
        description="A short verbatim quote (max ~30 words) from the document "
        "that this requirement is based on."
    )


class RequirementList(BaseModel):
    requirements: list[ExtractedRequirement]


# ---------- LLM output: test case generation ----------

class GeneratedTestCase(BaseModel):
    requirement_id: str = Field(
        description="ID of the requirement this test verifies, exactly as given in the input."
    )
    title: str = Field(description="What the test verifies, phrased as 'Verify ...'.")
    test_type: str = Field(description="One of the requested test types.")
    priority: Priority
    preconditions: str = Field(description="State required before the test starts.")
    steps: list[str] = Field(description="Ordered, concrete actions. One action per item.")
    test_data: str = Field(
        description="Specific input values used, or empty string if none are needed."
    )
    expected_result: str = Field(
        description="Observable outcome that proves the requirement is met."
    )


class TestCaseBatch(BaseModel):
    test_cases: list[GeneratedTestCase]


# ---------- App-side records ----------

class Requirement(BaseModel):
    id: str
    title: str
    description: str
    category: str
    priority: str
    source_excerpt: str = ""


REVIEW_STATUSES = ["Pending", "Approved", "Rejected"]


class TestCase(BaseModel):
    id: str
    requirement_id: str
    title: str
    test_type: str
    priority: str
    preconditions: str
    steps: list[str]
    test_data: str
    expected_result: str
    # Human-in-the-loop review
    review_status: str = "Pending"
    reviewer_comment: str = ""
    revision: int = 1  # bumped each time the model rewrites it from reviewer feedback
