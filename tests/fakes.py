"""Offline stand-ins for Groq and the embedding model, so tests need no network or API key."""

import hashlib
import re

import numpy as np

from tcgen.schemas import (
    ExtractedRequirement,
    GeneratedTestCase,
    RequirementList,
    TestCaseBatch,
)


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector: crude, deterministic, good enough to rank chunks."""

    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        return out


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, user, schema):
        self.calls.append((system, user, schema.__name__))
        if schema is RequirementList:
            # Reports the same requirement from two overlapping sections, plus one without a source ID.
            return RequirementList(
                requirements=[
                    ExtractedRequirement(
                        source_id="FR-01", title="Cart minimum", description="Subtotal must be at least INR 199.",
                        category="Functional", priority="High", source_excerpt="below INR 199",
                    ),
                    ExtractedRequirement(
                        source_id="FR-01", title="Cart Minimum!", description="dup",
                        category="Functional", priority="High", source_excerpt="",
                    ),
                    ExtractedRequirement(
                        source_id="", title="Delivery fee", description="Free delivery from INR 499.",
                        category="Business Rule", priority="High", source_excerpt="INR 499",
                    ),
                ]
            )
        if schema is TestCaseBatch:
            ids = re.findall(r"^- \[([^\]]+)\]", user, flags=re.MULTILINE)
            cases = [
                GeneratedTestCase(
                    requirement_id=rid, title=f"Verify {rid} happy path", test_type="Functional",
                    priority="High", preconditions="User logged in",
                    steps=["Open cart", "  ", "Tap Checkout"], test_data="", expected_result="Checkout opens",
                )
                for rid in ids
            ]
            # Something the pipeline must drop: points at a requirement not in this batch.
            cases.append(
                GeneratedTestCase(
                    requirement_id="REQ-999", title="Hallucinated", test_type="Functional", priority="Low",
                    preconditions="", steps=["x"], test_data="", expected_result="y",
                )
            )
            return TestCaseBatch(test_cases=list(reversed(cases)))
        raise AssertionError(f"unexpected schema {schema}")
