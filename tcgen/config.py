"""Central settings. Values here are the defaults the app and tests share."""

# Groq models offered in the UI. All support strict JSON-schema structured output.
MODELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B (best quality)",
    "openai/gpt-oss-20b": "GPT-OSS 20B (faster, higher free-tier limits)",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Embedding model used for retrieval (runs locally via fastembed / ONNX, no API key).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval chunking.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_PER_REQUIREMENT = 3

# Requirement extraction reads the document in sections this large (characters).
# Kept small so each call stays well inside Groq free-tier tokens-per-minute limits.
EXTRACTION_SECTION_SIZE = 3000
EXTRACTION_SECTION_OVERLAP = 300

# How many requirements go into one generation call.
GENERATION_BATCH_SIZE = 3

MAX_DOCUMENT_CHARS = 200_000
MIN_DOCUMENT_CHARS = 200

TEST_TYPES = [
    "Functional",
    "Negative",
    "Boundary",
    "Integration",
    "UI",
    "API",
    "Security",
    "Performance",
    "Usability",
    "Accessibility",
    "Regression",
    "UAT",
]
DEFAULT_TEST_TYPES = ["Functional", "Negative", "Boundary"]

EXECUTION_STATUSES = ["Not Run", "Pass", "Fail", "Blocked"]

# Recordings (meetings, walkthroughs, demo videos) are transcribed with Groq Whisper.
RECORDING_TYPES = [".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".ogg", ".mpeg"]
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"   # fast, keeps the spoken language
TRANSLATE_MODEL = "whisper-large-v3"          # translations to English need the full model
AUDIO_CHUNK_SECONDS = 600                      # 10-minute chunks ≈ 2.4 MB each, far below Groq's 25 MB limit
MAX_RECORDING_SECONDS = 60 * 60                # keeps a run inside the free tier's audio limits
