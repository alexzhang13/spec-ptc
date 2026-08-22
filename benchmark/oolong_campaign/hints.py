"""Root-prompt hints for OOLONG tasks."""

# NOTE: the exact private "v8" text from rlm-minimal-training is not public.
# This is the closest public instruction (rlm repo,
# training/environments/oolong/env.py::_QUESTION_INSTRUCTION). If you have the
# real v8 string, paste it here — every run's manifest records which was used.
HINT_V8_TREC = (
    "The context contains thousands of general-knowledge questions, one per "
    "line. Each line has a User ID and a question, and each question's answer "
    "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
    "'description and abstract concept', 'abbreviation', 'human being'. "
    "Answer the following aggregate question."
)

HINT_V8_PAIRS = (
    "The context contains thousands of general-knowledge questions, one per "
    "line, each with a User ID. Each question's answer can be labelled as one "
    "of 6 categories: 'numeric value', 'entity', 'location', 'description and "
    "abstract concept', 'abbreviation', 'human being' (the data does not "
    "provide the labels; infer them). Answer the following question about "
    "pairs of users exactly and exhaustively."
)

HINT_NAME = "v8-public-approx"
