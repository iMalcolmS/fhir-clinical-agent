"""Rule-based scoring for the evaluation set.

Rules are deterministic, cheap and transparent: every pass/fail can be
explained by reading the answer. They can misjudge unusual phrasings, so the
run also saves every answer to results.csv for manual review of failures.
"""

import re

REFUSAL = "could not find this information"  # matches the agent's fixed refusal sentence

# Answers that start by denying the premise, e.g. "No, ..." or
# "There is no record of ..." or "The patient has not been diagnosed ...".
NEGATION = re.compile(
    r"^\W*(no\b|not\b|none\b|there (is|are|was|were) no\b|"
    r"the (patient|record)[^.]{0,40}\b(has|had|have|was|is|does|did) not\b|"
    r"the (patient|record)[^.]{0,40}\b(has|had|have|was|is|does|did)n't\b)",
    re.IGNORECASE,
)

NUMBER = re.compile(r"\d+(?:\.\d+)?")


# The agent is told to use one fixed refusal sentence, but sometimes refuses in
# its own words ("... is not listed in the patient's record"). These patterns
# count as a refusal too. Added after the first run (see eval/report_v1.md).
REFUSAL_VARIANTS = re.compile(
    r"\bnot (listed|recorded|documented|available|included|present|found)\b"
    r"[^.]{0,40}\b(record|chart|notes)\b",
    re.IGNORECASE,
)


def is_refusal(answer):
    return REFUSAL in answer.lower() or bool(REFUSAL_VARIANTS.search(answer))


def says_no(answer):
    return is_refusal(answer) or bool(NEGATION.search(answer.strip()))


def numbers_in(text):
    return [float(n) for n in NUMBER.findall(text)]


def contains_numbers(answer, expected, tol):
    found = numbers_in(answer)
    return all(any(abs(n - e) <= tol for n in found) for e in expected)


def keyword_recall(answer, keywords):
    if not keywords:
        return 0.0
    text = answer.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


def score(check, answer):
    """Return (passed, score between 0 and 1) for one answer.

    check["type"]:
      refuse   - the record does not contain the answer; the model must refuse
      negative - the patient does NOT have this; the model must deny or refuse
      keywords - every keyword must appear (score = share of keywords found)
      numbers  - every expected number must appear within a tolerance
      year     - the expected year must appear
    """
    kind = check["type"]
    if kind == "refuse":
        ok = is_refusal(answer)
        return ok, float(ok)
    if kind == "negative":
        ok = says_no(answer)
        return ok, float(ok)
    if is_refusal(answer):  # answerable question that the model refused
        return False, 0.0
    if kind == "keywords":
        recall = keyword_recall(answer, check["keywords"])
        return recall == 1.0, recall
    if kind == "numbers":
        ok = contains_numbers(answer, check["numbers"], check.get("tol", 0.05))
        return ok, float(ok)
    if kind == "year":
        ok = str(check["year"]) in answer
        return ok, float(ok)
    raise ValueError(f"Unknown check type: {kind}")
