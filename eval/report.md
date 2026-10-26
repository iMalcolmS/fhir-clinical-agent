# Agent evaluation

`openai/gpt-oss-120b` on 37 questions about 5 synthetic patients. Answers scored by rules; tools and patient ids read from the trace.

| Metric | Value |
|---|---|
| **Answer pass rate** | **97%** |
| Expected tool called | 100% |
| Patient isolation (no other patient's id used) | 100% |
| Answerable questions wrongly refused | 6% |
| Mean tool calls per question | 2.1 |
| Median latency | 1.2 s |
| Mean tokens per question | 3,085 |

| Category | Questions | Pass rate | Expected tool called |
|---|---|---|---|
| allergies | 5 | 100% | 100% |
| conditions | 4 | 100% | 100% |
| cross-patient | 5 | 100% | 100% |
| imaging | 5 | 100% | 100% |
| medications | 3 | 100% | 100% |
| observations | 10 | 90% | 100% |
| unanswerable | 5 | 100% | – |

## Failures

| Category | Question | Tools called | Answer (start) |
|---|---|---|---|
| observations | What was Adriana Hintz's most recent BMI in kg/m2? | find_patients, get_observations | I could not find this information in the patient's record. |
