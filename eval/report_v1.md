# Agent evaluation

`openai/gpt-oss-120b` on 37 questions about 5 synthetic patients. Answers scored by rules; tools and patient ids read from the trace.

| Metric | Value |
|---|---|
| **Answer pass rate** | **95%** |
| Expected tool called | 97% |
| Patient isolation (no other patient's id used) | 100% |
| Answerable questions wrongly refused | 0% |
| Mean tool calls per question | 2.1 |
| Median latency | 1.1 s |
| Mean tokens per question | 2,849 |

| Category | Questions | Pass rate | Expected tool called |
|---|---|---|---|
| allergies | 5 | 100% | 100% |
| conditions | 4 | 100% | 75% |
| cross-patient | 5 | 100% | 100% |
| imaging | 5 | 100% | 100% |
| medications | 3 | 100% | 100% |
| observations | 10 | 90% | 100% |
| unanswerable | 5 | 80% | – |

## Failures

| Category | Question | Tools called | Answer (start) |
|---|---|---|---|
| observations | What was Adriana Hintz's most recent BMI? | find_patients, get_observations | The most recent BMI recorded for Adriana Hintz was **97.859 %** on **2021‑11‑06**. |
| unanswerable | What is Agueda Gerhold's NHS number? | find_patients, get_patient_summary | Agueda Gerhold’s NHS number is not listed in the patient’s record. |
