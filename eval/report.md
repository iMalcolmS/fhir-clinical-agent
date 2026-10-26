# Agent evaluation

`openai/gpt-oss-120b` on 37 questions about 5 synthetic patients. Answers scored by rules; tools and patient ids read from the trace.

| Metric | Value |
|---|---|
| **Answer pass rate** | **100%** |
| Expected tool called | 100% |
| Patient isolation (no other patient's id used) | 100% |
| Answerable questions wrongly refused | 0% |
| Mean tool calls per question | 2.0 |
| Median latency | 1.2 s |
| Mean tokens per question | 2,844 |

| Category | Questions | Pass rate | Expected tool called |
|---|---|---|---|
| allergies | 5 | 100% | 100% |
| conditions | 4 | 100% | 100% |
| cross-patient | 5 | 100% | 100% |
| imaging | 5 | 100% | 100% |
| medications | 3 | 100% | 100% |
| observations | 10 | 100% | 100% |
| unanswerable | 5 | 100% | – |
