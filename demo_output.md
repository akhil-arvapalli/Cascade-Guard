# CascadeGuard demo output

_generated 2026-10-31 06:27_

## Scenario 1 — faithful run

- steps: 3, max CHRS: 0.12717468738555907, max CHAF: 1.0
- alerted: False, analyze: 2.031 ms

| step | kind | H | H_cascade | CHRS | fidelity |
|---|---|---|---|---|---|
| s1_retrieve | retrieve | 0.21195781230926514 | 0.21195781230926514 | 0.12717468738555907 | cheap |
| s2_reason | reason | None | 0.19202214618265856 | 0.11521328770959513 | cheap |
| s3_synthesize | synthesize | None | 0.16162771057411254 | 0.09697662634446752 | cheap |

## Scenario 2 — cascading hallucination (paper Fig.1)

- steps: 5, max CHRS: 0.5068609218882888, max CHAF: 1.2796667843865535
- alerted: True, analyze: 1.911 ms

| step | kind | H | H_cascade | CHRS | fidelity |
|---|---|---|---|---|---|
| s1_plan | plan | None | 0.0 | 0.0 | cheap |
| s2_retrieve | retrieve | 0.07902383804321289 | 0.07902383804321289 | 0.047414302825927734 | cheap |
| s3_reason | reason | 0.7845014333724976 | 0.8036941309635282 | 0.4822164785781169 | cheap |
| s4_tool | tool_call | 0.6261879205703735 | 0.7406271596677398 | 0.4443762958006439 | cheap |
| s5_synth | synthesize | 0.6601470112800598 | 0.844768203147148 | 0.5068609218882888 | cheap |

**ALERT** at `s4_tool` (S_n=0.8961) — root cause: s4_tool (85%), s3_reason (15%)

Run `python demo.py --serve` to reproduce in the dashboard.
