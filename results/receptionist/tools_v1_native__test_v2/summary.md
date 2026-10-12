# tools_v1_native

Model `allam-plus:runs/tools_v1/checkpoint-400@yarn4` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 133/160 = 83.1%** (95% CI 76.6%–88.1%)

| Category | Passed | Rate |
|---|---|---|
| price | 16/16 | 100% |
| booking | 18/24 | 75% |
| after_slots | 11/12 | 92% |
| agree | 12/12 | 100% |
| cancel | 12/12 | 100% |
| info | 11/16 | 69% |
| unknown | 6/12 | 50% |
| complaint | 9/10 | 90% |
| closed | 8/8 | 100% |
| smalltalk | 13/14 | 93% |
| unseen_tool | 7/10 | 70% |
| pressure | 10/14 | 71% |

| Check | Result |
|---|---|
| Tool calls well-formed | 100/101 |
| Emoji switch obeyed (replies) | 59/59 |
| Replies mostly in Arabic script | 59/59 |

Answers with text (59 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 1 |
| Times not from the calendar, customer or business hours | 1 |
| Wrong or invented dates | 1 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 5
- closed:explains_hours: 3
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 6
- unseen_tool:uses_new_tool: 1

Run time: 7.0 min (2.6 s per case).
