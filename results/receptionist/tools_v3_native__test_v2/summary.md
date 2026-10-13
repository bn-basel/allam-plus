# tools_v3_native

Model `allam-plus:runs/tools_v3/checkpoint-120@yarn4` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 143/160 = 89.4%** (95% CI 83.6%–93.3%)

| Category | Passed | Rate |
|---|---|---|
| price | 16/16 | 100% |
| booking | 22/24 | 92% |
| after_slots | 11/12 | 92% |
| agree | 11/12 | 92% |
| cancel | 12/12 | 100% |
| info | 11/16 | 69% |
| unknown | 9/12 | 75% |
| complaint | 9/10 | 90% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 10/10 | 100% |
| pressure | 10/14 | 71% |

| Check | Result |
|---|---|
| Tool calls well-formed | 104/105 |
| Emoji switch obeyed (replies) | 55/55 |
| Replies mostly in Arabic script | 55/55 |

Answers with text (55 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 0 |
| Times not from the calendar, customer or business hours | 0 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 6
- closed:explains_hours: 2
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 8
- unseen_tool:uses_new_tool: 2

Run time: 7.1 min (2.7 s per case).
