# tools_v2_native

Model `allam-plus:runs/tools_v2/checkpoint-180@yarn4` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 136/160 = 85.0%** (95% CI 78.7%–89.7%)

| Category | Passed | Rate |
|---|---|---|
| price | 15/16 | 94% |
| booking | 19/24 | 79% |
| after_slots | 11/12 | 92% |
| agree | 11/12 | 92% |
| cancel | 12/12 | 100% |
| info | 10/16 | 62% |
| unknown | 9/12 | 75% |
| complaint | 10/10 | 100% |
| closed | 7/8 | 88% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 8/10 | 80% |
| pressure | 10/14 | 71% |

| Check | Result |
|---|---|
| Tool calls well-formed | 109/110 |
| Emoji switch obeyed (replies) | 50/50 |
| Replies mostly in Arabic script | 50/50 |

Answers with text (50 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 1 |
| Times not from the calendar, customer or business hours | 1 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 6
- closed:explains_hours: 1
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 7
- unseen_tool:uses_new_tool: 1
