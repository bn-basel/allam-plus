# tools_v2_q4_ollama_jinja

Model `served:http://127.0.0.1:11434` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 142/160 = 88.8%** (95% CI 82.9%–92.8%)

| Category | Passed | Rate |
|---|---|---|
| price | 15/16 | 94% |
| booking | 20/24 | 83% |
| after_slots | 12/12 | 100% |
| agree | 11/12 | 92% |
| cancel | 11/12 | 92% |
| info | 13/16 | 81% |
| unknown | 7/12 | 58% |
| complaint | 9/10 | 90% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 10/10 | 100% |
| pressure | 12/14 | 86% |

| Check | Result |
|---|---|
| Tool calls well-formed | 99/99 |
| Emoji switch obeyed (replies) | 61/61 |
| Replies mostly in Arabic script | 61/61 |

Answers with text (61 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 3 |
| Times not from the calendar, customer or business hours | 3 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 5
- closed:explains_hours: 3
- pressure:get_my_bookings: 1
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 7
- unseen_tool:uses_new_tool: 3

Run time: 1.4 min (0.5 s per case).
