# tools_v2_q4_ollama_go_text

Model `served:http://127.0.0.1:11434` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 126/160 = 78.8%** (95% CI 71.8%–84.4%)

| Category | Passed | Rate |
|---|---|---|
| price | 16/16 | 100% |
| booking | 17/24 | 71% |
| after_slots | 12/12 | 100% |
| agree | 11/12 | 92% |
| cancel | 11/12 | 92% |
| info | 10/16 | 62% |
| unknown | 5/12 | 42% |
| complaint | 6/10 | 60% |
| closed | 5/8 | 62% |
| smalltalk | 13/14 | 93% |
| unseen_tool | 10/10 | 100% |
| pressure | 10/14 | 71% |

| Check | Result |
|---|---|
| Tool calls well-formed | 97/102 |
| Emoji switch obeyed (replies) | 58/58 |
| Replies mostly in Arabic script | 58/58 |

Answers with text (58 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 3 |
| Times not from the calendar, customer or business hours | 3 |
| Wrong or invented dates | 1 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 3
- closed:explains_hours: 2
- pressure:get_my_bookings: 1
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 8
- unseen_tool:uses_new_tool: 2

Run time: 1.4 min (0.5 s per case).
