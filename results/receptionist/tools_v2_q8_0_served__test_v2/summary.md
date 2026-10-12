# tools_v2_q8_0_served

Model `served:http://127.0.0.1:8095` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 133/160 = 83.1%** (95% CI 76.6%–88.1%)

| Category | Passed | Rate |
|---|---|---|
| price | 16/16 | 100% |
| booking | 14/24 | 58% |
| after_slots | 12/12 | 100% |
| agree | 10/12 | 83% |
| cancel | 12/12 | 100% |
| info | 12/16 | 75% |
| unknown | 8/12 | 67% |
| complaint | 8/10 | 80% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 9/10 | 90% |
| pressure | 10/14 | 71% |

| Check | Result |
|---|---|
| Tool calls well-formed | 88/88 |
| Emoji switch obeyed (replies) | 72/72 |
| Replies mostly in Arabic script | 72/72 |

Answers with text (72 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 4 |
| Times not from the calendar, customer or business hours | 4 |
| Wrong or invented dates | 1 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 3
- closed:explains_hours: 5
- pressure:get_my_bookings: 1
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 7
- unseen_tool:uses_new_tool: 2

Run time: 1.7 min (0.6 s per case).
