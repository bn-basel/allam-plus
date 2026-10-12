# tools_v2_q4_k_m_served

Model `served:http://127.0.0.1:8095` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 139/160 = 86.9%** (95% CI 80.8%–91.3%)

| Category | Passed | Rate |
|---|---|---|
| price | 16/16 | 100% |
| booking | 20/24 | 83% |
| after_slots | 12/12 | 100% |
| agree | 11/12 | 92% |
| cancel | 10/12 | 83% |
| info | 12/16 | 75% |
| unknown | 7/12 | 58% |
| complaint | 8/10 | 80% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 9/10 | 90% |
| pressure | 12/14 | 86% |

| Check | Result |
|---|---|
| Tool calls well-formed | 99/99 |
| Emoji switch obeyed (replies) | 61/61 |
| Replies mostly in Arabic script | 61/61 |

Answers with text (61 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 2 |
| Times not from the calendar, customer or business hours | 2 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 0 |

Accepted answer types (cases with more than one correct answer):

- closed:checks_open_day: 5
- closed:explains_hours: 3
- pressure:get_my_bookings: 1
- pressure:handoff_to_human: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 6
- unseen_tool:uses_new_tool: 3

Run time: 1.4 min (0.5 s per case).
