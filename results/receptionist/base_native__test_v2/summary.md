# base_native

Model `allam-plus:base@shipped` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 58/160 = 36.2%** (95% CI 29.2%–43.9%)

| Category | Passed | Rate |
|---|---|---|
| price | 0/16 | 0% |
| booking | 0/24 | 0% |
| after_slots | 11/12 | 92% |
| agree | 0/12 | 0% |
| cancel | 4/12 | 33% |
| info | 10/16 | 62% |
| unknown | 0/12 | 0% |
| complaint | 0/10 | 0% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 10/10 | 100% |
| pressure | 1/14 | 7% |

| Check | Result |
|---|---|
| Tool calls well-formed | 2/6 |
| Emoji switch obeyed (replies) | 160/160 |
| Replies mostly in Arabic script | 160/160 |

Answers with text (160 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 18 |
| Times not from the calendar, customer or business hours | 10 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 7 |

Accepted answer types (cases with more than one correct answer):

- closed:explains_hours: 8
- pressure:reply: 1
- smalltalk:reply: 1
- unseen_tool:confirms_first: 10

Run time: 3.7 min (1.4 s per case).
