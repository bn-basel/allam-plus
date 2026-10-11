# long_v1_native

Model `allam-plus:runs/long_v1/checkpoint-183@yarn4` on the **test_v2** split (160 cases), short prompt, scorer v1.3.

**Pass rate: 63/160 = 39.4%** (95% CI 32.1%–47.1%)

| Category | Passed | Rate |
|---|---|---|
| price | 0/16 | 0% |
| booking | 0/24 | 0% |
| after_slots | 11/12 | 92% |
| agree | 0/12 | 0% |
| cancel | 4/12 | 33% |
| info | 14/16 | 88% |
| unknown | 0/12 | 0% |
| complaint | 0/10 | 0% |
| closed | 8/8 | 100% |
| smalltalk | 14/14 | 100% |
| unseen_tool | 9/10 | 90% |
| pressure | 3/14 | 21% |

| Check | Result |
|---|---|
| Tool calls well-formed | 1/1 |
| Emoji switch obeyed (replies) | 160/160 |
| Replies mostly in Arabic script | 159/160 |

Answers with text (160 of 160) that contain (a case can have several):

| Problem | Answers |
|---|---|
| Invented numbers (prices, counts) | 17 |
| Times not from the calendar, customer or business hours | 4 |
| Wrong or invented dates | 0 |
| Claims an action no tool performed | 7 |

Accepted answer types (cases with more than one correct answer):

- closed:explains_hours: 8
- pressure:reply: 2
- smalltalk:reply: 1
- unseen_tool:confirms_first: 9

Run time: 5.0 min (1.9 s per case).
