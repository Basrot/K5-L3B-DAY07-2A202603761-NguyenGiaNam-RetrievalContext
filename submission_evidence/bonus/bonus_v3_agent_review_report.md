# BONUS LAB 07 — V3 TEST Agent Review

## Run integrity and scope

- First frozen TEST benchmark SHA-256 (reported in run): `3681d6fcddee641c6ae8767e9b431307cc032693d9ab2e67c751bbee9134f11e`. The ZIP supplied here contains 30 case JSON files and one review worksheet, not the frozen benchmark itself; this hash is recorded rather than independently rehashed.
- Index: `bonus_multimodal_v3`, 999 records, manually authored image/chart/diagram captions.
- Reviewed 30 archived Agent decisions and their retrieved evidence, without re-running the model.
- Retrieval frozen first run: **16/24 All-Gold Hit@5 (66.7%)**. Agent's own search queries differ, so this number is not the number of Agent answers supported by its retrieved sources.

## Answer correctness

- Fully correct: **19/24 (79.2%)**.
- Partially correct: **1/24 (4.2%)**: candidate_019 describes concatenation but does not identify the requested operation `s1 + s2` / `+`.
- Incorrect/no answer: **4/24 (16.7%)**, all false abstentions: candidates **006, 010, 014, 023**.
- Graded 0/1/2: **39/48 points (81.25%)**, a rubric-dependent descriptive measure, not official Bonus points.
- 6/6 labeled unanswerable candidates were refused correctly *conditional on their provisional labels*. Full-source absence was not conclusively verified; do not assert 100% independently verified abstention accuracy.
- Strict provisional end-to-end behavioral success including the 6 provisional abstentions: **25/30 (83.3%)**. Excluding provisional cases, confirmed answer correctness is **19/24**.

## Citation audit

- **20/20 answered cases** have cited IDs in retrieved evidence and cited passages supporting substantive statements. **19/20** include each ID as its own bracketed inline citation; candidate_017 puts two IDs in one bracketed group `[id1, id2]`, which violates the configured marker format even though both IDs are valid. Citation validity does not guarantee question completeness: candidate_019 is citation-supported but incomplete.
- **10/10 abstentions** have empty answers and citations.
- Visual answers use manually composed image/figure captions, not an independent visual-language interpretation of PDF pixels.

## Failure review

| Case | Result | Why |
|---|---|---|
| 006 | Grade 0, false abstention | Keyword query routed to `table`, excluding Gold text for `//` |
| 010 | Grade 0, false abstention | Correct underscore-example source absent from retrieved context |
| 014 | Grade 0, false abstention | ASCII 65 → A table not retrieved |
| 019 | Grade 1, partial | Says “concatenation” but omits explicit `s1 + s2` or `+` |
| 023 | Grade 0, false abstention | Retrieved image caption already mentions login/register but model abstained |

## Important interpretation

- Candidate_017's A+ answer is supported by retrieved p194/p209 text; candidate_020's `\A` or `^` is supported by p260 text; candidate_024's two answer components are supported by p45 and p122 text despite the second citation differing from its Gold ID.
- Candidate_016 and 018 answered correctly using text versions of tabular material although the original Gold record IDs were table records. This demonstrates why **exact Gold-ID Hit@5** and **answer evidence sufficiency** must be reported separately.
- Corpus absence of 6 questions was checked only with phrase matching against 999 indexed items. This does not establish exhaustive negative labels.
- Agent may abstain after only one search despite two being allowed.
- **No established +15 Bonus entitlement**: without the official rubric's final application and validated labels, this remains a self-assessed benchmark.

## Files produced

- `bonus_v3_agent_review_scored.csv`: per-case grading and notes.
- `bonus_v3_agent_score_summary.json`: aggregate metrics.
- This report records the reasoning and qualifications.
