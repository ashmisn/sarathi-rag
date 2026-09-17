# Failure analysis

These cases are taken from `evaluation/agent_test_output.txt`, which is an existing manual test transcript rather than a held-out evaluation set.

| Case | Observed behavior | Likely cause | Fix to prioritize |
| --- | --- | --- | --- |
| Gap in Median at 75 km/h | Returned the 81–120 km/h band and `90 m` instead of the 66–80 km/h band and `70 m`. | The generator applied the wrong range boundary after retrieval. | Add speed-band parsing and a numeric answer validator before generation. |
| Transverse bars at 70 km/h, `d1 = 40 m` | Compared the query against the ≤50 and 51–65 bands, then never gave a clean 66–80 compliance verdict. | Planner/generator did not map 70 to the three-set band. | Make the compliance evaluator select exactly one speed band and require a verdict. |
| Object hazard on a sharp curve | Borrowed Narrow Bridge placement distances as “guidance” for an object hazard. | Related-topic retrieval was treated as authority for a different sign type. | Reject cross-type extrapolation unless the retrieved source explicitly applies to the queried type. |
| Hospital sign area | Repeated dimensions but did not calculate `600 × 800 = 480,000 mm²` or `450 × 600 = 270,000 mm²`. | Prompt requested arithmetic, but no post-generation calculation check existed. | Add deterministic arithmetic checks for benchmarked dimensions. |
| U-Turn Prohibited clear visibility | Answer cited a general/other-topic clause and mixed in road-marking preview distances even though no speed was given. | Retrieval fallback was too permissive and citation validation was absent. | Return “not found” when the requested field is absent; validate every citation against retrieved rows. |

The transcript also shows a positive control: STOP sign size at 55 km/h and speed-hump radius at 30 km/h were answered with the expected values and relevant clauses. Those should remain in regression tests.
