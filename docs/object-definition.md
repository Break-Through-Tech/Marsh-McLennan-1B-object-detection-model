# What Counts as an Object (Draft)

Status: draft for the team to agree on before labeling the gold set (plan Task #4). The items under "Open questions" need a team decision or an answer from our Challenge Advisor.

## Definition

An **object** is a physical, non-human thing named in the incident description that was involved in how the injury happened: the person was using it, struck it, was struck by it, fell from or onto it, tripped on it, was cut or burned by it, or swallowed it.

| Label it | Do not label it |
| --- | --- |
| Products and tools: `ladder`, `knife`, `hedge clipper` | People: `mom`, `patient`, `another child` |
| Furniture and fixtures: `bed`, `table`, `door`, `shelf` | Body parts: `head`, `left ankle` |
| Surfaces and structures: `floor`, `stairs`, `railing` | Injuries, symptoms, diagnoses: `laceration`, `dizziness` |
| Vehicles: `scooter`, `bicycle`, `car door` | Places: `home`, `school`, `bathroom` |
| Substances and small items: `coin`, `glass bottle` | Times and measurements: `2 days ago`, `7 ft` |

## Labeling rules

There are two files to label, both in `data/gold/` and both following the rules below:

- `gold_sample.csv` (200 rows) is the test set. The final precision score is computed on it, so do not tune the extractor against it.
- `dev_sample.csv` (100 rows) is used only for tuning, never for the final score.

1. Label only what the text says. Do not infer an object that is not written (for "WALKING UPSTAIRS" label `stairs`; for "FELL" with nothing else, write `NONE`).
2. Write each object once, lowercase and singular, as the shortest phrase that still identifies it: `car seat`, not `seat`; `table`, not `short wooden table`.
3. Separate multiple objects with a semicolon: `bed; floor`.
4. If no object is mentioned, or the narrative shows `***` where the object should be, write `NONE` in `gold_objects`. Always fill in `labeler` so the row counts as labeled.
5. Label from the text alone. Do not look up the NEISS product codes; NEISS codes at most three products and often skips objects the narrative mentions.

## Input and output

Input: one incident description (the narrative with the age/sex prefix and the diagnosis section removed, as produced by `src/clean_neiss.py`).

Output for v1: the list of object phrases for that description.

```
input:   FELL AND LANDED ON THE LID OF A TRASH CAN AND HAS FELT PAIN IN HER RIGHT RIBS
output:  trash can
```

Every extractor writes a CSV with `case_id` and `objects` (semicolon-separated), which is what `src/evaluate.py` scores against the gold set.

## Open questions

- **Classification target.** The challenge asks us to extract *and classify* objects. Which categories will the final workers' compensation test set use: NEISS product codes, Marsh's own categories, or free text? This decides what "classify" means for v2.
- **Activities.** NEISS codes sports as products (`BASKETBALL, ACTIVITY AND RELATED EQUIPMENT`). Is "playing basketball" an object, or only a physical ball?
- **Parts versus wholes.** For "the lid of a trash can", do we label `trash can`, `lid`, or both? The example above assumes the whole.
- **Animals.** NEISS does not code them, but workers' compensation descriptions may involve them (dog bites). Label them or not?
- **Background objects.** For "got out of bed and twisted her leg", is `bed` involved? The definition above says yes, because she was using it.
