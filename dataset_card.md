---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- dialogue
- common-ground
- maptask
- vision-language
size_categories:
- 10K<n<100K
---

# Seeing Is Not Sharing (SINS) Binary Common-Ground Judgment Dataset

## Dataset Description

SINS is a 13,077-instance binary common-ground judgment dataset derived from
the public Grounded Misunderstandings in MapTask (GMMT) annotations. Each row
preserves identifiers and context-window provenance while mapping GMMT status
to `gold_label`: `aligned` becomes `yes`; `pending` and `misunderstood` become
`no`.

## Data Fields

The release contains `ref_id`, `dialogue_id`, `map_id`, `utt_id`,
`transaction_id`, `context_transaction_ids`, `end_utt_id_of_context`,
`timed_unit_ids`, `expression`, `status`, and `gold_label`.

## Excluded Material

The SINS HF dataset does not contain dialogue context. It does not contain MapTask maps,
images, OCR, or image-derived text. It also does not contain
generated prompts, model predictions, log-probabilities, or copies of the
underlying MapTask source files. Users who have authorised local access to
MapTask may reconstruct context with this repository's scripts and a local
GMMT checkout; reconstructed files must remain local.

## Licensing and Provenance

SINS is released under CC BY 4.0. It is derived from
[`chnln/grounded-misunderstandings-in-maptask`](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask),
which is also CC BY 4.0. Please cite both the SINS paper and GMMT when using
this dataset. See `NOTICE` in the code repository for details.
