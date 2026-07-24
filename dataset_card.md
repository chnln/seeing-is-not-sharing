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

SINS is a 13,077-instance binary common-ground judgment (interpretation
matching judgment) dataset, derived from the public Grounded Misunderstandings
in MapTask (GMMT) annotations. Each instance asks whether the giver and
follower interpret the referring expression as the same landmark. Rows
preserve identifiers and context-window provenance while mapping GMMT status to
`gold_label`: `aligned` becomes `yes`; `pending` and `misunderstood` become
`no`.

## Related Resources

| Resource | Location |
| --- | --- |
| SINS code and prompts | [GitHub](https://github.com/chnln/seeing-is-not-sharing) |
| SINS paper | [ACL Anthology](https://aclanthology.org/2026.sigdial-1.49/) · [arXiv:2606.31719](https://arxiv.org/abs/2606.31719), SIGDIAL 2026 |
| GMMT code | [GitHub](https://github.com/chnln/grounded-misunderstandings-in-maptask) |
| GMMT dataset | [Hugging Face](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask) |
| GMMT paper | [LREC Proceedings](https://lrec.elra.info/lrec2026-main-392) · [arXiv:2511.03718](https://arxiv.org/abs/2511.03718), LREC 2026 |

## Load with Datasets

```python
from datasets import load_dataset

ds = load_dataset("chnln/seeing-is-not-sharing", split="train")
```

## Data Fields

The release contains `ref_id`, `dialogue_id`, `map_id`, `utt_id`,
`transaction_id`, `context_transaction_ids`, `end_utt_id_of_context`,
`timed_unit_ids`, `expression`, `status`, and `gold_label`.

## Excluded Material

The SINS HF dataset does not contain dialogue context. It does not contain MapTask maps,
images, OCR, or image-derived text. It also does not contain
generated prompts, model predictions, log-probabilities, or copies of the
underlying MapTask source files. Users who download MapTask themselves may
reconstruct context with the SINS code repository and a local GMMT checkout.

## Licensing and Provenance

SINS is released under CC BY 4.0. It is derived from
[GMMT](https://huggingface.co/datasets/chnln/grounded-misunderstandings-in-maptask),
which is also CC BY 4.0. Citation details for both papers are provided below.
See `NOTICE` in the code repository for details.

## Citation

SINS builds on the GMMT data and annotation scheme introduced in our LREC 2026
paper. If you use SINS, please cite both the SINS paper and the GMMT paper.

### SINS: Task and Experiments

Published at SIGDIAL 2026.

```bibtex
@inproceedings{li2026seeing,
  title = {Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue},
  author = {Li, Nan and Gatt, Albert and Poesio, Massimo},
  editor = {Choi, Jinho D. and Chen, Yun-Nung and Funakoshi, Kotaro and Emami, Ali},
  booktitle = {Proceedings of the 27th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  month = aug,
  year = {2026},
  address = {Atlanta, Georgia, USA},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2026.sigdial-1.49/},
  pages = {694--710}
}
```

Paper: [ACL Anthology](https://aclanthology.org/2026.sigdial-1.49/) ·
[arXiv preprint](https://arxiv.org/abs/2606.31719).

### GMMT: Data and Annotation Scheme

Published at LREC 2026.

```bibtex
@inproceedings{li2026grounded,
  title = {Grounded Misunderstandings in Asymmetric Dialogue: A Perspectivist Annotation Scheme for MapTask},
  author = {Li, Nan and Gatt, Albert and Poesio, Massimo},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {4988--5001},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  url = {https://lrec.elra.info/lrec2026-main-392},
  doi = {10.63317/59anbt78wyj7}
}
```

Paper: [LREC Proceedings](https://lrec.elra.info/lrec2026-main-392) ·
[arXiv preprint](https://arxiv.org/abs/2511.03718).
