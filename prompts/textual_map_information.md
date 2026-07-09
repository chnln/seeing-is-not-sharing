# Textual Map-Information Prompt Examples

The SIGDIAL paper used two textual map-information conditions. The dataset does not contain
these blocks, landmark lists, discrepancy descriptions, OCR, or any other image-derived text.
They are reproduced here only as the two published q1ec1/m12 prompt examples from the paper
appendix; this lightweight release does not generate them.

## Text-Landmark-Names

System prompt map-information line:

> You are given the list of landmark names on each participant's map (see below in the dialogue context).

The exact q1ec1/m12 user-prompt block is in
`fixtures/q1ec1_m12_landmark_names.txt`.

## Text-Discrepancy-Detail

System prompt map-information line:

> You are given the landmark names on each participant's map and a description of how the two maps differ (see below in the dialogue context).

The exact q1ec1/m12 user-prompt block is in
`fixtures/q1ec1_m12_discrepancy_detail.txt`.
