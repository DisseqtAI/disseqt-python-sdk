# Draft golden sets

These files are **drafts**: every label was written by an assistant, not by two independent
humans, so every item has `"source": "draft"`.

They exist to (1) exercise the calibration pipeline end to end, (2) give human raters something to
start from, and (3) show the input shape each judge kind expects. A calibration run against them
reports statistics, but a report **cannot allow gating** from draft or silver items: only items
with `"source": "human"` in the held-out test split count.

To turn a draft into evidence:

    disseqt judges golden export --golden drafts/<kind>.jsonl --out label.csv   # blind: labels hidden
    # two raters each fill `rater` and `label`; disagreements need `adjudicated_label`
    disseqt judges golden import rater_a.csv rater_b.csv --out .disseqt/golden/<kind>.jsonl
    disseqt judges calibrate --kind <kind> --model service:<model> --golden .disseqt/golden

Gating needs 30+ human items per class for binary kinds (100+ for multiclass), so a handful of
drafts is only a starting point.
