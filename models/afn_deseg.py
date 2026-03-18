# TODO: Implement AFN-DeSeg wrapper (frozen pretrained nuclear segmentation model).
# CRITICAL: This model must NEVER be trained or fine-tuned during SA-CUT training.
# Output mask dtype must be float32 in [0, 1].
# Log mask quality metrics: coverage ratio, connected component stats.
