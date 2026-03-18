# TODO: Implement UnpairedTPAFDataset
# Loads unpaired TPAF and H&E patches.
# TPAF normalization: percentile clipping [1st, 99th] → [0, 1] (never min-max).
# TPAF must be loaded with tifffile for 16-bit support (PIL clips to 8-bit).
