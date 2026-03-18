# TODO: Implement StructureLoss nn.Module (L_struct — mask consistency).
# Self-consistency: input nuclear mask vs. mask extracted from generated H&E.
# NOT a cross-domain pixel-wise loss (never compare TPAF source to H&E target).
# struct_loss_mode: threshold | pretrained_detector
# Warm-up logic is handled by the trainer (struct_warmup_epochs, struct_rampup_epochs).
