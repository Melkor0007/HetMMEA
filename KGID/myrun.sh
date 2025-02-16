#!/bin/bash
export HF_HOME=/root/huggingface

CHECKPOINT_PATH="/root/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/548275c8b99de56dec203c0e793be18e030f2f4c"

PKL_FILE="/root/autodl-tmp/fr_dbp15k_link_img_dict_full.pkl"
SAVE_PATH="/root/autodl-tmp/savepath"
ATTRS_PATH="/root/SNAG_MMEA/data/mmkg/DBP15K/fr_en/training_attrs_1"
cd Qwen-VL

python -u gen_img_desc.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --pkl-file "$PKL_FILE" \
  --save-path "$SAVE_PATH" \
  --training-attrs "$ATTRS_PATH" \