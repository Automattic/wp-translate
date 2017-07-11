#!/bin/bash

VOCAB_SOURCE=charmaps/en.tsv
VOCAB_TARGET=charmaps/es.tsv
TRAIN_SOURCES=wp-data/processed/wpcom-es-source.txt
TRAIN_TARGETS=wp-data/processed/wpcom-es-target.txt
TRAIN_STEPS=1000

MODEL_DIR=models/test

python -m bin.train \
  --config_paths="
      ./wp-translate-model.yml,
      ./train_seq2seq.yml,
      ./text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
