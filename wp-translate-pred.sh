#!/bin/bash

DEV_SOURCES=wp-data/wponly-processed/jetpack-es-source.txt

MODEL_DIR=models/en2es-goldilocks
PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python bin/infer.py \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 3" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt
