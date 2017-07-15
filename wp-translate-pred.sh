#!/bin/bash

DEV_SOURCES=wp-data/wponly-processed/jetpack-es-source.txt

MODEL_DIR=models/en2es
PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python bin/infer.py \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 8" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt
