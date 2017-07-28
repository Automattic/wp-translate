#!/bin/bash

MODEL_DIR=models/en2es-goldilocks
PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

DEV_SOURCES=wp-data/wponly-processed/jetpack-2017-es-source.txt
OUTFILE=${PRED_DIR}/jetpack-2017-pred.txt

python bin/infer.py \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${OUTFILE}
