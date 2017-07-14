#!/bin/bash

VOCAB_SOURCE=charmaps/en.tsv
VOCAB_TARGET=charmaps/es.tsv

#WP Data
WP_TRAIN_SOURCES=wp-data/wponly-processed/wpcom-es-source.txt
WP_TRAIN_TARGETS=wp-data/wponly-processed/wpcom-es-target.txt

#common crawl data
NMT_TRAIN_SOURCES=wp-data/wp-nmt-processed/commoncrawl.es-en.en.txt
NMT_TRAIN_TARGETS=wp-data/wp-nmt-processed/commoncrawl.es-en.es.txt

TRAIN_STEPS=10000

MODEL_DIR=models/en2es

while true
do

echo "TrainingNMTData" >> training.log

python bin/train.py \
  --config_paths="
      ./wp-translate-model.yml,
      ./train_seq2seq.yml,
      ./text_metrics.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $NMT_TRAIN_SOURCES
      target_files:
        - $NMT_TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 20 \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR 2>&1 | tee -a training.log


echo "TrainingWPData" >> training.log

python bin/train.py \
  --config_paths="
      ./wp-translate-model.yml,
      ./train_seq2seq.yml,
      ./text_metrics.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $WP_TRAIN_SOURCES
      target_files:
        - $WP_TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 20 \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR 2>&1 | tee -a training.log


done
