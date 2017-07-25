#!/bin/bash

VOCAB_SOURCE=charmaps/en.tsv
VOCAB_TARGET=charmaps/es.tsv

TRAIN_SOURCES_PREFIX=wp-data/mixed-nmt-wp/en2es.en.rnd.txt.segment
TRAIN_TARGETS_PREFIX=wp-data/mixed-nmt-wp/en2es.es.rnd.txt.segment

DEV_SOURCES=wp-data/wponly-processed/jetpack-es-source.txt
DEV_TARGETS=wp-data/wponly-processed/jetpack-es-target.txt

TRAIN_STEPS=500000
EVAL_STEPS=200000000
BATCH_SIZE=32

MODEL_DIR=models/en2es-goldilocks

while true
do

#for i in 17 18 19 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16;
#  do
    #TRAIN_SOURCES="$TRAIN_SOURCES_PREFIX$i"
    #TRAIN_TARGETS="$TRAIN_TARGETS_PREFIX$i"
		TRAIN_SOURCES=wp-data/wponly-processed/wpcom-es-source.txt
		TRAIN_TARGETS=wp-data/wponly-processed/wpcom-es-target.txt

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
      --batch_size $BATCH_SIZE \
      --train_steps $TRAIN_STEPS \
      --eval_every_n_steps $EVAL_STEPS \
      --output_dir $MODEL_DIR 2>&1 | tee -a training.log \
		|| { echo 'failed' ; exit 1; }

#  done
done
