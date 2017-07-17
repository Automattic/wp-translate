#!/bin/bash


TRAIN_SOURCES_PREFIX=wp-data/mixed-nmt-wp/en2es.en.rnd.txt.segment
TRAIN_TARGETS_PREFIX=wp-data/mixed-nmt-wp/en2es.es.rnd.txt.segment

TRAIN_WP_SOURCE=wp-data/wponly-processed/wpcom-es-source.txt
TRAIN_WP_TARGET=wp-data/wponly-processed/wpcom-es-target.txt

for i in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19;
	do
		TRAIN_SOURCES="$TRAIN_SOURCES_PREFIX$i"
		TRAIN_TARGETS="$TRAIN_TARGETS_PREFIX$i"

		cat $TRAIN_WP_SOURCE >> $TRAIN_SOURCES
		cat $TRAIN_WP_TARGET >> $TRAIN_TARGETS
done
