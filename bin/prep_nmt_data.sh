#! /usr/bin/env bash

# Download generic translation data for augmenting the WordPress
# strings. This should increase the size of our vocabulary

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

OUTPUT_DIR=nmt-data/wmt16

OUTPUT_DIR_DATA="${OUTPUT_DIR}/raw"

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/europarl-v7-es-en.tgz \
  http://www.statmt.org/europarl/v7/es-en.tgz

#echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

#echo "Downloading News Commentary v11. This may take a while..."
#wget -nc -nv -O ${OUTPUT_DIR_DATA}/nc-v11.tgz \
#  http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
#
#echo "Downloading dev/test sets"
#wget -nc -nv -O  ${OUTPUT_DIR_DATA}/dev.tgz \
#  http://data.statmt.org/wmt16/translation-task/dev.tgz
#wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz \
#  http://data.statmt.org/wmt16/translation-task/test.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-es-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-es-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-es-en"

mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
#mkdir -p "${OUTPUT_DIR_DATA}/nc-v11"
#tar -xvzf "${OUTPUT_DIR_DATA}/nc-v11.tgz" -C "${OUTPUT_DIR_DATA}/nc-v11"
#mkdir -p "${OUTPUT_DIR_DATA}/dev"
#tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
#mkdir -p "${OUTPUT_DIR_DATA}/test"
#tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.es" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.es" \
  > "${OUTPUT_DIR}/train.de"
wc -l "${OUTPUT_DIR}/train.de"

echo "All done."
