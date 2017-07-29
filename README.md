# WP Translate

Attempting to translate WP strings with a character based sequence to sequence neural network.

Mostly built on https://github.com/farizrahman4u/seq2seq with the main idea coming from http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Experimental


# Setup

For running on AWS and using GPUs (p2.xlarge is about 150x faster than my macbook pro) I used the instructions and config setup from the Fast AI class: http://wiki.fast.ai/index.php/AWS_install

Setup from there:

```
sudo apt-get install libcupti-dev
git clone git@github.com:Automattic/wp-translate.git
cd wp-translate
pip install -r requirements.txt
cd ..
git clone git@github.com:google/seq2seq.git tf-seq2seq
cd tf-seq2seq
```

We need to work around an prevent matplotlib from causing some failures (from the seq2seq instructions):

```
echo "backend : Agg" >> $HOME/.config/matplotlib/matplotlibrc
```

Add to your LD_LIBRARY_PATH:

```
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

We need to apply a patch to seq2seq to get it working with TF 1.2 https://github.com/google/seq2seq/pull/254

Kinda hacky, probably a better way...

Edit the tf-seq2seq/.git/config and add this like to the origin:

```
 fetch = +refs/pull/*/head:refs/remotes/origin/pr/*
```

Now you can build/install with the necessary PR with:

```
git fetch origin
git checkout pr/254
pip install -e .
```

To test the seq2seq installation run:

```
python -m unittest seq2seq.test.pipeline_test
```

## How to run

We need to encode our text data so that it can go into the model. The desired output is two parallel text files, one for the source language (always English in our case) and one for the target language. The lines of each file align with each other. To be able to efficiently encode all characters, we build a character mapping which maps an index to a unicode data point (theoretically we could map to sequences of characters also there are some tools in the seq2seq lib for doing this).

So to build training data, we need a charmap for the target and source languages, then we use them to build the parallel text files which will just consist of the index numbers for the characters separated by whitespace.

### Build the charmaps

Create English/Spanish character mapping (encoding) from a .po file:

```
python wp_translate_charmap.py wp-data/all-es.po msgid charmaps/wp-en.tsv
python wp_translate_charmap.py wp-data/all-es.po msgstr charmaps/wp-es.tsv
```

We also train on generic translation data, so we need a charmap that works across all our data.

Create an English/Spanish character mapping (encoding) from a text file:

```
python wp_translate_charmap_text.py nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.en charmaps/nmt-en.tsv
python wp_translate_charmap_text.py nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.es charmaps/nmt-es.tsv
```

The different charmaps will not perfectly overlap, so we can hackily merge them with something like:

```
cut -f 2 charmaps/wp-en.tsv | sort -n > tmp.chars
cut -f 2 charmaps/nmt-en.tsv | sort -n > tmp2.chars
diff tmp.chars tmp2.chars | grep -e "^<" >> charmaps/all-en.tsv
```

And then edit the last couple of lines of all-en.tsv to give those missing characters some indices. There should only be a small handful of missing characters.

### Data organization

```
/
- charmaps - the list of
- wp-data
 - 2015 - downloaded wpcom and jetpack translation files from 2015
 - 2017 - downloaded translation files from 2017
 - wponly-processed - preprocessed/encoded training files from the 2015 and 2017 dirs
- models
  - goldilocks - the trained model for en to es translation
- predictions - output files from
```

### Prep the training data

Creating the generic training data from common crawl corpus:

```
python wp_translate_ptf_filter.py charmaps/en.tsv charmaps/es.tsv nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.en nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.es nmt-data/wmt16/commoncrawl.es-en.en nmt-data/wmt16/commoncrawl.es-en.es
python wp_translate_ptf_encode.py charmaps/en.tsv nmt-data/wmt16/commoncrawl.es-en.en wp-data/wp-nmt-processed/commoncrawl.es-en.en.txt
python wp_translate_ptf_encode.py charmaps/es.tsv nmt-data/wmt16/commoncrawl.es-en.es wp-data/wp-nmt-processed/commoncrawl.es-en.es.txt
```

When using generic data, we want training sets that are balanced 50-50 between the wp data and a subset of the generic data. This way we can generalize better and make up for lack of vocab in the wp data. Putting them in the same file means that we will have mini-batches that contain both sets of data.

Take the generic NMT training data and randomize it, split it up, and then append the wp data onto it:

```
wc -l nmt-data/wmt16/commoncrawl.es-en.en  #get the number of lines total
seq 1 316368 | shuf > seq.num
paste seq.num nmt-data/wmt16/commoncrawl.es-en.en | sort | sed "s/^[0-9]*\s//" | head -n 260000 > wp-data/mixed-nmt-wp/en2es.en.rnd
paste seq.num nmt-data/wmt16/commoncrawl.es-en.es | sort | sed "s/^[0-9]*\s//" | head -n 260000 > wp-data/mixed-nmt-wp/en2es.es.rnd

python wp_translate_ptf_encode.py charmaps/en.tsv wp-data/mixed-nmt-wp/en2es.en.rnd wp-data/mixed-nmt-wp/en2es.en.rnd.txt
python wp_translate_ptf_encode.py charmaps/es.tsv wp-data/mixed-nmt-wp/en2es.es.rnd wp-data/mixed-nmt-wp/en2es.es.rnd.txt

split -l 13000 -d wp-data/mixed-nmt-wp/en2es.en.rnd.txt wp-data/mixed-nmt-wp/en2es.en.rnd.txt.segment
split -l 13000 -d wp-data/mixed-nmt-wp/en2es.es.rnd.txt wp-data/mixed-nmt-wp/en2es.es.rnd.txt.segment
./append-data-segs.sh
```

That should generate 20 segments of training data that can be cycled through during training.

Create the parallel text formatted files that contain the encoded data.

Encode the common crawl generic data.

```
> python wp_translate_ptf_encode.py charmaps/all-en.tsv nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.en wp-data/wp-nmt-processed/commoncrawl.es-en.en.txt
```

Encode the WP data (and then decode to check it):

```
> python wp_translate_po2ptf.py wp-data/2015/wpcom-es.po charmaps/en.tsv charmaps/es.tsv wp-data/wponly-processed/wpcom-es-source.txt wp-data/wponly-processed/wpcom-es-target.txt
 > python wp_translate_ptf_decode.py charmaps/en.tsv wp-data/processed/wpcom-es-source.txt tmp.txt
 ```

This same final step can get used for generating data for evaluation also.

### Run the training

Our training script will alternatively run on the generic data (to learn the language) and on the WP data to learn the unique WP structure.

```
./wp-translate-train.sh
```

Verify you are using the GPU:

```
nvidia-smi
```


### Infer results

Predict results:

```
./wp-translate-pred.sh
```

Decode the results from the output

```
cp models/en2es/pred/predictions.txt predictions/jetpack-2015.enc
python wp_translate_ptf_decode.py charmaps/es.tsv predictions/jetpack-2015.enc predictions/jetpack-2015.txt
```

### Evaluate results

```
python wp_translate_eval.py charmaps/es.tsv predictions/jetpack-2015.enc wp-data/wponly-processed/jetpack-es-target.txt predictions/jetpack-2015-output.diff
```


## Notes from training prototype model

The prototype model consists of:
- a 3 layer bidirectional encoder
- a 3 layer attention decoder
- 512 units in the internal layers
- LTSM cells
- max character length of 500 chars

Trained to translate from en data to es_ES

It was trained on:
- 100k steps of combo data from common crawl data and wpcom data from 2015
- 120k steps of just the wpcom data from 2015

Here is a sampling of the loss across those steps:

![Training Loss](https://raw.githubusercontent.com/automattic/wp-translate/master/trainingloss.png)

The large decrease is when we switched from the joint training data to only using wpcom data.

## Notes from evaluating prototype model

We ran predictions for the following data:
- jetpack 2015 strings - 1655 strings
- jetpack 2017 strings - 2541 strings
- vantage theme 2017 strings - 573 strings
- yoast strings - 1092 strings
- wpcom 2017 strings - 17874 strings (note that this is a superset of the 13226 strings we trained on)

All predictions were run using beam search with a width of 5.

We evaluated the results based on what percentage of the strings were 100% translated correctly character for character.

| Project        | Exact Matches | Off by < 4 chars |
| -------------- |:-------------:| ----------------:|
| jetpack 2015   | 54.32% (899)  | 5.08%            |
| jetpack 2017   | 36.36% (924)  | 3.00%            |
| yoast plugin   | 06.59% (72)   | 2.66%            |
| vantage theme  | 12.04% (69)   | 7.85%            |
| wpcom 2017     | 49.75% (8893) | 2.90%            |


The percent off by less than 4 chars is determined using:

```
grep "^?" predictions/jetpack-2015-output.diff | tr -d '? ' | awk 'length($1) < 4 { print $1 }' | wc -l
```


## Ideas for Improvements

- Switch from one char per byte pair encoding and handle unknown words
  - https://google.github.io/seq2seq/nmt/#data-format and https://arxiv.org/abs/1508.07909
  - https://google.github.io/seq2seq/inference/#unk-token-replacement-using-a-copy-mechanism
- train on all the data that exists for all themes, plugins, projects
  - this gives us more data to train on, but will make it harder to evaluate
- calculate some sort of confidence score (maybe based on beam search data)
  - this can then be a cutoff where we only accept translations that we are pretty certain about
- use a more complex model
  - Google translate uses a much larger network: https://research.googleblog.com/2016/09/a-neural-network-for-machine.html
  - this probably requires getting the model to train across multiple gpus: https://github.com/google/seq2seq/issues/44
