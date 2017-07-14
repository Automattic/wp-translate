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

We need to encode our text data so that it can go into the model. The desired output is two parallel text files, one for the source language (always English in our case) and one for the target language. The lines of each file align with each other. To be able to efficiently encode all characters, we build a character mapping which maps an index to a unicode data point (theoretically we could map sequences of characters also).

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

### Prep the training data


```
python wp_translate_ptf_filter.py charmaps/en.tsv charmaps/es.tsv nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.en nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.es nmt-data/wmt16/commoncrawl.es-en.en nmt-data/wmt16/commoncrawl.es-en.es
python wp_translate_ptf_encode.py charmaps/en.tsv nmt-data/wmt16/commoncrawl.es-en.en wp-data/wp-nmt-processed/commoncrawl.es-en.en.txt
python wp_translate_ptf_encode.py charmaps/es.tsv nmt-data/wmt16/commoncrawl.es-en.es wp-data/wp-nmt-processed/commoncrawl.es-en.es.txt
```

Create the parallel text formatted files that contain the encoded data.

Encode the common crawl generic data.

```
> python wp_translate_ptf_encode.py charmaps/all-en.tsv nmt-data/wmt16/raw/common-crawl/commoncrawl.es-en.en wp-data/wp-nmt-processed/commoncrawl.es-en.en.txt
```

Encode the WP data (and then decode to check it):


```
> python wp_translate_po2ptf.py wp-data/2015/wpcom-es.po charmaps/en.tsv charmaps/es.tsv wp-data/processed/wpcom-es-source.txt wp-data/processed/wpcom-es-target.txt
 > python wp_translate_ptf_decode.py charmaps/en.tsv wp-data/processed/wpcom-es-source.txt tmp.txt
 ```

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
python wp_translate_ptf_decode.py charmaps/es.tsv models/en2es/pred/predictions.txt tmp.txt
```

### Evaluate results

TBD


## Notes from initial training

Started with en2es translation
- Trained on wpcom data for 100k steps or so (13k examples)
 - saw some Spanish on the output, but was worried that we didn't have a big enough vocab in the training data
- Started training on a subset of the common crawl corpus data (removed lines with characters not in our charmap) to get a wider vocab (300k examples)
- alternated 10k steps on common crawl data, then 10k steps on wp data. Didn't want 300k examples to overwhelm the 13k. (yes this is very hacky and ad hoc)
