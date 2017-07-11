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


### Set up the data

Create an English character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgid charmaps/en.tsv
```

Create a Spanish character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgstr charmaps/es.tsv
```

Create the parallel text formatted files that contain the encoded data.

```
 > python wp_translate_po2ptf.py wp-data/2015/wpcom-es.po charmaps/en.tsv charmaps/es.tsv wp-data/processed/wpcom-es-source.txt wp-data/processed/wpcom-es-target.txt
 > python wp_translate_ptf_decode.py charmaps/en.tsv wp-data/processed/wpcom-es-source.txt tmp.txt
 ```

### Run the training

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
