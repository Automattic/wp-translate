# WP Translate

Attempting to translate WP strings with a character based sequence to sequence neural network.

Mostly built on https://github.com/farizrahman4u/seq2seq with the main idea coming from http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Experimental


## New running

Set up the data:

```
 > python wp_translate_po2ptf.py wp-data/2015/wpcom-es.po charmaps/en.tsv charmaps/es.tsv wp-data/processed/wpcom-es-source.txt wp-data/processed/wpcom-es-target.txt
 > python wp_translate_ptf_decode.py charmaps/en.tsv wp-data/processed/wpcom-es-source.txt tmp.txt
 ```


## Basic Running

Still changing rapidly, but this is what I am playing with

Train a Spanish model on Jetpack 3.5 translation data:

```
python wp_translate_train.py wp-data/2015/jetpack-3.5-es.po charmaps/en.tsv charmaps/es.tsv models/en2es/en2es
```

Evaluate a model:

```
python wp_translate_eval.py models/en2es/en2es.yml models/en2es/en2es_99_0.000715292000677.h5 charmaps/en.tsv charmaps/es.tsv wp-data/2015/wpcom-es.po
```

Create an English character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgid charmaps/en.tsv
```

Create a Spanish character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgstr charmaps/es.tsv
```


## Basic AWS Setup

Training on an AWS GPU (p2.xlarge) looks like it is at least 150x faster than running on my circa 2015 Macbook Pro.

For setting up AWS I used the instructions and config setup from the Fast AI class: http://wiki.fast.ai/index.php/AWS_install
