# WP Translate

Attempting to translate WP strings with a character based sequence to sequence neural network.

Mostly built on https://github.com/farizrahman4u/seq2seq with the main idea coming from http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Experimental


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
