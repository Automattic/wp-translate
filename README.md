# WP Translate

Atteempting to translate WP strings with a character based sequence to sequence neural network.

Experimental


## Basic Running

Still changing rapidly, but this is what I am playing with

Train a Spanish model on Jetpack 3.5 translation data:

```
python wp_translate_train.py wp-data/2015jetpack-3.5-es.po charmaps/en.tsv charmaps/es.tsv en2es
```

Test a model (untested code):

```
python wp_translate_predict.py ???
```

Create an English character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgid charmaps/en.tsv
```

Create a Spanish character mapping (encoding):

```
python wp_translate_charmap.py wp-data/all-es.po msgstr charmaps/es.tsv
```
