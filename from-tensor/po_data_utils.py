"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from tensorflow.python.platform import gfile
from six.moves import urllib

import polib

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def create_char_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  """Create vocabulary file (if it does not exist yet) from data file.

  Create a vocabulary from a file. The vocab consists of unique characters.
  We include all the words in the file because we are lazy.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        for character in stru.decode("utf-8"):
          if character in vocab:
            vocab[character] += 1
          else:
            vocab[character] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        print("WARNING - Truncating vocab of size %d" % len(vocab_list))
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def translation_to_token_ids(string, vocabulary):
  """Convert a string to list of integers representing token-ids.
  Tokenization is entirely character based right now.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    string: a string to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  return [vocabulary.get(c, UNK_ID) for c in string.decode("utf-8")]


def po_data_to_token_ids(data_path, target_path, vocabulary_path, target_lang):
  """Tokenize po data file and turn into token-ids using given vocabulary file.

  This function takes a .po file, loads it, and converts the strings into token
  based on the vocabulary file. Converting to ints makes it easier to avoid newlines
  and other weird chars. May also help us if we try and be smarter in the future.
  Saves the result to target_path.

  Args:
    data_path: path to the po data file
    target_path: path where the file with token-ids will be created with one translation per line.
    vocabulary_path: path to the vocabulary file.
    target_lang: the language to pull from the .po file (either 'en' or something else)
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    po = polib.pofile(data_path)
    with gfile.GFile(target_path, mode="w") as tokens_file:
      counter = 0
      for entry in po:
        counter += 1
        if counter % 100000 == 0:
          print("  tokenizing line %d" % counter)
        if ( '' == entry.msgstr ):
          continue
        if ( 'en' == target_lang ):
          string = entry.msgid
        else:
          string = entry.msgstr
        token_ids = translation_to_token_ids(string, vocab)
        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_po_data(training_po, dev_po, data_dir, vocabulary_size, target_lang):
  """Get .po data into data_dir, create vocabularies and tokenize data.

  Args:
    training_po: po file with the training data
    dev_po: po file for testing
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the vocabulary to create and use.
    target_lang: the target language code

  Returns:
    A tuple of 3 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for target training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for target development data-set,
      (5) path to the English vocabulary file,
      (6) path to the target vocabluary file.
  """

  # Create vocabularies of the appropriate sizes.
  target_vocab_path = os.path.join(data_dir, "vocab%d.%s" % vocabulary_size, target_lang)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % vocabulary_size)
  create_vocabulary(target_vocab_path, train_path + ".%s" % target_lang, vocabulary_size)
  create_vocabulary(en_vocab_path, train_path + ".en", vocabulary_size)

  # Create token ids for the training data.
  target_train_ids_path = train_path + (".ids%d.%s" % vocabulary_size, target_lang)
  en_train_ids_path = train_path + (".ids%d.en" % vocabulary_size)
  po_data_to_token_ids(train_path + ".%s" % target_lang, target_train_ids_path, target_vocab_path)
  po_data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path)

  # Create token ids for the development data.
  target_dev_ids_path = dev_path + (".ids%d.%s" % vocabulary_size, target_lang)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  po_data_to_token_ids(dev_path + ".%s" % target_lang, target_dev_ids_path, target_vocab_path)
  po_data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path)

  return (en_train_ids_path, target_train_ids_path,
          en_dev_ids_path, target_dev_ids_path,
          en_vocab_path, target_vocab_path)
