# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
import difflib

################
#
# Take encoded text file, and compare it to the expected output

if ( len(sys.argv) < 3 ):
    sys.exit( 'charmap, encoded file, exp file, err file' )

charmap_file=sys.argv[1]
enc_file=sys.argv[2]
exp_file=sys.argv[3]
err_file=sys.argv[4]

MAXLENGTH = 500  #max length of input text

table = util.EncodedCharacterTable(charmap_file, MAXLENGTH)

print('Load Data...')

with open(enc_file, 'r') as f:
    pred_lines = f.readlines()

with open(exp_file, 'r') as f:
    exp_lines = f.readlines()

fh = open( err_file, 'w' )

total = 0
correct = 0

print('Comparing...')
for pred_s, exp_s in zip( pred_lines, exp_lines ):
    pred_text = table.decode_from_string(pred_s)
    exp_text = table.decode_from_string(exp_s)
    total = total + 1
    if ( pred_text == exp_text ):
        correct = correct + 1
    else:
        diff = difflib.ndiff(exp_text, pred_text)
        fh.write( ''.join(diff) + "\n\n" )

fh.close()

print('Accuracy: ' + str(correct/total) + '% (' + str(correct) + '/' + str(total) + ')' )
