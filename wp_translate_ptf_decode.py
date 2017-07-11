# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys

################
#
# Take encoded text files, and output the real strings

if ( len(sys.argv) < 3 ):
    sys.exit( 'charmap, encoded file, out file' )

charmap_file=sys.argv[1]
enc_file=sys.argv[2]
out_file=sys.argv[3]

MAXLENGTH = 500  #max length of input text

table = util.EncodedCharacterTable(charmap_file, MAXLENGTH)

print('Load Data...')

with open(enc_file, 'r') as f:
    lines = f.readlines()

fh = open( out_file, 'w' )

print('Decoding...')
for i, s in enumerate(lines):
    text = table.decode_from_string(s)
    fh.write( text + "\n\n" )

fh.close()

print('Done.')
