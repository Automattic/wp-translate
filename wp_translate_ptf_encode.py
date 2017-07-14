# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys

################
#
# Take text files, and output an encoded file

if ( len(sys.argv) < 3 ):
    sys.exit( 'charmap, ptf file, encoded file' )

charmap_file=sys.argv[1]
in_file=sys.argv[2]
enc_file=sys.argv[3]

MAXLENGTH = 500  #max length of input text

table = util.EncodedCharacterTable(charmap_file, MAXLENGTH)

print('Load Data...')

with open(in_file, 'r') as f:
    lines = f.readlines()

fh = open( enc_file, 'w' )

print('Encoding...')
for i, s in enumerate(lines):
    text = table.encode_to_string(s.decode('utf-8'))
    fh.write( text + "\n" )

fh.close()

print('Done.')
