# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
from itertools import izip

################
#
# Take text files, and output an encoded file

if ( len(sys.argv) < 3 ):
    sys.exit( 'charmap, target file, source file, target out, source out' )

in_charmap_file=sys.argv[1]
out_charmap_file=sys.argv[2]
src_file=sys.argv[3]
tar_file=sys.argv[4]
src_out_file=sys.argv[5]
tar_out_file=sys.argv[6]

MAXLENGTH = 500  #max length of input text

intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Filtering...')

src_out_fh = open( src_out_file, 'w' )
tar_out_fh = open( tar_out_file, 'w' )

for src_ln, tar_ln in izip(open(src_file), open(tar_file)):
    if (len(src_ln) > MAXLENGTH):
        continue
    if (len(tar_ln) > MAXLENGTH):
        continue

    try:
	s = intable.encode(src_ln)
	s = outtable.encode(tar_ln)
    except:
        continue

    src_out_fh.write( src_ln )
    tar_out_fh.write( tar_ln )

src_out_fh.close()
tar_out_fh.close()

print('Done.')
