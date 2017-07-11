# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys

################
#
# Take a .po translation file and encode it and convert it to parallel text format
# because we are encoding all chars, we need to output individual numbers

if ( len(sys.argv) < 5 ):
    sys.exit( 'need to specify .po file, in charmap, out charmap, in file, out file' )

po_file=sys.argv[1]
in_charmap_file=sys.argv[2]
out_charmap_file=sys.argv[3]
in_file=sys.argv[4]
out_file=sys.argv[5]

MAXLENGTH = 500  #max length of input text

print('Load Charmaps...')
intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

ifh = open( in_file, 'w' )
ofh = open( out_file, 'w' )

print('Generating...')
for i, t in enumerate(po_data):
    ifh.write( intable.encode_to_string(t.msgid) + "\n" )
    ofh.write( outtable.encode_to_string(t.msgstr) + "\n" )

ifh.close()
ofh.close()

print('Done.')
