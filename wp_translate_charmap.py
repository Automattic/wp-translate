# -*- coding: utf-8 -*-
from __future__ import print_function
from six.moves import range
import wp_translate_util as util
import sys

if ( len(sys.argv) < 1 ):
    sys.exit( 'need to specify .po file, string_type, charmap file ' )

po_file=sys.argv[1]
str_type=sys.argv[2]
charmap_file=sys.argv[3]

util.build_encoded_char_map( po_file, str_type, charmap_file )
