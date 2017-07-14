# -*- coding: utf-8 -*-
from __future__ import print_function
from six.moves import range
import wp_translate_util as util
import sys

if ( len(sys.argv) < 2 ):
    sys.exit( 'need to specify input file, charmap file ' )

in_file=sys.argv[1]
charmap_file=sys.argv[2]

util.build_text_char_map( in_file, charmap_file )
