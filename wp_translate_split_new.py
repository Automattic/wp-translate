# -*- coding: utf-8 -*-
from __future__ import print_function
import wp_translate_util as util
import sys

if ( len(sys.argv) < 2 ):
    sys.exit( 'need to specify old .po file, and new .po file' )

old_po_file=sys.argv[1]
new_po_file=sys.argv[2]

old_po_data = util.load_translated_po_data( old_po_file )
new_po_data = util.load_translated_po_data( new_po_file )

old_entries = {}
new_entries = {}

for t in old_po_data:
    old_entries[t.msgid] = t.msgstr

for t in new_po_data:
    if t.msgid not in old_entries:
        new_entries[t.msgid] = t.msgstr

for i,s in new_entries.iteritems():
    print "\n"
    print "msgid \"" + i + "\"\n"
    print "msgstr \"" + s + "\"\n"
    print "\n"
