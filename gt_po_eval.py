# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import polib as polib
import sys
from google.cloud import translate
import difflib
from copy import deepcopy

################
#
# Take a .po translation file, run it through google translate api, and then
# evaluate the translations that we know about and create a completely translated .po file

if ( len(sys.argv) < 5 ):
    sys.exit( 'need to specify .po file, lang, out po file, diff file' )

po_file=sys.argv[1]
lang=sys.argv[2]
out_file=sys.argv[3]
diff_file=sys.argv[4]

print('Load Data...')
po = polib.pofile(po_file)

translate_client = translate.Client()
total = 0
correct = 0
untranslated = 0

#Prep output files
out_po = polib.POFile()
out_po.metadata = po.metadata

dfh = open( diff_file, 'w' )

entry = polib.POEntry(
    msgid=u'Welcome',
    msgstr=u'Bienvenue',
)
po.append(entry)

print('Translating...')

for entry in po:
    translation = translate_client.translate(
        entry.msgid,
        source_language='en',
        target_language=lang,
        model='nmt'
    )

    out_entry = deepcopy(entry)
    out_entry.msgstr=translation['translatedText']
    out_po.append(out_entry)

    if ( entry.msgstr == '' ):
    	untranslated = untranslated + 1
    else:
        total = total + 1
    	if ( translation['translatedText'] == entry.msgstr ):
            correct = correct + 1
    	else:
            diff = difflib.ndiff([entry.msgstr], [translation['translatedText']])
            dfh.write( u"\n".join(diff).encode('utf-8') + "\n\n" )

po.save(out_file)

dfh.close()

perc = float(correct)/float(total) * 100
print('Accuracy: ' + '%.2f' % perc + '% (' + str(correct) + '/' + str(total) + ')' )
print('Newly Translated Strings: ' + str(untranslated) )

