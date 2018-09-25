# -*- coding: utf-8 -*-
from __future__ import print_function
from six.moves import range
import polib as polib
import sys
from google.cloud import automl_v1beta1

import difflib
from copy import deepcopy
import time

model_ids = { "es": "TRL113781196667304762" }
project_id = 'machine-translation-api-216900'

# we want to send batches to the translate api
def batch_gen(data, batch_size):
    for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

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

model_id = model_ids[lang] or sys.exit( 'no model for ' + lang)

print('Load Data...')
po = polib.pofile(po_file)

prediction_client = automl_v1beta1.PredictionServiceClient()
name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
params = {}
def get_prediction(content):
    payload = {'text_snippet': {'content': content}}
    request = prediction_client.predict(name, payload, params)
    return request.payload[0].translation.translated_content.content

total = 0
correct = 0
untranslated = 0
cnt = 0

#Prep output files
out_po = polib.POFile()
out_po.metadata = po.metadata

dfh = open( diff_file, 'w' )

entry = polib.POEntry(
    msgid=u'Welcome',
    msgstr=u'Bienvenue',
)
# po.append(entry)

print('Translating...')

for entry in po:
    if(cnt > 10):
        break
    if ( cnt % 100 == 0 ):
        print( total )
    t = None
    while t == None:
        try:
            t = get_prediction(entry.msgid)
        except:
            print('Translation errror. pausing.', sys.exc_info()[0])
            time.sleep(105)

    cnt = cnt + 1
    out_entry = deepcopy(entry)
    out_entry.msgstr=t
    out_po.append(out_entry)

    if ( entry.msgstr == '' ):
        untranslated = untranslated + 1
    else:
        total = total + 1
        if ( t == entry.msgstr ):
            correct = correct + 1
        else:
            diff = difflib.ndiff([entry.msgstr], [t])
            dfh.write( u"\n".join(diff) + u"\n\n" )

print('Done! Saving to ' + out_file)
out_po.save(out_file)

dfh.close()

perc = float(correct)/float(total) * 100
print('Accuracy: ' + '%.2f' % perc + '% (' + str(correct) + '/' + str(total) + ')' )
print('Newly Translated Strings: ' + str(untranslated) )

