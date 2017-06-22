# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
import seq2seq
from keras.callbacks import ModelCheckpoint
from seq2seq.models import SimpleSeq2Seq
from seq2seq.models import AttentionSeq2Seq

if ( len(sys.argv) < 4 ):
    sys.exit( 'need to specify model, in charmap, out charmap, .po test file, ' )

model_file=sys.argv[1]
in_charmap_file=sys.argv[2]
out_charmap_file=sys.argv[3]
po_file=sys.argv[4]


# Parameters for the model and dataset
MAXLENGTH = 500  #max length of input text

print('Load Charmaps...')
intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

print('Load model...')

f = open(model_file)
model = model_from_yaml(f.read())
model.compile(loss='mse', optimizer='rmsprop')

print('Vectorization...')
x = np.zeros((len(po_data), MAXLENGTH, intable.maxval), dtype=np.bool)
y_exp = []
y_act = []
for i, t in enumerate(po_data):
    x[i] = intable.encode(t.msgid)
    y_exp[i] = t.msgstr

x = np.array( x, dtype=np.bool )

print(x.shape)

print('Run model...')

for i,s in x:
    y = model.predict(s, verbose=0)
    y_act[i] = outtable.decode( y )

print('Compare...')

total = count(y)
correct = 0

#TODO for generating diffs of strings, looks like github.com/samg/diffy will do the trick
# or otherwise look at import difflib


for i in range(1, total):
    if ( y_exp[i] == y_act[i] ):
        print colors.ok + u"2714" + colors.close + " " + y_act[i] + "\n"
        correct+=1
    else:
        print colors.fail + "X" + colors.close + " " + y_exp[i] + "\n"
        print "  " + y_act[i] + "\n"
    print "\n"

print "\n"
print str(correct) + '/' + str(total) + "  (" + str(correct/total) + "%)\n\n"

