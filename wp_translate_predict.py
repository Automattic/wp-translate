# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
import seq2seq
from keras.models import model_from_yaml
from seq2seq.models import SimpleSeq2Seq
from seq2seq.models import AttentionSeq2Seq

if ( len(sys.argv) < 5 ):
    sys.exit( 'need to specify model, weights, in charmap, out charmap, .po test file, ' )

model_file=sys.argv[1]
weights_file=sys.argv[2]
in_charmap_file=sys.argv[3]
out_charmap_file=sys.argv[4]
po_file=sys.argv[5]

# Parameters for the model and dataset
MAXLENGTH = 500  #max length of input text

print('Load Charmaps...')
intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

#TODO remove
po_data = po_data[0:100]

print('Load model...')

# not loading sequential model for some reason
# so for now let's just build it and assume is the same as when we trained
#f = open(model_file)
#model = model_from_yaml(f.read())

model = util.create_model( intable.maxval, outtable.maxval )

model.load_weights(weights_file)
model.compile(loss='mse', optimizer='rmsprop')

print('Vectorization...')
x = np.zeros((len(po_data), MAXLENGTH, intable.maxval), dtype=np.bool)
y_exp = [None]*len(po_data)
y_act = [None]*len(po_data)
for i, t in enumerate(po_data):
    x[i] = intable.encode(t.msgid)
    y_exp[i] = t.msgstr

x = np.array( x, dtype=np.bool )

print(x.shape)

print('Run model...')

y = model.predict(x, verbose=1)
for i,s in enumerate(y):
    print(s)
    y_act[i] = outtable.decode( s )

sys.exit()
print('Compare...')

total = len(y)
correct = 0

#TODO for generating diffs of strings, looks like github.com/samg/diffy will do the trick
# or otherwise look at import difflib


for i in range(1, total):
    if ( y_exp[i] == y_act[i] ):
        #print util.colors.ok + u"2714" + util.colors.close + " " + y_act[i] + "\n"
        print(y_act[i])
        correct+=1
    else:
        #print util.colors.fail + "X" + util.colors.close + " " + y_exp[i] + "\n"
        #print "  " + y_act[i] + "\n"
        print(y_act[i])
    #print "\n"

#print "\n"
#print str(correct) + '/' + str(total) + "  (" + str(correct/total) + "%)\n\n"

