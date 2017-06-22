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

if ( len(sys.argv) < 3 ):
    sys.exit( 'need to specify .po file, in charmap, out charmap' )

po_file=sys.argv[1]
in_charmap_file=sys.argv[2]
out_charmap_file=sys.argv[3]

################################3
# Current limitations
# - 500 char max length
# - 


# Parameters for the model
HIDDEN_SIZE = 1024
LAYERS = 2
MAXLENGTH = 500  #max length of input text

# Training
BATCH_SIZE = 128
ITERATINOS=60
EPOCHS=100

intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

print('Vectorization...')
x = np.zeros((len(po_data), MAXLENGTH, intable.maxval), dtype=np.bool)
y = np.zeros((len(po_data), MAXLENGTH, outtable.maxval), dtype=np.bool)
for i, t in enumerate(po_data):
    x[i] = intable.encode(t.msgid)
    y[i] = outtable.encode(t.msgstr)

x = np.array( x, dtype=np.bool )
y = np.array( y, dtype=np.bool )

print(x.shape)
print(y.shape)

print('Build model...')

model = SimpleSeq2Seq(input_dim=intable.maxval, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=outtable.maxval, depth=LAYERS)

#much more intricate model
#model = AttentionSeq2Seq(input_dim=intable.maxval, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=outtable.maxval, depth=LAYERS)

model.compile(loss='mse', optimizer='rmsprop')

for iteration in range(1, ITERATIONS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.05, callbacks=[ModelCheckpoint('/output/weights_{epoch}_{val_loss}.h5')])

