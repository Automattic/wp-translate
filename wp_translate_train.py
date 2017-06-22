# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from seq2seq.models import AttentionSeq2Seq

if ( len(sys.argv) < 1 ):
    sys.exit( 'need to specify .po file' )

po_file=sys.argv[1]

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True

HIDDEN_SIZE = 1024
BATCH_SIZE = 128
LAYERS = 2
MAXCHAR = 0x30C5 #ugh so much to cover
MAXLENGTH = 500  #max length of input text

ctable = util.UTF8CharacterTable(MAXCHAR,MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

print('Vectorization...')
x = np.zeros((len(po_data), MAXLENGTH, MAXCHAR), dtype=np.bool)
y = np.zeros((len(po_data), MAXLENGTH, MAXCHAR), dtype=np.bool)
for i, t in enumerate(po_data):
    x[i] = ctable.encode(t.msgid)
    y[i] = ctable.encode(t.msgstr)

x = np.array( x, dtype=np.bool )
y = np.array( y, dtype=np.bool )

print(x.shape)
print(y.shape)

print('Build model...')

model = SimpleSeq2Seq(input_dim=MAXCHAR, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=MAXCHAR, depth=LAYERS)

#much more intricate model
#model = AttentionSeq2Seq(input_dim=MAXCHAR, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=MAXCHAR, depth=LAYERS)

model.compile(loss='mse', optimizer='rmsprop')

for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=128, epochs=100, validation_split=.05, callbacks=[ModelCheckpoint('/output/weights_{epoch}_{val_loss}.h5')])

