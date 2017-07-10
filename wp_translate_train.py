# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range
import wp_translate_util as util
import sys
import seq2seq
from keras.callbacks import ModelCheckpoint

if ( len(sys.argv) < 4 ):
    sys.exit( 'need to specify .po file, in charmap, out charmap' )

po_file=sys.argv[1]
in_charmap_file=sys.argv[2]
out_charmap_file=sys.argv[3]
model_prefix=sys.argv[4]

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
ITERATIONS=10
EPOCHS=100

print('Load Charmaps...')
intable = util.EncodedCharacterTable(in_charmap_file, MAXLENGTH)
outtable = util.EncodedCharacterTable(out_charmap_file, MAXLENGTH)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )
#test a subset
#po_data = po_data[0:100]

print('Vectorization...')
x = np.zeros((len(po_data), MAXLENGTH, intable.maxval), dtype=np.bool)
y = np.zeros((len(po_data), MAXLENGTH, outtable.maxval), dtype=np.bool)
for i, t in enumerate(po_data):
    #print(t.msgid)
    #print(t.msgstr)
    #print
    x[i] = intable.encode(t.msgid)
    y[i] = outtable.encode(t.msgstr)

x = np.array( x, dtype=np.bool )
y = np.array( y, dtype=np.bool )

#debug encodings
print("X")
print(x.shape)
#for row in x:
#    print(np.where(row == True))
#    print(intable.decode( row ))

print("Y")
print(y.shape)
#for row in y:
#    print(np.where(row == True))
#    print(outtable.decode( row ))

print('Build model...')

model = util.create_model( intable.maxval, outtable.maxval )

model.compile(loss='categorical_crossentropy', optimizer='sgd')

model_yml = model.to_yaml()
with open( model_prefix + ".yml", "w") as yml_file:
    yml_file.write(model_yml)

for iteration in range(1, ITERATIONS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.05, verbose=1, callbacks=[ModelCheckpoint(model_prefix + '_' + str(iteration) + '_{epoch}_{val_loss}.h5')])

