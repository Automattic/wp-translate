# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range
import rnn256_util as util
import sys

if ( len(sys.argv) < 3 ):
    sys.exit( 'need to specify .po file and output model' )

po_file=sys.argv[1]
model_file=sys.argv[2]

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True

# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 1024
BATCH_SIZE = 128
LAYERS = 2
MAXCHAR = 0x30C5 #ugh so much to cover

ctable = util.CharacterTable(MAXCHAR)

print('Load Data...')
po_data = util.load_translated_po_data( po_file )

MAXLEN = 0
for i, t in enumerate(po_data):
    if ( MAXLEN < len(t.msgid) ):
	MAXLEN = len(t.msgid)
    if ( MAXLEN < len(t.msgstr) ):
	MAXLEN = len(t.msgid)

print('Vectorization...')
x = np.zeros((len(po_data), MAXLEN, MAXCHAR), dtype=np.bool)
y = np.zeros((len(po_data), MAXLEN, MAXCHAR), dtype=np.bool)
for i, t in enumerate(po_data):
    x[i] = ctable.encode(t.msgid)
    y[i] = ctable.encode(t.msgstr)

x = np.array( x, dtype=np.bool )
y = np.array( y, dtype=np.bool )

# Shuffle (X, y) in unison to get good randomization
# TODO: maybe we should not do this?
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(x) - len(x) // 10
(x_train, x_val) = (slice_X(x, 0, split_at), slice_X(x, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(x_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(None, MAXCHAR)))
# For the decoder's input, we repeat the encoded input for each time step
# TODO: hmmmm...
#model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(MAXCHAR)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(x_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowX, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(util.colors.ok + '☑' + util.colors.close if correct == guess else util.colors.fail + '☒' + util.colors.close, guess)
        print('---')
    util.save_model( model_file, model )
