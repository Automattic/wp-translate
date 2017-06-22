# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import polib


class CharacterTable:
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation up to a max utf-8 val
          - inefficient, but easy
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''

    def __init__(self, maxval, maxlength):
        self.maxval = maxval
        self.maxlength = maxlength

    def encode(self, string):
        X = np.zeros((self.maxlength, self.maxval), dtype=np.bool)
        last = 0
        for i, c in enumerate(string):
            v = ord(c)
            if ( v >= self.maxval ):
                print( 'Unsupported char [' + c + '] ' + str(v) )
                print( string )
            	v = 0
            X[i, v] = 1
            last = i
        for i in xrange( last, self.maxlength ):
            X[i, 0] = 0 #use null char as end of string for padding
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(char(x) for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def save_model( f, model ):
    # serialize model to JSON
    model_json = model.to_json()
    with open(f + ".json", "w") as json_file:
    	json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f + ".h5")
    print("Saved model to " + f )

def load_model( f ):
    # load json and create model
    json_file = open(f + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f + ".h5")
    print("Loaded model from " + f )
    return loaded_model

def load_translated_po_data( f ):
    '''
	Load only translated data and just get msgid and msgstr
    '''
    po = polib.pofile(f)
    data = []
    for entry in po:
        if ( '' == entry.msgstr ):
          continue
      	#TODO: we should split long text into sentences or some other blocks somehow
        if ( len(entry.msgid) > 500 ):
          continue
        if ( len(entry.msgstr) > 500 ):
          continue
      	data.append(entry)
    return data

