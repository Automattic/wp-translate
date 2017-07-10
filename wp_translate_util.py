# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import polib as polib
from seq2seq.models import SimpleSeq2Seq
from seq2seq.models import Seq2Seq
from seq2seq.models import AttentionSeq2Seq
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation
from keras.models import Model

class UTF8CharacterTable:
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


class EncodedCharacterTable:
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation based on an input file
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''

    def __init__(self, charmap_file, maxlength):
        self.maxlength = maxlength
        print('Loading Char Map...')
        charmap_fh = open(charmap_file, 'r')
        with open(charmap_file, 'r') as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.encode_map = {}
        self.decode_map = {}
        for x in content:
            i,c = x.split("\t")
            c = unichr(int(c))
            self.encode_map[c] = int(i)
            self.decode_map[int(i)] = c
        self.maxval = len( self.encode_map )
        print(self.encode_map)

    def encode(self, string):
        X = np.zeros((self.maxlength, self.maxval), dtype=np.bool)
        last = 0
        for i, c in enumerate(string):
            v = self.encode_map[c]
            if ( v >= self.maxval ):
                print( 'Unsupported char [' + c + '] ' + str(v) )
                print( string )
                v = 0
            X[i, v] = 1
            last = i
        #use null char as end of string, leave zeros for padding
        # using end char as padding makes that char too common
        X[last+1, 0] = 1
        return X

    def decode(self, X, calc_argmax=True):
	print(np.amax(X,axis=-1))
        if calc_argmax:
            X = np.argmax(X, axis=-1)
        print(X)
        return ''.join(self.decode_map[x] for x in X)


def build_encoded_char_map( po_file, str_type, char_file ):
    '''
        Convert all chars in the given string type (msgid, msgstr) into an encoded character mapping
        and store them as their unicode points
    '''
    po = polib.pofile(po_file)
    charmap = { 0: 0 } #null char is first char
    idx = 1
    for entry in po:
        s = ''
        if ( str_type == 'msgstr' ):
            s = entry.msgstr
        elif ( str_type == 'msgid' ):
            s = entry.msgid
        else:
            raise Exception('Incorrect str_type in getting data from po file')
        if ( '' == s ):
          continue
        for i, c in enumerate(s):
            c = ord(c)
            if c not in charmap:
                charmap[c] = idx
                idx+=1

    inv_map = {v: k for k, v in charmap.iteritems()}
    print( inv_map )
    with open(char_file, "w") as fh:
        for i in xrange(0,idx-1):
            fh.write( str(i) + "\t" + str(inv_map[i]) + "\n" )

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

def create_model( in_dim, out_dim ):
    # Parameters for the model
    HIDDEN_SIZE = 512
    LAYERS = 2
    MAXLENGTH = 500  #max length of input text

    # ok model
    model = Seq2Seq(input_dim=in_dim, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=out_dim, depth=LAYERS, peek=True, readout_activation='softmax')

    #add an extra layer to avoid NaN when calculating the loss function
    # seq2seq uses tanh on output which can result in NaN
    # hacky based on https://github.com/farizrahman4u/seq2seq/issues/189#issuecomment-308312812
    #model.add(TimeDistributed(Dense(out_dim)))
    #model.add(Activation('softmax'))

    #much more intricate model - runs out of memory (only on GPU?)
    #model = AttentionSeq2Seq(input_dim=in_dim, input_length=MAXLENGTH, hidden_dim=HIDDEN_SIZE, output_length=MAXLENGTH, output_dim=out_dim, depth=LAYERS, readout_activation='softmax')
    return model

def load_translated_po_data( f ):
    '''
        Load only translated data and just get msgid and msgstr
    '''
    po = polib.pofile(f)
    data = []
    for entry in po.translated_entries():
        if ( '' == entry.msgstr ):
          continue
        #TODO: we should split long text into sentences or some other blocks somehow
        if ( len(entry.msgid) > 500 ):
          continue
        if ( len(entry.msgstr) > 500 ):
          continue
        data.append(entry)
    return data

