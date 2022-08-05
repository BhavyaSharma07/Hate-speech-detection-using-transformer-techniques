#!/usr/bin/env python
# coding: utf-8

from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from keras import Input, Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, Bidirectional, Dense,     LSTM, Conv1D, Dropout, concatenate
import torch
import pickle
from torch import nn
from transformers import *
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split


from preprocessor import clean_txt
import nltk
nltk.download('wordnet')
nltk.download('stopwords')


from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
        
        
        
class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, dropout=0.1, kernel_num=3, kernel_sizes=[2,3,4], num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.num_labels)
        
    def forward(self, inputs, labels=None):
        output = inputs.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
            

            
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        
        
def get_train_examples(text):
    text = train_df['comment_text'].values
    labels = train_df[train_df.columns[2:]].values
    examples = []
    for i in range(len(train_df)):
        examples.append(InputExample(ids[i], text[i], labels=labels[i]))
    return examples


def get_features_from_examples(text, max_seq_len, tokenizer):
    features = []

    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[:(max_seq_len - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(tokens)
    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids
                                      ))
    return features
    


def get_dataset_from_features(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(input_ids,
                            input_mask,
                            segment_ids
                            )
    return dataset
    
    
    
    
def build_model():
    
    max_features = 15000
    max_len = 150
    embed_size = 150
    
    embedding_matrix = np.genfromtxt("./weights/glove_trained_embedding.csv", delimiter=',')
    main_input1 = Input(shape=(max_len,), name='main_input1')
    x1 = (Embedding(max_features + 1, 300, input_length=max_len,
                    weights=[embedding_matrix], trainable=False))(main_input1)
    x1 = SpatialDropout1D(0.4)(x1)
    x2 = Bidirectional(LSTM(75, dropout=0.5, return_sequences=True))(x1)
    x = Dropout(0.55)(x2)
    x = Bidirectional(LSTM(50, dropout=0.5, return_sequences=True))(x)
    hidden = concatenate([
        Attention(max_len)(x1),
        Attention(max_len)(x2),
        Attention(max_len)(x)
    ])
    hidden = Dense(32, activation='selu')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(16, activation='selu')(hidden)
    hidden = Dropout(0.5)(hidden)
    output_lay1 = Dense(8, activation='sigmoid')(hidden)
    model = Model(inputs=[main_input1], outputs=output_lay1)
    model.load_weights(filepath='./weights/final_clf_model.hdf5')
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=['binary_accuracy'])
    
    return model



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])

def predict():
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        input_txt = []
        model = build_model()
        clean_text =  clean_txt(data)
        input_txt.append(clean_text)
        max_features = 15000
        max_len = 150


        tk = Tokenizer(lower=True, filters='', num_words=max_features, oov_token=True)
        tk.fit_on_texts(input_txt)
        tokenized = tk.texts_to_sequences(input_txt)
        x_test = pad_sequences(tokenized, maxlen=max_len)

        vpp = model.predict(x_test)
        vpp = vpp.flatten(order='C')
        vpp_str = list(vpp)
        vpp_str = ', '.join(map(str, vpp_str))
        
        
        batch = 8
        seq_len = 256
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        basemodel = BertModel.from_pretrained(pretrained_weights)
        features = get_features_from_examples(data, seq_len, tokenizer)
        dataset = get_dataset_from_features(train_features)
        
        dataloader = DataLoader(dataset, batch_size=batch)
        
        y_pred = []
        device = torch.device(type='cpu')
        model = torch.load('./weights/bert_cnn_clf.pth', map_location=torch.device('cpu')) 
        model.eval()
        
        
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            with torch.no_grad():
               inputs,_ = basemodel(input_ids, segment_ids, input_mask)
               logits = model(inputs)
        
            y_pred.append(logits)
        
        preds = torch.cat(y_pred, dim=0).float().cpu().detach().numpy()
        preds = preds.flatten(order='C')
        preds_str = list(preds)
        preds_str = ', '.join(map(str, preds_str))
        
        #pred_ix = np.argmax(preds, axis=1)
        

    return render_template('result.html', foobar = vpp_str, bts = preds_str)



if __name__ == "__main__":
    app.run(debug=True)
    
