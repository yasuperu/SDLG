import os
import webbrowser
import numpy as np
import re
import random
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import LambdaCallback

load_saved_model = True
train_model = True

seq_length = 20

filename = "data/dataset_SVG_path.txt"

with open(filename, encoding='utf-8-sig') as f:
    text = f.read()

start_story = '| ' * seq_length
    
text = start_story + text
text = text.replace('d="', start_story)
text = text.replace('\n', ' ')
text = text.replace('\t', ' ')
text = text.replace('<?', ' ')
text = text.replace('<path', ' ')
text = re.sub(r'<?xml.*?>', '', text)
text = re.sub(r'<svg.*?>', '', text)
text = re.sub(r'</svg>', '', text)
text = re.sub(r'<g.*?>', '', text)
text = re.sub(r'</g>', '', text)
text = re.sub(r'<path.*?>', '', text)
text = re.sub(r'</path>', '', text)
text = re.sub(r'<text.*?>', '', text)
text = re.sub(r'</text>', '', text)
text = re.sub(r'fill="*.*?>', '', text)
text = re.sub(r'Z"', '', text)
text = re.sub(r'stroke-opacity="*.*?>', '', text)
text = re.sub(r'fill-opacity="*.*?>', '', text)
text = re.sub(r'stroke="*.*?>', '', text)
text = re.sub(r'stroke-width="*.*?>', '', text)

tokenizer = Tokenizer(char_level = False, filters = '')
   
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

token_list = tokenizer.texts_to_sequences([text])[0]

def generate_sequences(token_list, step):
    
    X = []
    y = []

    for i in range(0, len(token_list) - seq_length, step):
        X.append(token_list[i: i + seq_length])
        y.append(token_list[i + seq_length])
    

    y = to_categorical(y, num_classes = total_words)
    
    num_seq = len(X)
    print('Number of sequences:', num_seq, "\n")
    
    return X, y, num_seq

step = 1
seq_length = 20

X, y, num_seq = generate_sequences(token_list, step)

X = np.array(X)
y = np.array(y)

if load_saved_model:
    model = load_model('savedModels/aesop_no_dropout_10_svg.h5')

else:

    n_units = 256
    embedding_size = 100

    text_in = Input(shape = (None,))
    embedding = Embedding(total_words, embedding_size)
    x = embedding(text_in)
    x = LSTM(n_units)(x)
    text_out = Dense(total_words, activation = 'softmax')(x)

    model = Model(text_in, text_out)

    opti = RMSprop(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opti)

def sample_with_temp(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, next_words, model, max_sequence_len, temp):
    output_text = seed_text
    
    seed_text = start_story + seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        
        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample_with_temp(probs, temperature = temp)
        
        if y_class == 0:
            output_word = ''
        else:
            output_word = tokenizer.index_word[y_class]
            
        if output_word == "|":
            break
            
        output_text += output_word + ' '
        seed_text += output_word + ' '
            
            
    return output_text

def on_epoch_end(epoch, logs):
    seed_text = "M500.0,500.0 "
    gen_words = 500

    print('Temp 0.2')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.2))
    print('Temp 0.33')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.33))
    print('Temp 0.5')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.5))
    print('Temp 1.0')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 1))

    
    
if train_model:
    epochs = 200
    batch_size = 32
    num_batches = int(len(X) / batch_size)
    callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks = [callback], shuffle = True)

model.save('savedModels/aesop_no_dropout_10_svg.h5')

seed_text = "M500.0,500.0 "
gen_words = random.randint(100, 1000)
temp = 1.0

newPath = generate_text(seed_text, gen_words, model, seq_length, temp)

input = open("data/template.svg", "rt")
output = open("output/SDLG.svg", "wt")

for line in input:
    output.write(line.replace("SDLG1", newPath))

input.close()
output.close()

webbrowser.open('file://' + os.path.realpath("output/SDLG.svg"))