import os
import webbrowser
import numpy as np
import random
from SDLG.tokenizer import tokenizer, token_list
from SDLG.gen_seq import generate_sequences, step
from SDLG.input import start_story, seq_length
from SDLG.model_load import model
from SDLG.sample import sample_with_temp
from tensorflow.keras.callbacks import LambdaCallback

train_model = True

X, y, num_seq = generate_sequences(token_list, step)

X = np.array(X)
y = np.array(y)



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
    epochs = 1
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