from tensorflow.keras.preprocessing.text import Tokenizer
from SDLG.input import text

tokenizer = Tokenizer(char_level = False, filters = '')
   
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

token_list = tokenizer.texts_to_sequences([text])[0]