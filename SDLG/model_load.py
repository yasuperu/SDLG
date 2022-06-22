from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from SDLG.tokenizer import total_words

load_saved_model = True

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