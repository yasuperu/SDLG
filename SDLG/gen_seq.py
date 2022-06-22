from tensorflow.keras.utils import to_categorical
from SDLG.tokenizer import total_words


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