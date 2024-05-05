import numpy as np
from pathlib import Path
import keras
import tensorflow as tf
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = keras.utils.get_file("spa-eng.zip", origin=url, cache_dir="datasets",
                            extract=True)
text = (Path(path).with_name("spa-eng") /
        "spa.txt").read_text(encoding='utf-8')

text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)  # separates the pairs into 2 lists

vocab_size = 6000
max_length = 50
text_vec_layer_en = keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)
text_vec_layer_es = keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)
text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

es_vocab = np.array(text_vec_layer_es.get_vocabulary())
np.save('./artifacts/vocabulary.npy', es_vocab)

X_train = tf.constant(sentences_en[:100_000])
X_valid = tf.constant(sentences_en[100_000:])
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])
Y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
Y_valid = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])


def make_model():
    encoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)
    embed_size = 128
    encoder_input_ids = text_vec_layer_en(encoder_inputs)
    decoder_input_ids = text_vec_layer_es(decoder_inputs)
    encoder_embedding_layer = keras.layers.Embedding(vocab_size, embed_size,
                                                     mask_zero=True)
    decoder_embedding_layer = keras.layers.Embedding(vocab_size, embed_size,
                                                     mask_zero=True)
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)
    encoder = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, return_state=True, dropout=0.3))
    encoder_outputs, *encoder_state = encoder(encoder_embeddings)
    encoder_state = [tf.concat(encoder_state[::2], axis=-1),  # short-term (0 & 2)
                     tf.concat(encoder_state[1::2], axis=-1)]  # long-term (1 & 3)
    decoder = keras.layers.LSTM(512, return_sequences=True, dropout=0.2)
    decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)
    attention_layer = keras.layers.Attention()
    attention_outputs = attention_layer([decoder_outputs, encoder_outputs])
    output_layer = keras.layers.Dense(vocab_size, activation="softmax")
    Y_proba = output_layer(attention_outputs)
    model = keras.Model(inputs=[encoder_inputs, decoder_inputs],
                        outputs=[Y_proba])
    return model


def train_and_save():
    model = make_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                  metrics=["accuracy"])
    model.fit((X_train, X_train_dec), Y_train, epochs=10,
              validation_data=((X_valid, X_valid_dec), Y_valid))
    model.save("./artifacts/encodeco_attention_nmt_6000.keras")


train_and_save()
