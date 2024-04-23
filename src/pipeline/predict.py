import tensorflow as tf
import keras
import numpy as np

loaded_model = keras.saving.load_model(
    "./artifacts/encodeco_attention_nmt_6000.keras", compile=False)
es_vocab = np.load("./artifacts/vocabulary.npy")

max_length = 50


def translate(english_sentence):
    translation = ""
    for word_idx in range(max_length):
        X = tf.constant([english_sentence])  # encoder input
        X_dec = tf.constant(["startofseq " + translation])  # decoder input
        y_proba = loaded_model.predict(
            (X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = es_vocab[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()
