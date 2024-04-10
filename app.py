import gradio as gr
import tensorflow as tf
import keras
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

vocab = list(np.load("vocabulary.npy"))

model = keras.saving.load_model("encodeco_attention_nmt_5000.keras")


def translate(english_sentence):
    translation = ""
    for word_idx in range(50):
        X = tf.constant([english_sentence])  # encoder input
        X_dec = tf.constant(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = vocab[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


demo = gr.Interface(
    fn=translate,
    inputs=["text"],
    outputs=[gr.Textbox(label="spanish_translation", lines=3)],
)

demo.launch()
