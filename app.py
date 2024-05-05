import gradio as gr
from src.pipeline.predict import translate

demo = gr.Interface(
    fn=translate,
    inputs=["text"],
    outputs=[gr.Textbox(label="spanish_translation", lines=3)],
)

demo.launch()
