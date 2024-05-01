# CrossTalk
<a href="https://huggingface.co/spaces/Saptarshi003/CrossTalk"><img src="https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-demo-yellow"></a>

![demo](./imgs/demo.png)


## Overview
A English to Spanish Neural Machine Translation system, enabling the translation of English sentences
into Spanish.
Key features include - 
 - Bidirectional LSTM architecture coupled with Luong's attention mechanism. 
 - Vocabulary size of 5000.
 - Deployed Using Gradio
 - Achieving a notable 70% translation accuracy on the Spa-Eng dataset by Anki 

## Usage

 1. Install all the packages

 ```python
 python3 -m pip install -r requirements.txt
 ```

 2. Run the Gradio app

 ```python
 gradio app.py
 ```

## To-do
Build a command line interface such that input can be given and output can be received through command-line