# Neural_Machine_Translation
<a href="https://huggingface.co/spaces/Saptarshi003/Neural_Machine_Translation"><img src="https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-demo-yellow"></a>

![demo](./imgs/demo.png)


## Overview
A English to Spanish Neural Machine Translation (NMT) system. 
Key features include - 
 - A bidirectional LSTM architecture coupled with Luong's attention mechanism, ensuring accurate and contextually relevant translations. 
- A vocabulary size of 5000 and encoder embedding size of 128.
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