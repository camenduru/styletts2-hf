import gradio as gr
import styletts2importable
import ljspeechimportable
import torch
import os
import pickle
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
voices = {}
# todo: cache computed style, load using pickle
if os.path.exists('voices.pkl', 'rb') as f:
    voices = pickle.load(f)
else:
    for v in voicelist:
        voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')
def synthesize(text, voice):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 300:
        raise gr.Error("Text must be under 300 characters")
    v = voice.lower()
    return (24000, styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1))
def clsynthesize(text, voice):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 300:
        raise gr.Error("Text must be under 300 characters")
    return (24000, styletts2importable.inference(text, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=20, embedding_scale=1))
def ljsynthesize(text):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 300:
        raise gr.Error("Text must be under 300 characters")
    noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
    return (24000, ljspeechimportable.inference(text, noise, diffusion_steps=7, embedding_scale=1))


with gr.Blocks() as vctk:
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            voice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-1', interactive=True)
        with gr.Column(scale=1):
            btn = gr.Button("Synthesize", variant="primary")
            audio = gr.Audio(interactive=False, label="Synthesized Audio")
            btn.click(synthesize, inputs=[inp, voice], outputs=[audio], concurrency_limit=4)
with gr.Blocks() as clone:
    with gr.Row():
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            clvoice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=300)
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio")
            clbtn.click(clsynthesize, inputs=[clinp, clvoice], outputs=[claudio], concurrency_limit=4)
with gr.Blocks() as lj:
    with gr.Row():
        with gr.Column(scale=1):
            ljinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
        with gr.Column(scale=1):
            ljbtn = gr.Button("Synthesize", variant="primary")
            ljaudio = gr.Audio(interactive=False, label="Synthesized Audio")
            ljbtn.click(ljsynthesize, inputs=[ljinp], outputs=[ljaudio], concurrency_limit=4)
with gr.Blocks(title="StyleTTS 2", css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown("""# StyleTTS 2

[Paper](https://arxiv.org/abs/2306.07691) - [Samples](https://styletts2.github.io/) - [Code](https://github.com/yl4579/StyleTTS2)

A free demo of StyleTTS 2. **I am not affiliated with the StyleTTS 2 Authors.**

**Before using this demo, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.**

Is there a long queue on this space? Duplicate it and add a more powerful GPU to skip the wait! **Note: Thank you to Hugging Face for their generous GPU grant program!**

**NOTE: StyleTTS 2 does better on longer texts.** For example, making it say "hi" will produce a lower-quality result than making it say a longer phrase.""")
    gr.DuplicateButton("Duplicate Space")
    gr.TabbedInterface([vctk, clone, lj], ['Multi-Voice', 'Voice Cloning', 'LJSpeech'])
    gr.Markdown("""
Demo by by [mrfakename](https://twitter.com/realmrfakename). I am not affiliated with the StyleTTS 2 authors.

Run this demo locally using Docker:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/styletts2-styletts2:latest python app.py
```
""")
if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False)

