import gradio as gr
import styletts2importable
import ljspeechimportable
import torch
import os
from tortoise.utils.text import split_and_recombine_text
import numpy as np
import pickle
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
voices = {}
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
# todo: cache computed style, load using pickle
# if os.path.exists('voices.pkl'):
    # with open('voices.pkl', 'rb') as f:
        # voices = pickle.load(f)
# else:
for v in voicelist:
    voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')
# def synthesize(text, voice, multispeakersteps):
#     if text.strip() == "":
#         raise gr.Error("You must enter some text")
#     # if len(global_phonemizer.phonemize([text])) > 300:
#     if len(text) > 300:
#         raise gr.Error("Text must be under 300 characters")
#     v = voice.lower()
#     # return (24000, styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1))
#     return (24000, styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=multispeakersteps, embedding_scale=1))
def synthesize(text, voice, lngsteps, password, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 7500:
        raise gr.Error("Text must be <7.5k characters")
    texts = split_and_recombine_text(text)
    v = voice.lower()
    audios = []
    for t in progress.tqdm(texts):
        audios.append(styletts2importable.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
    return (24000, np.concatenate(audios))
# def longsynthesize(text, voice, lngsteps, password, progress=gr.Progress()):
#     if password == os.environ['ACCESS_CODE']:
#         if text.strip() == "":
#             raise gr.Error("You must enter some text")
#         if lngsteps > 25:
#             raise gr.Error("Max 25 steps")
#         if lngsteps < 5:
#             raise gr.Error("Min 5 steps")
#         texts = split_and_recombine_text(text)
#         v = voice.lower()
#         audios = []
#         for t in progress.tqdm(texts):
#             audios.append(styletts2importable.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
#         return (24000, np.concatenate(audios))
#     else:
#         raise gr.Error('Wrong access code')
def clsynthesize(text, voice, vcsteps, progress=gr.Progress()):
    # if text.strip() == "":
    #     raise gr.Error("You must enter some text")
    # # if global_phonemizer.phonemize([text]) > 300:
    # if len(text) > 400:
    #     raise gr.Error("Text must be under 400 characters")
    # # return (24000, styletts2importable.inference(text, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=20, embedding_scale=1))
    # return (24000, styletts2importable.inference(text, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=vcsteps, embedding_scale=1))
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 7500:
        raise gr.Error("Text must be <7.5k characters")
    texts = split_and_recombine_text(text)
    audios = []
    for t in progress.tqdm(texts):
        audios.append(styletts2importable.inference(t, styletts2importable.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=vcsteps, embedding_scale=1))
    return (24000, np.concatenate(audios))
def ljsynthesize(text, steps, progress=gr.Progress()):
    # if text.strip() == "":
    #     raise gr.Error("You must enter some text")
    # # if global_phonemizer.phonemize([text]) > 300:
    # if len(text) > 400:
    #     raise gr.Error("Text must be under 400 characters")
    noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
    # return (24000, ljspeechimportable.inference(text, noise, diffusion_steps=7, embedding_scale=1))
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 7500:
        raise gr.Error("Text must be <7.5k characters")
    texts = split_and_recombine_text(text)
    audios = []
    for t in progress.tqdm(texts):
        audios.append(ljspeechimportable.inference(t, noise, diffusion_steps=steps, embedding_scale=1))
    return (24000, np.concatenate(audios))


with gr.Blocks() as vctk: # just realized it isn't vctk but libritts but i'm too lazy to change it rn
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            voice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-2', interactive=True)
            multispeakersteps = gr.Slider(minimum=3, maximum=15, value=7, step=1, label="Diffusion Steps", info="Theoretically, higher should be better quality but slower, but we cannot notice a difference. Try with lower steps first - it is faster", interactive=True)
            # use_gruut = gr.Checkbox(label="Use alternate phonemizer (Gruut) - Experimental")
        with gr.Column(scale=1):
            btn = gr.Button("Synthesize", variant="primary")
            audio = gr.Audio(interactive=False, label="Synthesized Audio")
            btn.click(synthesize, inputs=[inp, voice, multispeakersteps], outputs=[audio], concurrency_limit=4)
with gr.Blocks() as clone:
    with gr.Row():
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            clvoice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=300)
            vcsteps = gr.Slider(minimum=3, maximum=20, value=20, step=1, label="Diffusion Steps", info="Theoretically, higher should be better quality but slower, but we cannot notice a difference. Try with lower steps first - it is faster", interactive=True)
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio")
            clbtn.click(clsynthesize, inputs=[clinp, clvoice, vcsteps], outputs=[claudio], concurrency_limit=4)
# with gr.Blocks() as longText:
#     with gr.Row():
#         with gr.Column(scale=1):
#             lnginp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
#             lngvoice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value='m-us-1', interactive=True)
#             lngsteps = gr.Slider(minimum=5, maximum=25, value=10, step=1, label="Diffusion Steps", info="Higher = better quality, but slower", interactive=True)
#             lngpwd = gr.Textbox(label="Access code", info="This feature is in beta. You need an access code to use it as it uses more resources and we would like to prevent abuse")
#         with gr.Column(scale=1):
#             lngbtn = gr.Button("Synthesize", variant="primary")
#             lngaudio = gr.Audio(interactive=False, label="Synthesized Audio")
#             lngbtn.click(longsynthesize, inputs=[lnginp, lngvoice, lngsteps, lngpwd], outputs=[lngaudio], concurrency_limit=4)
with gr.Blocks() as lj:
    with gr.Row():
        with gr.Column(scale=1):
            ljinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            ljsteps = gr.Slider(minimum=3, maximum=20, value=3, step=1, label="Diffusion Steps", info="Theoretically, higher should be better quality but slower, but we cannot notice a difference. Try with lower steps first - it is faster", interactive=True)
        with gr.Column(scale=1):
            ljbtn = gr.Button("Synthesize", variant="primary")
            ljaudio = gr.Audio(interactive=False, label="Synthesized Audio")
            ljbtn.click(ljsynthesize, inputs=[ljinp, ljsteps], outputs=[ljaudio], concurrency_limit=4)
with gr.Blocks(title="StyleTTS 2", css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown("""# StyleTTS 2

[Paper](https://arxiv.org/abs/2306.07691) - [Samples](https://styletts2.github.io/) - [Code](https://github.com/yl4579/StyleTTS2)

A free demo of StyleTTS 2. **I am not affiliated with the StyleTTS 2 Authors.**

#### Help this space get to the top of HF's trending list! Please give this space a Like!

**Before using this demo, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.**

**NOTE: StyleTTS 2 does better on longer texts.** For example, making it say "hi" will produce a lower-quality result than making it say a longer phrase.""")
    gr.HTML("""<script async src="https://www.googletagmanager.com/gtag/js?id=G-KP5GWL8NN5"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-KP5GWL8NN5');
</script>
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "jydi4lprw6");
</script>""")
    # gr.TabbedInterface([vctk, clone, lj, longText], ['Multi-Voice', 'Voice Cloning', 'LJSpeech', 'Long Text [Beta]'])
    gr.TabbedInterface([vctk, clone, lj], ['Multi-Voice', 'Voice Cloning', 'LJSpeech', 'Long Text [Beta]'])
    gr.Markdown("""
Demo by [mrfakename](https://twitter.com/realmrfakename). I am not affiliated with the StyleTTS 2 authors.

Run this demo locally using Docker:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/styletts2-styletts2:latest python app.py
```
""")
if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False, share=True)

