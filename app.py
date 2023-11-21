import gradio as gr
import styletts2importable
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
voices = {
    'angie': styletts2importable.compute_style('voices/angie.wav'),
    'daniel': styletts2importable.compute_style('voices/daniel.wav'),
    'dotrice': styletts2importable.compute_style('voices/dotrice.wav'),
    'lj': styletts2importable.compute_style('voices/lj.wav'),
    'mouse': styletts2importable.compute_style('voices/mouse.wav'),
    'pat': styletts2importable.compute_style('voices/pat.wav'),
    'tom': styletts2importable.compute_style('voices/tom.wav'),
    'william': styletts2importable.compute_style('voices/william.wav'),
}
def synthesize(text, voice):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 500:
        raise gr.Error("Text must be under 500 characters")
    v = voice.lower()
    return (24000, styletts2importable.inference(text, voices[v], alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1))

with gr.Blocks(title="StyleTTS 2", css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown("""# StyleTTS 2

[Paper](https://arxiv.org/abs/2306.07691) - [Samples](https://styletts2.github.io/) - [Code](https://github.com/yl4579/StyleTTS2)

A free demo of StyleTTS 2. Not affiliated with the StyleTTS 2 Authors.

**Before using this demo, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.**

This space does NOT allow voice cloning. We use some default voice from Tortoise TTS instead.

Is there a long queue on this space? Duplicate it and add a GPU to skip the wait!""")
    gr.DuplicateButton("Duplicate Space")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            voice = gr.Dropdown(['Angie', 'Daniel', 'Tom', 'LJ', 'Pat', 'Tom', 'Dotrice', 'Mouse', 'William'], label="Voice", info="Select a voice. We use some voices from Tortoise TTS.", value='Tom', interactive=True)
        with gr.Column(scale=1):
            btn = gr.Button("Synthesize")
            audio = gr.Audio(interactive=False, label="Synthesized Audio")
            btn.click(synthesize, inputs=[inp, voice], outputs=[audio], concurrency_limit=4)
    
if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False)

