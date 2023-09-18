import gradio as gr

js = "(x) => confirm('Press a button!')"

with gr.Blocks() as demo:
    btn = gr.Button()
    checkbox = gr.Checkbox()
    
    btn.click(None, None, checkbox, _js=js)
    
demo.launch()