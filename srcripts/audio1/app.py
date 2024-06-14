# import gradio as gr
# from function import process_stream

# def start_gradio():
#     with gr.Blocks() as demo:
#         input_audio = gr.Audio(sources=["microphone"], streaming=True)
#         text = gr.Text(show_label=False)
#         state = gr.State(value=(None, ""))  # Initial state: (stream, remaining_text)
#         enable_speaker = gr.Checkbox(label="Enable Speaker", value=True)

#         def transcribe_and_check(new_chunk, state, enable_speaker):
#             new_state, responses = process_stream(state, new_chunk, enable_speaker)
#             for response in responses:
#                 yield new_state, response

#         input_audio.change(transcribe_and_check, inputs=[input_audio, state, enable_speaker], outputs=[state, text])
#         button = gr.Button("Stop")
#         button.click(
#             cancels=[text, input_audio],
#             inputs=[],
#             outputs=[],
#         )

#     demo.queue()
#     demo.launch()

# if __name__ == "__main__":
#     start_gradio()
# import gradio as gr
# from function import transcribe, translate, speak

# with gr.Blocks() as demo:
#     input_audio = gr.Audio(sources=["microphone"], streaming=True)
#     text = gr.Textbox(placeholder="Transcribed text will appear here")
#     translated_text = gr.Textbox(placeholder="Translated text will appear here")
#     output_audio = gr.Audio()
    
#     gr.Interface(
#         transcribe,
#         ["state", input_audio],
#         ["state", text],
#         live=True,
#         title="Real-Time Speech-to-Text",
#     )
    
    # gr.Interface(
    #     translate,
    #     ["text"],
    #     ["text"],
    #     live=True,
    #     title="Translation",
    # )
    
    # # gr.Interface(
    # #     speak,
    # #     ["text"],
    # #     ["audio"],
    # #     live=True,
    # #     title="Text-to-Speech",
    # # )
    
#     demo.launch()
import gradio as gr
from function import transcribe, translate, speak


def test(text):
    return text
with gr.Blocks() as demo:
    input_audio = gr.Audio(sources=["microphone"], streaming=True)
    text = gr.Textbox(placeholder="Transcribed text will appear here")
    text1 = gr.Textbox(placeholder="Transcribed text will appear here",visible=False)
    text2 = gr.Textbox(placeholder="Transcribed text will appear here",visible=False)
    text3 = gr.Textbox(placeholder="Transcribed text will appear here",visible=False)
    translated_text = gr.Textbox(placeholder="Translated text will appear here")
    output_audio = gr.Audio(streaming=True, autoplay=True, label="Translated audio stream may not work properly on iPhone yet.",interactive= False)
    
    gr.Interface(
        transcribe,
        ["state", input_audio],
        ["state", text,text1],
        live=True,
        title="Real-Time Speech-to-Text",
    )
    # with gr.Row():
    #     text1 = gr.Textbox(placeholder="Transcribed text will appear here",visible=False)
    #     text2 = gr.Textbox(placeholder="Transcribed text will appear here",visible=False)
    gr.on(text.change(translate,[text1],[text2,text3],trigger_mode='always_last'))
    gr.on(text2.change(speak,[text1],[output_audio],trigger_mode="always_last"))
    
    # gr.Interface(
    #     translate,
    #     [text],
    #     [translated_text],
    #     live=True,
    #     title="Translation",
    # )
    #text.change(translate,[text] , [translated_text], every=3)
    # gr.Interface(
    #     speak,
    #     [text],
    #     [output_audio],
    #     live=True,
    #     title="Text-to-Speech",
    # )
    
    # transcribe_dep = demo.load(transcribe,inputs= [input_audio],outputs= [text], every=3)
    # # translate_dep = demo.load(translate, text, translated_text, every=3, cancels=[transcribe_dep])
    # # speak_dep = demo.load(speak, translated_text, output_audio, every=3, cancels=[translate_dep])

    demo.launch()
