"""
Gradio app for medical Visual AI inference with OpenVINO

Usage: python app.py [model] [device]
"""

import argparse
import time

import gradio as gr

from medical_inference_openvino import MedicalOV

css = """
.text textarea {font-size: 24px !important;}
"""


def process_inputs(image, question):
    if image is None:
        return "Please upload an image."
    if not question:
        return "Please enter a question."
    return medicalov.run_inference_image(image, question)


def reset_inputs():
    return None, "", ""


def launch_demo(title):
    with gr.Blocks(css=css) as demo:
        gr.Markdown(f"# {title} OpenVINO Demo")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload an Image", height=300, width=500)
            with gr.Column():
                text_input = gr.Textbox(label="Enter a Question", elem_classes="text")
                output_text = gr.Markdown(label="Answer", height=600)

        with gr.Row():
            process_button = gr.Button("Process")
            reset_button = gr.Button("Reset")

        gr.Examples(
            [["Describe this X-Ray"]],
            text_input,
        )

        gr.Markdown(
            "NOTE: This OpenVINO model is unvalidated. Results are provisional and may contain errors. Use this demo to explore AI PC and OpenVINO optimizations"
        )
        gr.Markdown("For research purposes only.")

        process_button.click(process_inputs, inputs=[image_input, text_input], outputs=output_text)
        text_input.submit(process_inputs, inputs=[image_input, text_input], outputs=output_text)
        reset_button.click(reset_inputs, inputs=[], outputs=[image_input, text_input, output_text])
        demo.launch(server_port=7790)


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("device")
args = parser.parse_args()

print(f"Loading {args.model} to {args.device}")
start = time.perf_counter()
medicalov = MedicalOV(args.model)
medicalov.load_model(args.device)
end = time.perf_counter()
print(f"Model loading completed in {end-start:.2f} seconds")

# WIP, for now we hardcode the title to the two tested models
title = "google/medgemma-4b-it" if "gemma" in args.model else "microsoft/maira-2"
launch_demo(title)
