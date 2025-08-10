import re
import time
import warnings

warnings.filterwarnings("ignore")

from optimum.intel import OVModelForVisualCausalLM
from PIL import Image
from transformers import AutoProcessor, logging

logging.set_verbosity_error()


class MedicalOV:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = None
        self.model = None
        self.processor = None
        self.trust_remote_code = "maira" in model_path

    def load_model(self, device):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.model = OVModelForVisualCausalLM.from_pretrained(self.model_path, device=device, trust_remote_code=self.trust_remote_code)

    def prepare_inputs_image(self, image, question):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": image}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs

    def run_inference_image(self, image, question):
        inputs = self.prepare_inputs_image(image, question)
        start = time.perf_counter()
        ov_output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
        )
        end = time.perf_counter()
        print(f"Inference duration: {end-start:.2f} seconds")
        input_length = inputs["input_ids"].shape[-1]
        ov_output_ids = ov_output_ids[0][input_length:]

        answer = self.processor.decode(ov_output_ids, skip_special_tokens=True)
        # workaround: strip tags from models that return structured output. #TODO support structured output
        text_answer = re.sub(r"<[^>]+>", "", answer)
        print(f"Answer: {text_answer}")
        return text_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    parser.add_argument("--model", help="path to OpenVINO model directory")
    parser.add_argument("--device", choices=["GPU", "CPU"], help="Inference device (CPU or GPU)")
    args = parser.parse_args()

    print(f"Loading {args.model} to {args.device}")
    start = time.perf_counter()
    medicalov = MedicalOV(args.model)
    medicalov.load_model(args.device)
    end = time.perf_counter()
    print(f"Model loading completed in {end-start:.2f} seconds")

    image = Image.open(args.image)
    medicalov.run_inference_image(image, "Describe this X-Ray")
