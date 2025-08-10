# OpenVINO Medical AI demo

> [!CAUTION]
> This repository is work in progress!

This repository contains a [gradio](https://www.gradio.app) demo for running inference on medical visual language models
with [OpenVINO](https://github.com/openvinotoolkit/openvino) using OpenVINO's integration in Hugging Face transformers,
[optimum-intel](https://github.com/huggingface/optimum-intel).

This is example code to build upon, not intended to be production ready inference code.

> [!NOTE]
> The inference code is adapted from the
> [medgemma-4b-it model card](https://huggingface.co/google/medgemma-4b-it). This prompt format works
> with other medical visual visual language models, but results may not be optimal. You should adapt the script for other models.


## Usage

### Install dependencies

Install or upgrade `optimum[openvino] and gradio`. This command updates optimum-intel and all dependencies,
including transformers. Note that medgemma needs transformers 4.52 or later.

```
pip install --upgrade --upgrade-strategy eager optimum[openvino] gradio
```

### Export a model

By default the model will be exported with INT8 weights

```
optimum-cli export openvino -m google/medgemma-4b-it medgemma-4b-it-ov
```

### Run the Gradio demo

Use GPU if you have a recent Intel laptop with iGPU or an Intel discrete GPU. Otherwise use CPU.

```
python app.py medgemma-4b-it-ov GPU
```

It may take a while to compile the model. This will be faster on subsequent runs.

The output from app.py will show `Running on local URL:  http://127.0.0.1:7790`. Click on this link to open the demo in your browser (or type in this URL manually).

### Run inference in a script

```
python medical_inference_openvino.py xray.jpg --model medgemma-4b-it-ov
```

xray.jpg source: https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png (public domain, CC0 license)
