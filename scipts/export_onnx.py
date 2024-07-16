import os
import torch
from PIL import Image
import numpy as np
import open_clip
import clip

text1 = open_clip.tokenize(["a diagram", "a dog", "a cat"])
text2 = clip.tokenize(["a diagram", "a dog", "a cat"])
text_input = clip.tokenize(["a diagram"])

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
                                                             pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
model.output_dict = True
image = preprocess(Image.open(r"E:\TypesExamples\000000001296.jpg")).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text1)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)

import warnings
import torch

"""
https://dev.to/andreygermanov/export-segment-anything-neural-network-to-onnx-the-missing-parts-43c8#export_sam_decoder

"""
onnx_model_path = "metaclip_text_encoder.onnx"

dummy_inputs = {"image": None,
                "text": text_input}
output_names = ["text_features", "logit_scale"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
        )


def run_onnx():
    import onnxruntime
    sess = onnxruntime.InferenceSession('weights/metaclip_text_encoder.onnx', providers=['CPUExecutionProvider'])
    result = sess.run(["text_features"], {"image": [text_input[0].numpy()]})

    for x, y in zip(result[0][0], text_features[0].detach().numpy()):
        print(x, y)
        print(x - y)


def split_onnx_model():
    onnx_model_path = r'weights/metaclip_text_encoder.onnx'
    pref, ext = os.path.splitext(onnx_model_path)
    max_size_mb = 10
    # Read the entire ONNX model as bytes
    with open(onnx_model_path, "rb") as f:
        model_bytes = f.read()

    total_len = len(model_bytes)
    model_size_mb = total_len / 1024 / 1024  # Convert MB to bytes

    # Calculate the size of each part
    num_parts = int(np.ceil(model_size_mb / max_size_mb))  # Round up division

    # Split into parts
    parts_len = 0
    for i_part in range(num_parts):
        start = (i_part) * max_size_mb * 1024 * 1024
        end = (i_part + 1) * max_size_mb * 1024 * 1024
        part_filename = f"{pref}_{i_part:02}{ext}"
        with open(part_filename, "wb") as f:
            f.write(model_bytes[start:end])
            parts_len += len(model_bytes[start:end])
    print(parts_len)
    print(total_len)
