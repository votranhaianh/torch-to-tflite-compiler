# PyTorch to TFLite Compiler ⚡📉

A streamlined toolkit for deep learning engineers to bridge the gap between PyTorch research and edge deployment. This repository automates the conversion of complex `nn.Module` architectures into highly optimized TensorFlow Lite (`.tflite`) formats.

## 🌟 Features
- **ONNX Bridge:** Reliable conversion from PyTorch -> ONNX -> TensorFlow -> TFLite.
- **Post-Training Quantization:** Supports INT8, FP16, and Dynamic Range Quantization to reduce model size by up to 4x with minimal accuracy loss.
- **Graph Optimization:** Automatically fuses layers (e.g., Conv2d + BatchNorm2d) prior to conversion.
- **Validation Suite:** Built-in cosine similarity checks to ensure the converted TFLite model matches PyTorch output tensors.

## 🛠️ Usage
Perfect for deploying models to mobile devices, embedded systems, or cloud environments requiring extremely low latency.

```bash
python convert.py --model my_resnet.pth --quantize int8 --output optimized_model.tflite
```