# Day 06

**Topic:** PyTorch, Model File Formats, ONNX & TensorRT
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Surveyed the model serialization landscape from .pt/pickle to SafeTensors to ONNX to TensorRT engines. Implemented format conversions, analyzed size and safety tradeoffs, and mapped the full PyTorch-to-TensorRT compilation pipeline.

## Key insight
SafeTensors is strictly better than pickle for model distribution: same size, faster cold load via lazy seeking, and zero code-execution attack surface.

## Code / experiment
Notebook: [`pytorch-formats-onnx-trt.ipynb`](./pytorch-formats-onnx-trt.ipynb)
Key demo: Format size comparison and TensorRT compilation pipeline walkthrough

## References
- *Inference Engineering* Ch 4.2 (Philip Kiely, Baseten Books 2026)
- HuggingFace SafeTensors specification
- NVIDIA TensorRT Developer Guide
