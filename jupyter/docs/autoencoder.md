# Autoencoder

Tiny linear autoencoder that reproduces a 3-element input using identity-like weights. Useful to show how Fuse maps simple neural computations to ONNX.

## For beginners
- This model reads a vector of size 3 and returns a vector of size 3.
- It uses two linear layers with ReLU in between. In this toy example the weights are set to identity so the output equals the input.
- Try changing the input in the inline `@proof` to see what the model returns.

## For experts
- Ops used: `Reshape`, `MatMul`, `Add`, `Relu`, `Reshape`.
- The example demonstrates parameter initializers (typed constants), short computation graphs, and reshaping to add batch dims.
- Shapes: input `f32[3]` -> reshape to `1x3` -> matmul with `3x3` weight -> output reshaped back to `[3]`.

## Run & export
- Run inline tests: `python -m pytest tests/test_cookbook.py -k autoencoder -q`
- Export to ONNX: `./.venv/bin/fuse onnx -f jupyter/cookbook/autoencoder.fuse -o onnx/cookbook/autoencoder.onnx`

## Notes
- Keeps everything explicit: typed constants for weights/biases and INT64 shape tensors for `Reshape` to match ONNX signatures.
- Good starting point for learning how `MatMul` and `Reshape` patterns lower into ONNX nodes.