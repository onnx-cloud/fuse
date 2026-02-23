# Simple TF -> ONNX export example
import tensorflow as tf
import tf2onnx

m = tf.keras.Sequential(
    [tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(10)]
)
spec = (tf.TensorSpec((None, 32), tf.float32, name="x"),)
model_proto, _ = tf2onnx.convert.from_keras(m, input_signature=spec, opset=14)
with open("onnx/playbook/tf_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
print("Exported -> onnx/playbook/tf_model.onnx")
