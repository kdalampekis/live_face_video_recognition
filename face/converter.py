import tensorflow as tf
# Load the saved model
loaded_model = tf.saved_model.load('/Users/kostasbekis/live_face_recognition/face/results')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('/Users/kostasbekis/live_face_recognition/face/results')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
