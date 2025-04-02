import tensorflow as tf

from frame_generator import FrameGenerator
from utils import build_classifier

# settings
num_frames_per_video = 20
batch_size = 16
frame_resolution = 224
epochs = 2
learning_rate = 0.001

# output signature
output_signature = (tf.TensorSpec(shape=(None, None, None, 3),
                                  dtype=tf.float32),
                    tf.TensorSpec(shape=(),
                                  dtype=tf.int16)
                    )

# set up data generators
train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        path='videos/train',
        n_frames=num_frames_per_video,
        training=True),
    output_signature=output_signature
)
train_ds = train_ds.batch(batch_size)


val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(
        path='videos/val',
        n_frames=num_frames_per_video),
    output_signature=output_signature
)
val_ds = val_ds.batch(batch_size)

# set up model
model = build_classifier(
    batch_size=batch_size,
    num_frames=num_frames_per_video,
    resolution=frame_resolution,
    num_classes=2
)

# set up training
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

results = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1
)

# Save the model
tf.saved_model.save(model, '//results')
