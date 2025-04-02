import tensorflow as tf
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model

# Create an ImageDataGenerator for videos
video_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

# Define train and validation generators
train_generator = video_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='binary', shuffle=True)
val_generator = video_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='binary', shuffle=False)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top (fully connected) layers for action recognition
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # 1 output neuron for binary action detection (pill taking or not)


# Create a new model for fine-tuning
model = Model(inputs=base_model.input, outputs=x)

# Freeze the initial layers of MobileNetV2
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your dataset
model.fit(train_dataset, epochs=20, validation_data=val_dataset)

for layer in base_model.layers:
    if layer.name.startswith('block_14'): # Unfreeze block_14 and all higher blocks
        layer.trainable = True
    else:
        layer.trainable = False

# Compile the model after unfreezing some layers (optional)
# You may need to lower the learning rate for fine-tuning to prevent overfitting
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training the model with a lower learning rate (optional)
model.fit(train_dataset, epochs=20, validation_data=val_dataset)

