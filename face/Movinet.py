import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import layers
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np

import tensorflow as tf

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

      output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int16))
      train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 10, training=True),
                                                output_signature=output_signature)

      val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 10),
                                              output_signature=output_signature)

      model_id = 'a0'
      resolution = 224

      tf.keras.backend.clear_session()

      backbone = movinet.Movinet(model_id=model_id)
      backbone.trainable = False

      # Set num_classes=600 to load the pre-trained weights from the original model
      model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
      model.build([None, None, None, None, 3])



      checkpoint_dir = f'movinet_{model_id}_base'
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
      checkpoint = tf.train.Checkpoint(model=model)
      status = checkpoint.restore(checkpoint_path)
      status.assert_existing_objects_matched()

      def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
          """Builds a classifier on top of a backbone model."""
          model = movinet_model.MovinetClassifier(
              backbone=backbone,
              num_classes=num_classes)
          model.build([batch_size, num_frames, resolution, resolution, 3])

          return model

      model = build_classifier(batch_size, num_frames, resolution, backbone, 10)

num_epochs = 2

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

results = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    verbose=1)