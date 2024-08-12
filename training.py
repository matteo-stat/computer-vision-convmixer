import keras
import tensorflow as tf
import convmixerlib as cm

# some parameters
NUM_CLASSES = 10
BATCH_SIZE=64
SEED = 1993
FULL_TRAINING = False

# load cifar10 data
(images, labels), (images_test, labels_test) = keras.datasets.cifar10.load_data()

# convert images data to float
images = tf.cast(images, dtype=tf.float32)
images_test = tf.cast(images_test, dtype=tf.float32)

# convert labels to one hot encoding
labels = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.float32)
labels = tf.reshape(labels, shape=(labels.shape[0], NUM_CLASSES))

# split data in training and validation sets
(images_train, labels_train), (images_val, labels_val) = cm.processing.train_val_split(images, labels, val_perc=0.1)

# data pipelines
ds_full = (
    tf.data.Dataset.from_tensor_slices((images, labels))
    .shuffle(buffer_size=len(labels))
    .batch(batch_size=BATCH_SIZE)
    .map(cm.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_train = (
    tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    .shuffle(buffer_size=len(labels_train))
    .batch(batch_size=BATCH_SIZE)
    .map(cm.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_val = (
    tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    .shuffle(buffer_size=len(labels_val))
    .batch(batch_size=BATCH_SIZE)
    .map(cm.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices(images_test)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# build convmixer model
model = cm.models.build_convmixer_classifier(
    input_shape=(32, 32, 3),
    patches_embedding_dimension=256,
    depth=8,
    patch_size=2,
    kernel_size=5,
    num_classes=10,
    rescale_inputs=True,
    dropout_rate=0.15
)

# model summary
model.summary()

# compile model
model.compile(
    optimizer=keras.optimizers.AdamW(weight_decay=0),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# fit the model first using validation set to assess model performances on test data
if not FULL_TRAINING:
    history = model.fit(
        ds_train,
        epochs=30,
        validation_data=ds_val,
    )
# fit the model again but on the whole available data using the best hyperparameters previously found
else:
    history = model.fit(
        ds_full,
        epochs=25,
    )

# save training plot
cm.plots.plot_training_history(show=False, output_path='output-plots')

# predictions on test data
predictions = []
for batch_test in ds_test:
    predictions.append(model.predict_on_batch(batch_test))
predictions = tf.concat(predictions, axis=0)
predictions = tf.math.argmax(predictions, axis=-1)
predictions = tf.cast(predictions, dtype=tf.uint8)
predictions = tf.expand_dims(predictions, axis=1)

# accuracy on test data
accuracy = tf.cast(labels_test == predictions, dtype=tf.uint8)
accuracy = accuracy.numpy().sum() / len(accuracy)
print(f'test_categorical_accuracy: {accuracy:.4f}')
