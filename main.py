import tensorflow as tf
import csv
import keras
import cvalib as cva

# params
BATCH_SIZE = 128
NUM_CLASSES = 50
# MODEL can be 'baseline', 'convmixer'
MODEL = 'convmixer'
# TRAINING_MODE can be 'full', 'val'
TRAINING_MODE = 'val' 
EPOCHS = 20

# metadata
labels, path_files_images = cva.read.get_train_labels_and_path_files_images(
    path_file_labels='data/train_labels.csv',
    path_data='data/train'
)
path_files_images_test = cva.read.get_path_files_images(path_data='data/test')

# remap labels to labels ids for easier one-hot encoding
labels2ids = cva.processing.mapping_labels_to_labels_ids(labels=labels, output_path='data')    
labels = [labels2ids[label] for label in labels]
        
# check labels frequency (possible imbalanced data problem)
labels_frequency = {}
for label in labels:
    labels_frequency[label] = labels_frequency.get(label, 0) + 1
if min(labels_frequency.values()) != max(labels_frequency.values()):
    print(f'WARNING: possible imbalanced data problem!!!')

# split data into training and validation set
labels_train, labels_val, path_files_images_train, path_files_images_val = cva.processing.train_val_split(
    labels=labels,
    path_files_images=path_files_images,
    val_perc=0.15
)

# tensorflow input pipelines
ds_full = (
    tf.data.Dataset.from_tensor_slices((path_files_images, labels))
    .shuffle(buffer_size=len(labels))
    .map(cva.read.read_png_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(cva.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_train = (
    tf.data.Dataset.from_tensor_slices((path_files_images_train, labels_train))
    .shuffle(buffer_size=len(labels_train))
    .map(cva.read.read_png_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(cva.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_val = (
    tf.data.Dataset.from_tensor_slices((path_files_images_val, labels_val))
    .shuffle(buffer_size=len(labels_val))
    .map(cva.read.read_png_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(cva.processing.data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices((path_files_images_test))
    .map(cva.read.read_png_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# setup model checkpoints
callback_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/_checkpoint.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# train model
if MODEL == 'baseline':
    # get model
    model = cva.models.build_baseline(input_shape=(32, 32, 3), num_classes=NUM_CLASSES)

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    # training
    if TRAINING_MODE == 'full':
        history = model.fit(ds_full, epochs=EPOCHS, callbacks=[callback_checkpoint])
    elif TRAINING_MODE == 'val':
        history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_val, callbacks=[callback_checkpoint])

    # best params
    EPOCHS = 6
    TRAINING_MODE = 'full'

elif MODEL == 'convmixer':
    model = cva.models.build_convmixer(
        input_shape=(32, 32, 3),
        patches_embedding_dimension=256,
        depth=8,
        patch_size=2,
        kernel_size=5,
        num_classes=NUM_CLASSES,
        rescale_inputs=True,
        dropout_rate=None
    )
        
    # compile model
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    # training
    if TRAINING_MODE == 'full':
        history = model.fit(ds_full, epochs=EPOCHS, callbacks=[callback_checkpoint])
    elif TRAINING_MODE == 'val':
        history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_val, callbacks=[callback_checkpoint])

# save training history
cva.models.save_plot_training_history(history=history.history, output_path='training-history')

# load and save best model
model.load_weights('models/_checkpoint.keras')
model.save(f'models/{MODEL}.keras')

# predictions on test data
predictions = []
for batch_test in ds_test:
    predictions.append(model.predict_on_batch(batch_test))
predictions = tf.concat(predictions, axis=0)
predictions = tf.math.argmax(predictions, axis=-1)
predictions = predictions.numpy().tolist()

# write subsmission file
ids2labels = {label_id: label for label, label_id  in labels2ids.items()}
with open('submission.csv', 'w') as file:
    csv_writer = csv.writer(file, delimiter=',', quotechar=None)
    csv_writer.writerow(['image_name', 'label'])
    for path_file_image_test, prediction in zip(path_files_images_test, predictions):
        label = ids2labels[prediction]
        filename = path_file_image_test.split('/')[-1].replace('.png', '')
        csv_writer.writerow([filename, label])
