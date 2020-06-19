import tensorflow as tf
import tensorflow_datasets as tfds

## Load datasets as train and test
dataset, info = tfds.load('beans', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

print(f"Number of training examples {info.splits['train'].num_examples}")
print(f"Number of test examples {info.splits['test'].num_examples}")

## Image augmentation functions
def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size = (150,150))
    return image, label

def augment(image, label):
    image, label = convert(image, label)
    image = tf.image.resize_with_crop_or_pad(image, 160, 160)
    image = tf.image.random_crop(image, size = (150,150,3))
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image, label

BATCH_SIZE = 64
NUM_EXAMPLES = info.splits['train'].num_examples
AUTOTUNE = tf.data.experimental.AUTOTUNE

augmented_train_batches = train_dataset\
                          .take(NUM_EXAMPLES)\
                           .cache()\
                            .shuffle(NUM_EXAMPLES // 4)\
                            .map(augment, num_parallel_calls = AUTOTUNE)\
                            .batch(BATCH_SIZE)\
                            .prefetch(AUTOTUNE)

augmented_test_batches = test_dataset\
                         .map(convert, num_parallel_calls = AUTOTUNE)\
                         .batch(10)\
                         .prefetch(AUTOTUNE)

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = 'adam', metrics = ['accuracy'])

    return model

aug_model = make_model()
# print(aug_model.summary())

history = aug_model.fit(augmented_train_batches, epochs = 10, validation_data= augmented_test_batches)