# # https://keras.io/api/applications/resnet/
import tensorflow as tf
import timeit
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

ResNet50_model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

ResNet101_model = tf.keras.applications.ResNet101(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

ResNet152_model = tf.keras.applications.ResNet152(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

ResNet50V2_model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

ResNet101V2_model = tf.keras.applications.ResNet101V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

ResNet152V2_model = tf.keras.applications.ResNet152V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)



# def test_(times,model):
#     for i in range(times):
#         model.compile(optimizer='adam',
#                             jit_compile=True)


# times = 50
# print(f"compile time of ResNet50_model",timeit.timeit(lambda:test_(times,ResNet50_model), number=10)/times)
# print(f"compile time of ResNet101_model",timeit.timeit(lambda:test_(times,ResNet101_model), number=10)/times)
# print(f"compile time of ResNet152_model",timeit.timeit(lambda:test_(times,ResNet152_model), number=10)/times)
# print(f"compile time of ResNet50V2_model",timeit.timeit(lambda:test_(times,ResNet50V2_model), number=10)/times)
# print(f"compile time of ResNet101V2_model",timeit.timeit(lambda:test_(times,ResNet101V2_model), number=10)/times)
# print(f"compile time of ResNet152V2_model",timeit.timeit(lambda:test_(times,ResNet152V2_model), number=10)/times)



img_path = '/home/lyc_build/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

conv_layer = tf.keras.layers.Conv2D(100, 3)
model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(100,3)]
)



@tf.function(jit_compile=True)
def call_model(image,model):
    model(image)

first_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet50_model), number=10)
second_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet50_model), number=10)
print('compile and execute:',first_encounter_time_usage)
print('Only execute:       ',second_encounter_time_usage)
print('Only compile:       ',first_encounter_time_usage - second_encounter_time_usage)


first_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet101_model), number=10)
second_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet101_model), number=10)
print('compile and execute:',first_encounter_time_usage)
print('Only execute:       ',second_encounter_time_usage)
print('Only compile:       ',first_encounter_time_usage - second_encounter_time_usage)





first_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet152_model), number=10)
second_encounter_time_usage = timeit.timeit(lambda: call_model(x,ResNet152_model), number=10)
print('compile and execute:',first_encounter_time_usage)
print('Only execute:       ',second_encounter_time_usage)
print('Only compile:       ',first_encounter_time_usage - second_encounter_time_usage)
