# reference:
#     https://www.tensorflow.org/text/tutorials/classify_text_with_bert
import timeit
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
tf.config.optimizer.set_jit(True)
tf.config.experimental_run_functions_eagerly(False)
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model =build_classifier_model()
classifier_model.compile(optimizer='adam',
                         jit_compile=True)

def test(text_test):
   
    
    return classifier_model(tf.constant(text_test))
  
   
m=[]
TIMES=6
for i in range(TIMES):
    n=timeit.timeit(lambda:test( ['this is such an amazing movie!']), number=10)
    m.append(n)
    

    print("execution time of bert model",n)
print("compile time",m[0]-m[1])