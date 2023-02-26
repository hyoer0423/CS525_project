# import tensorflow as tf
# import tensorflow_gnn as tfgnn
# import timeit
# # https://github.com/tensorflow/gnn
# # https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html
# # Model hyper-parameters:
# h_dims = {'user': 256, 'movie': 64, 'genre': 128}

# # Model builder initialization:
# gnn = tfgnn.keras.ConvGNNBuilder(
#     lambda edge_set_name: WeightedSumConvolution(),
#     lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
#         tf.keras.layers.Dense(h_dims[node_set_name]))
# )

# # Two rounds of message passing to target node sets:
# gnn_model = tf.keras.models.Sequential([
#     gnn.Convolve({'genre'}),  # sends messages from movie to genre
#     gnn.Convolve({'user'}),  # sends messages from movie and genre to users
#     tfgnn.keras.layers.Readout(node_set_name="user"),
#     tf.keras.layers.Dense(1)
# ])

# #tef
# def test_(times,model):
#     model( tf.ones([1,1433])   )
#     # for i in range(times):
#     #     model.compile(optimizer='adam',
#     #                         jit_compile=True)

# times = 50
# print(f"compile time of gnn_model",timeit.timeit(lambda:test_(times,gnn_model), number=10)/times)
# print(f"compile time of gnn_model",timeit.timeit(lambda:test_(times,gnn_model), number=10)/times)
# print(f"compile time of gnn_model",timeit.timeit(lambda:test_(times,gnn_model), number=10)/times)




import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("Citations shape:", citations.shape)

citations.sample(frac=1).head()

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
)
print("Papers shape:", papers.shape)

print(papers.sample(5).T)

print(papers.subject.value_counts())

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])


train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()



def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]


def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")


baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()


baseline_model(  tf.ones([1,1433])        )



import timeit
@tf.function(jit_compile=True)
def train_mnist():
    baseline_model(  tf.ones([1,1433])        )
   
print('compile and execute:',timeit.timeit(lambda: train_mnist(), number=10))
print('Only execute:       ',timeit.timeit(lambda: train_mnist(), number=10))
print('Only execute:       ',timeit.timeit(lambda: train_mnist(), number=10))
# @0.12
