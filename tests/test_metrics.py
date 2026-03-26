# # Code permettant de vérifier que les métriques DCT sont correctes
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import numpy as np

from src.analysis.filter_metrics import (
    get_filter,
    getSobelTF,
    getSymAntiSymTF,
    getDominantAngle,
    topKfilters,
    topKchannels,
)

model_path = r"C:\Users\Momoa\Downloads/baseline_resnet20_cifar10_he/model.keras"
model = tf.keras.models.load_model(model_path)

print("\n=== MODEL LOADED ===")
print(model.name)

print("\n=== 3x3 CONV LAYERS USED BY get_filter ===")
conv_layers = []
for l in model.layers:
    if 'conv2d' in str(type(l)).lower():
        if l.kernel_size == (3, 3):
            conv_layers.append(l)

num_layers = len(conv_layers)
print("Number of valid conv layers:", num_layers)

for layer_num in range(num_layers):   
    print(f"\n==============================")
    print(f"TESTING LAYER {layer_num}")
    print(f"==============================")

    filters = get_filter(model, layer_num)
    print("filters shape:", filters.shape)

    theta = getSobelTF(filters)
    print("theta shape:", theta.shape)

    sym, anti = getSymAntiSymTF(filters)
    err = np.max(np.abs((sym + anti) - filters))
    print("reconstruction error:", err)

    vec, domTheta = getDominantAngle(filters)
    print("vec shape:", vec.shape)
    print("domTheta shape:", domTheta.shape)
    print("domTheta sample:", domTheta[:5])

    top_filters = topKfilters(model, layer_num=layer_num, k=10)
    print("top 10% filters:", top_filters)

    f_num = 0
    top_channels = topKchannels(model, layer_num=layer_num, f_num=f_num, k=20)
    print(f"top 20% channels for filter {f_num}:", top_channels)

# layer_num = 0
# print(f"\n=== TESTING LAYER {layer_num} ===")
# filters = get_filter(model, layer_num)

# print("filters shape:", filters.shape)
# print("filters dtype:", filters.dtype)

# print("\n=== TEST getSobelTF ===")
# theta = getSobelTF(filters)
# print("theta shape:", theta.shape)
# print("theta sample:", theta[:5])

# print("\n=== TEST getSymAntiSymTF ===")
# sym, anti = getSymAntiSymTF(filters)
# print("sym shape:", sym.shape)
# print("anti shape:", anti.shape)

# reconstruction_error = np.max(np.abs((sym + anti) - filters))
# print("max reconstruction error:", reconstruction_error)

# print("\n=== TEST getDominantAngle ===")
# vec, domTheta = getDominantAngle(filters)
# print("vec shape:", vec.shape)
# print("domTheta shape:", domTheta.shape)
# print("domTheta sample:", domTheta[:5])

# print("\n=== TEST topKfilters ===")
# top_filters = topKfilters(model, layer_num=layer_num, k=10)
# print("top 10% filters:", top_filters)

# print("\n=== TEST topKchannels ===")
# top_channels = topKchannels(model, layer_num=layer_num, f_num=0, k=50)
# print("top 50% channels for filter 0:", top_channels)