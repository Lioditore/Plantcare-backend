
import cv2
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, pred_index=None):
    base = model.layers[0]                       # MobileNetV2
    conv_model = tf.keras.Model(base.input, base.output)

    head_input = tf.keras.Input(shape=base.output.shape[1:])
    x = head_input
    for layer in model.layers[1:]:               # GAP, dropout, dense, dropout, dense
        x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_out = conv_model(img_array, training=False)
        tape.watch(conv_out)
        preds = head_model(conv_out, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index

def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8(color * alpha + img_rgb * (1 - alpha))
