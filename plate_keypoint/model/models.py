
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow.keras.layers as L
import tensorflow.keras.optimizers as optim


def default_keypoint_model_224(num_keypoints=4) -> Model:
    output_filters = num_keypoints * 2
    input_placeholder = Input((224, 224, 3))
    mobilenet: Model = MobileNetV3Small(include_top=False)
    features = Model(inputs=mobilenet.inputs, outputs=mobilenet.get_layer(index=-7).output)
    head = Sequential([
        L.Dropout(0.5),
        L.SeparableConv2D(output_filters, 5, activation="relu"),
        L.SeparableConv2D(output_filters, 3, activation="sigmoid"),
        L.Reshape((-1, output_filters))
    ])
    x = input_placeholder
    x = features(x)
    out = head(x)
    model = Model(inputs=[input_placeholder], outputs=[out])

    optimizer = optim.Adam(1e-4)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    return model
