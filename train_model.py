import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

# ---------------- Paths ---------------- #
train_dir = "dataset/train"
img_size = (128, 128)   # smaller image size for speed
batch_size = 32
epochs = 15             # max epochs (early stopping will stop earlier)

# ---------------- Data Generators ---------------- #
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# ---------------- Base Model ---------------- #
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

# ---------------- Build Model ---------------- #
input_layer = Input(shape=(128,128,3))
x = base_model(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)

# ---------------- Phase 1: Train Top Layers ---------------- #
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
]

print("🔹 Phase 1: Training only top layers...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,          # short warm-up
    verbose=1,
    callbacks=callbacks
)

# ---------------- Phase 2: Fine-Tune Last Layers ---------------- #
print("🔹 Phase 2: Fine-tuning last 30 layers of MobileNetV2...")
base_model.trainable = True
for layer in base_model.layers[:-30]:   # freeze all but last 30
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # smaller LR
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks
)

# ---------------- Save Model ---------------- #
os.makedirs("model", exist_ok=True)
model.save("model/flower_model.h5")
np.save("model/class_labels.npy", np.array(list(train_gen.class_indices.keys())))

print("✅ Optimized MobileNetV2 model trained & saved with class labels.")
