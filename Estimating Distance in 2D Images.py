#!/usr/bin/env python
# coding: utf-8

# # ARRAY 1

# In[1]:


pip uninstall numpy h5py -y


# In[2]:


pip install --no-binary=h5py h5py


# In[3]:


pip uninstall numpy h5py -y


# In[4]:


pip install ultralytics opencv-python pandas


# In[5]:


pip install albumentations opencv-python


# In[6]:


pip install albumentations


# In[7]:


pip uninstall -y numpy


# In[8]:


pip list | findstr numpy


# In[9]:


pip list | findstr h5py


# In[10]:


import os
import numpy as np
folder_path = r"A:/project/array1/array"
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

print(f"Found {len(file_names)} npy files.")
print(file_names[:10])  # show first few file names


# In[11]:


import os
import numpy as np

folder_path = r"A:/project/array1/array"  # or your correct path
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

# Get shapes of all images
shapes = [np.load(os.path.join(folder_path, f)).shape for f in file_names]
heights = [s[0] for s in shapes]
widths = [s[1] for s in shapes]

max_height = max(heights)
max_width = max(widths)

print("Max height:", max_height)
print("Max width:", max_width)


# In[12]:


def pad_image(img, target_shape):
    padded = np.zeros(target_shape, dtype=np.float32)
    h, w = img.shape
    padded[:h, :w] = img
    return padded

# Load and pad all images
X = []
for f in file_names:
    img = np.load(os.path.join(folder_path, f))
    padded_img = pad_image(img, (max_height, max_width))
    X.append(padded_img)

X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension
print("Final shape:", X.shape)


# In[13]:


import pandas as pd

labels_df = pd.read_csv("A:/project/array1/labels1.csv")
y = labels_df["label"].values  # or labels_df.iloc[:, 1].values
print("Labels shape:", y.shape)


# In[14]:


np.save("A:/project/X_array1_padded.npy", X)
np.save("A:/project/y_array1.npy", y)


# In[15]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 1: Load data
# ----------------------------------------
X = np.load("A:/project/X_array1_padded.npy")
y = np.load("A:/project/y_array1.npy")

# Resize and normalize
X_resized = tf.image.resize(X, (256, 256)).numpy()
X_resized = X_resized / np.max(X_resized)

# Convert grayscale to RGB
X_rgb = np.repeat(X_resized, 3, axis=-1)  # (500, 256, 256, 3)

print("Input shape for ResNet50V2:", X_rgb.shape)

# ----------------------------------------
# Step 2: Build ResNet50V2 model with fine-tuning
# ----------------------------------------
base_model = ResNet50V2(include_top=False, input_shape=(256, 256, 3), weights='imagenet', pooling='avg')

# Unfreeze last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)  # Regression output
])

# Compile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mae', metrics=['mae'])
model.summary()

# ----------------------------------------
# Step 3: Train
# ----------------------------------------
history = model.fit(X_rgb, y, epochs=30, batch_size=16, validation_split=0.1)

# ----------------------------------------
# Step 4: Predict and visualize
# ----------------------------------------
preds = model.predict(X_rgb).flatten()

# Scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(y, preds, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("True Distance")
plt.ylabel("Predicted Distance")
plt.title("ResNet50V2 - Predicted vs Actual")
plt.grid(True)
plt.show()

# Training curve
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# Step 5: Save model and predictions
# ----------------------------------------
model.save("A:/project/resnet50v2_finetuned_model.h5")
pd.DataFrame({'True': y, 'Predicted': preds}).to_csv("A:/project/resnet50v2_predictions.csv", index=False)


# In[17]:


for i in range(20):
    print(f"Sample {i}: True = {y[i]:.2f}, Predicted = {preds[i]:.2f}")


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, monitor='val_mae', restore_best_weights=True),
    ModelCheckpoint("A:/project/best_resnet_model.h5", save_best_only=True, monitor='val_mae')
]

# Re-train
model.fit(X_rgb, y, epochs=50, batch_size=16, validation_split=0.1, callbacks=callbacks)


# In[19]:


preds = model.predict(X_rgb).flatten()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(y, preds, alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("True Distance")
plt.ylabel("Predicted Distance")
plt.title("ResNet50V2: Predicted vs Actual")
plt.grid(True)
plt.show()


# In[20]:


import seaborn as sns
errors = np.abs(y - preds)

sns.histplot(errors, kde=True, bins=30)
plt.title("Absolute Error Distribution")
plt.xlabel("Error (|True - Pred|)")
plt.grid(True)
plt.show()


# In[21]:


import pandas as pd
df = pd.DataFrame({'True': y, 'Predicted': preds})
df.to_csv("A:/project/resnet50v2_predictions.csv", index=False)
model.save("A:/project/resnet50v2_final_model.h5")


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y, preds)
mse = mean_squared_error(y, preds)
rmse = mse ** 0.5  # manually compute RMSE

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# # ✅ Final Evaluation Metrics of Array1
# #### Metric	Value 
# #### MAE (Mean Absolute Error)	4.02 pixels
# #### RMSE (Root Mean Squared Error)	6.67 pixels
# 
# # 📊 Interpretation
# #### MAE of 4.02 means on average, your model's predicted distance is only ~4 pixels off from the true value.
# 
# #### RMSE of 6.67 suggests there are some larger errors, but most are close to the ground truth.
# 
# #### This is a massive improvement from your original models (which were stuck around MAE ~45).

# # ARRAY 2

# ### Option A - Cropped Region Model

# In[23]:


print(labels_df.columns)


# In[24]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Paths
array2_path = "A:/project/arrays2"
image_dir = os.path.join(array2_path, "entire scan")
label_path = os.path.join(array2_path, "labels2.csv")

# Load labels
labels_df = pd.read_csv(label_path)

X = []
y = []

for i in tqdm(range(len(labels_df))):
    row = labels_df.iloc[i]
    fname = f"{int(row['ID'])}.npy"
    full_path = os.path.join(image_dir, fname)
    
    if not os.path.exists(full_path):
        print(f"Missing file: {fname}")
        continue

    image = np.load(full_path)

    # Extract coordinates
    y_start = int(row['location'])
    y_end = y_start + int(row['Width'])  # Capital W here
    x1 = int(row['x1'])
    x2 = int(row['x2'])


    # Crop jump region
    cropped = image[y_start:y_end, x1:x2]

    # Skip empty or invalid crops
    if cropped.size == 0:
        continue

    # Resize to fixed shape (e.g., 64x128)
    resized = tf.image.resize(cropped[..., np.newaxis], (64, 128)).numpy()
    X.append(resized)
    y.append(row['label'])

X = np.array(X)
y = np.array(y)

print("Final shape:", X.shape)


# # ✅ Cropped Image Sample Visualization Code

#  # Cropped region model (Option A) with ResNet50V2 to predict both x1 and x2 values.

# In[26]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Paths ---
array2_path = "A:/project/arrays2"
image_dir = os.path.join(array2_path, "entire scan")
label_path = os.path.join(array2_path, "labels2.csv")

# --- Load Labels ---
labels_df = pd.read_csv(label_path)
original_width = 1000
target_width = 128  # final image width after cropping/resizing
target_height = 64

# --- Prepare Image Crop ---
X_crop = []
y_scaled = []

for _, row in labels_df.iterrows():
    img_path = os.path.join(image_dir, f"{int(row['ID'])}.npy")
    if not os.path.exists(img_path):
        continue

    img = np.load(img_path)  # shape (H, W)
    y_pos = int(row['location'])  # vertical start
    width = int(row['Width'])     # height of jump
    x1 = float(row['x1'])
    x2 = float(row['x2'])

    # Crop vertically (with bounds checking)
    y1 = max(0, y_pos - target_height // 2)
    y2 = min(img.shape[0], y1 + target_height)
    crop = img[y1:y2, :]  # crop full width, limited height

    # Resize to (64, 128)
    crop_resized = tf.image.resize(crop[..., np.newaxis], [target_height, target_width]).numpy()
    crop_resized = crop_resized / 255.0  # normalize
    X_crop.append(crop_resized)

    # Scale x1, x2 from original_width → 0–1 → then to 0–128 for resized image
    x1_scaled = x1 / original_width * target_width
    x2_scaled = x2 / original_width * target_width
    y_scaled.append([x1_scaled, x2_scaled])

X_crop = np.array(X_crop)                    # (N, 64, 128, 1)
X_rgb = np.repeat(X_crop, 3, axis=-1)        # (N, 64, 128, 3)
y_scaled = np.array(y_scaled).astype("float32")  # (N, 2)

# --- Split ---
X_train, X_val, y_train, y_val = train_test_split(X_rgb, y_scaled, test_size=0.2, random_state=42)

# --- Build ResNet Model ---
base = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(64, 128, 3), weights='imagenet')
base.trainable = False  # freeze initially

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2)  # Predict x1, x2
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# --- Train ---
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint("resnet_crop_x1x2.h5", save_best_only=True)
                    ])


# In[27]:


# Load best model
model = tf.keras.models.load_model("resnet_crop_x1x2.h5", compile=False)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate MAE per coordinate
mae_x1 = np.mean(np.abs(y_pred[:, 0] - y_val[:, 0]))
mae_x2 = np.mean(np.abs(y_pred[:, 1] - y_val[:, 1]))
mae_overall = np.mean(np.abs(y_pred - y_val))

print(f"🔍 MAE Evaluation:")
print(f"  ▸ MAE (x1): {mae_x1:.2f} pixels")
print(f"  ▸ MAE (x2): {mae_x2:.2f} pixels")
print(f"  ▸ MAE (Overall): {mae_overall:.2f} pixels")


# # Final Evaluation of Array 2
# ## MAE (x1): 3.16 px → Very good accuracy for one coordinate.
# 
# ## MAE (x2): 4.21 px → Slightly higher, but still in a strong range.
# 
# ## Overall MAE: 3.69 px → Well below typical benchmark ranges for object coordinate prediction, but still above your target of <3 px.
# 
# ## ✅ Strengths:
# ## Both coordinates are already close to your <3 px goal — x1 is almost there.
# 
# ## No major bias indicated (assuming mean error ~0).

# In[28]:


import matplotlib.pyplot as plt

# 🔢 Number of samples to show
N = 5  # or 10 or 20
sample_indices = np.random.choice(len(X_val), N, replace=False)

plt.figure(figsize=(12, 4 * N))  # 🖼️ Tall figure to fit rows

for i, idx in enumerate(sample_indices):
    img = X_val[idx].squeeze()
    true_x1, true_x2 = y_val[idx]
    pred_x1, pred_x2 = y_pred[idx]

    ax = plt.subplot(N, 1, i + 1)  # 1 per row
    ax.imshow(img, cmap='gray', aspect='auto')

    # Ground Truth (Green)
    ax.axvline(x=true_x1, color='lime', linestyle='--', linewidth=2)
    ax.axvline(x=true_x2, color='lime', linestyle='--', linewidth=2)

    # Predictions (Red)
    ax.axvline(x=pred_x1, color='red', linestyle='-', linewidth=2)
    ax.axvline(x=pred_x2, color='red', linestyle='-', linewidth=2)

    ax.set_title(f"Sample {idx}  |  True: ({int(true_x1)}, {int(true_x2)})  |  Pred: ({int(pred_x1)}, {int(pred_x2)})",
                 fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[29]:


import pandas as pd
import numpy as np

# Number of samples to show
N = 20
sample_indices = np.random.choice(len(X_val), N, replace=False)

data = []

for i, idx in enumerate(sample_indices, start=1):
    true_x1, true_x2 = y_val[idx]
    pred_x1, pred_x2 = y_pred[idx]
    
    error_x1 = abs(true_x1 - pred_x1)
    error_x2 = abs(true_x2 - pred_x2)
    
    data.append([
        i, int(true_x1), int(true_x2), 
        round(pred_x1, 2), round(pred_x2, 2),
        round(error_x1, 2), round(error_x2, 2)
    ])

# Create DataFrame for better display
df = pd.DataFrame(data, columns=[
    "Sample", "True x1", "True x2", "Pred x1", "Pred x2", "Error x1", "Error x2"
])

print(df.to_string(index=False))


# # ARRAY 3

# In[30]:


pip install tensorflow opencv-python tqdm


# In[31]:


import os
import numpy as np
import pandas as pd

folder_path = r"A:/project/arrays3/entire scan"
labels_df = pd.read_csv(r"A:/project/arrays3/labels3.csv")

file_names = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

print(f"Found {len(file_names)} .npy files")
print("First 15 file names:", file_names[:15])
print("Labels shape:", labels_df.shape)
print("Columns:", labels_df.columns.tolist())

# Try first 15
for idx in range(15):
    try:
        img = np.load(os.path.join(folder_path, file_names[idx]))
        loc = int(labels_df.loc[idx, "location"])
        width = int(labels_df.loc[idx, "Width"])
        print(f"[{idx}] shape={img.shape}, loc={loc}, width={width}")
        cropped = img[loc:loc+width, :]
        print(f"✅ Cropped shape: {cropped.shape}")
    except Exception as e:
        print(f"❌ Error at idx {idx}: {e}")


# In[32]:


# ----------------------------------- #
# Step 0: Imports                     #
# ----------------------------------- #
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2

# ----------------------------------- #
# Step 1: Paths + Labels              #
# ----------------------------------- #
data_dir = r"A:\project\arrays3\entire scan"
labels_csv_path = r"A:\project\arrays3\labels3.csv"

df = pd.read_csv(labels_csv_path)
df = df.sort_values("ID").reset_index(drop=True)

# Convert ground-truth x1/x2 (originally on 1000px width) to the 256px frame, then normalize to [0,1]
y_px_256 = df[["x1", "x2"]].values.astype(np.float32) * (256.0 / 1000.0)
y = y_px_256 / 256.0  # normalized targets for training

# Meta (Width) normalization
meta = df["Width"].values.astype(np.float32)
meta_norm = (meta / meta.max()).reshape(-1, 1)

# ----------------------------------- #
# Step 2: Load & Preprocess Images    #
# ----------------------------------- #
X = []
for idx in df["ID"]:
    arr = np.load(os.path.join(data_dir, f"{idx}.npy"))  # (3600, 1000)
    arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)  # (256, 256)
    # Ensure to [0,255] for preprocess_input; if already 0..1, scale up
    arr = arr.astype(np.float32)
    if arr.max() <= 1.5:
        arr *= 255.0
    arr3 = np.stack([arr, arr, arr], axis=-1)  # (256,256,3)
    X.append(arr3)

X = np.asarray(X, dtype=np.float32)
X = preprocess_input(X)  # ResNet50V2 expects [-1, 1]

print(f"✅ Loaded {len(X)} images. Final image shape: {X.shape}")

# ----------------------------------- #
# Step 3: Train/Val Split             #
# ----------------------------------- #
Xc_train, Xc_val, Xm_train, Xm_val, y_train, y_val = train_test_split(
    X, meta_norm, y, test_size=0.2, random_state=42
)

# ----------------------------------- #
# Step 4: Model (constrained outputs) #
# ----------------------------------- #
image_input = Input(shape=(256, 256, 3), name="image_input")
meta_input  = Input(shape=(1,),            name="width_input")

base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=image_input)
base_model.trainable = False  # start frozen; you can unfreeze later for fine-tuning

# Image branch
x = layers.GlobalAveragePooling2D()(base_model.output)

# Meta branch
m = layers.Dense(16, activation='relu')(meta_input)
m = layers.Dense(8,  activation='relu')(m)

# Fusion head
h = layers.Concatenate()([x, m])
h = layers.Dense(128, activation='relu')(h)
h = layers.Dense(64,  activation='relu')(h)

# Predict center in [0,1] and positive length in (0,1]; then reconstruct x1,x2 with constraints
c = layers.Dense(1, activation='sigmoid', name="center")(h)   # center ∈ [0,1]
l_raw = layers.Dense(1, activation='linear', name="raw_len")(h)
l_pos = layers.Lambda(lambda z: tf.nn.softplus(z), name="length_sp")(l_raw)
l     = layers.Lambda(lambda z: tf.clip_by_value(z, 1e-6, 1.0), name="length_clip")(l_pos)

x1 = layers.Lambda(lambda t: tf.clip_by_value(t[0] - 0.5 * t[1], 0.0, 1.0), name="x1_norm")([c, l])
x2 = layers.Lambda(lambda t: tf.clip_by_value(t[0] + 0.5 * t[1], 0.0, 1.0), name="x2_norm")([c, l])

output = layers.Concatenate(name="x1x2")([x1, x2])

fusion_model = Model(inputs=[image_input, meta_input], outputs=output)

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.Huber(delta=1.0),  # robust to a few tough samples
    metrics=['mae']  # this is on normalized coords; px ≈ mae*256
)

fusion_model.summary()

# ----------------------------------- #
# Step 5: Callbacks                   #
# ----------------------------------- #
class ValPixelMAE(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict([Xc_val, Xm_val], verbose=0)
        val_mae_px = np.mean(np.abs(preds*256.0 - y_val*256.0))
        print(f"\n🧪 val_pixel_MAE: {val_mae_px:.2f} px")

early_stop = EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_mae', patience=10, factor=0.5, min_lr=1e-6, verbose=1)
ckpt       = ModelCheckpoint("model/fusion_best.keras", monitor='val_mae', save_best_only=True)

# ----------------------------------- #
# Step 6: Train                       #
# ----------------------------------- #
history = fusion_model.fit(
    [Xc_train, Xm_train], y_train,
    validation_data=([Xc_val, Xm_val], y_val),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop, reduce_lr, ValPixelMAE(), ckpt],
    verbose=1
)

# (Optional) Fine-tune: unfreeze top layers after convergence for a few more epochs
# for layer in base_model.layers[-50:]:
#     layer.trainable = True
# fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
#                      loss=tf.keras.losses.Huber(delta=1.0),
#                      metrics=['mae'])
# fusion_model.fit(
#     [Xc_train, Xm_train], y_train,
#     validation_data=([Xc_val, Xm_val], y_val),
#     epochs=30, batch_size=16,
#     callbacks=[early_stop, reduce_lr, ValPixelMAE()], verbose=1
# )

# ----------------------------------- #
# Step 7: Save + Evaluate in pixels   #
# ----------------------------------- #
os.makedirs("model", exist_ok=True)
fusion_model.save("model/fusion_final.keras")

preds = fusion_model.predict([X, meta_norm], verbose=0)
preds_px = preds * 256.0
y_px     = y * 256.0

mae_x1 = np.mean(np.abs(preds_px[:, 0] - y_px[:, 0]))
mae_x2 = np.mean(np.abs(preds_px[:, 1] - y_px[:, 1]))
overall_mae = (mae_x1 + mae_x2) / 2.0

print(f"\n📏 MAE x1: {mae_x1:.2f} px")
print(f"📏 MAE x2: {mae_x2:.2f} px")
print(f"📊 Overall MAE: {overall_mae:.2f} px")

np.save("model/preds_array3_px.npy", preds_px)
np.save("model/y_true_array3_px.npy", y_px)
print("✅ Model and predictions saved.")


# # Evaluation for Array 3
# ## MAE (x1): 1.99 px → Right on target.
# 
# ## MAE (x2): 1.79 px → Even better.
# 
# ## Overall MAE: 1.89 px → Below your goal, meaning predictions are extremely close to the true coordinates.
# 
# ##  Why this is impressive:
# ## At this error level, most predictions are within ±2 pixels of ground truth.
# 
# ## In real-world scale, for 256 px resized images, 1.89 px ≈ 0.74% relative error.
# 
# ## No strong bias toward over/under-prediction if the mean error is near zero.
# 
# 

# In[33]:


import numpy as np

# -----------------------------------
# Load saved arrays from training
# -----------------------------------
y_px     = np.load("model/y_true_array3_px.npy")   # shape: (N, 2)
preds_px = np.load("model/preds_array3_px.npy")    # shape: (N, 2)

# -----------------------------------
# Statistics function
# -----------------------------------
def print_stats(labels, preds, title_prefix=""):
    y = np.asarray(labels).reshape(-1).astype(float)
    p = np.asarray(preds).reshape(-1).astype(float)
    if y.shape != p.shape:
        raise ValueError(f"labels and preds must have same shape, got {y.shape} vs {p.shape}")

    err = y - p
    abs_err = np.abs(err)
    sq_err  = err ** 2

    print(f"{title_prefix}Length of labels: {len(y)}")
    print(f"{title_prefix}Length of predictions: {len(p)}")
    print(f"{title_prefix}Mean absolute error (MAE): {np.mean(abs_err):.4f}")
    print(f"{title_prefix}Mean squared error (MSE): {np.mean(sq_err):.4f}")
    print(f"{title_prefix}Average of errors (label - prediction): {np.mean(err):.4f}")
    print(f"{title_prefix}Standard deviation of errors: {np.std(err, ddof=1):.4f}")
    print(f"{title_prefix}Median of errors: {np.median(err):.4f}")
    print(f"{title_prefix}Max of labels: {np.max(y):.4f}")
    print(f"{title_prefix}Minimum of labels: {np.min(y):.4f}")
    print(f"{title_prefix}Average of labels: {np.mean(y):.4f}")
    print(f"{title_prefix}Standard deviation of labels: {np.std(y, ddof=1):.4f}")
    print()

# -----------------------------------
# Run stats for each case
# -----------------------------------
print_stats(y_px[:, 0], preds_px[:, 0], "x1 | ")
print_stats(y_px[:, 1], preds_px[:, 1], "x2 | ")
print_stats(y_px.reshape(-1), preds_px.reshape(-1), "Combined x1+x2 | ")


# In[ ]:





# In[34]:


import numpy as np
import matplotlib.pyplot as plt

# If y_px / preds_px aren't in memory, uncomment:
# y_px     = np.load("model/y_true_array3_px.npy")   # shape (N,2)
# preds_px = np.load("model/preds_array3_px.npy")    # shape (N,2)

def make_plots(y_col, p_col, name):
    y = y_px[:, y_col].astype(float)
    p = preds_px[:, y_col].astype(float)
    err = y - p
    mae = np.mean(np.abs(err))

    # ---- Figure: scatter + histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter: True vs Pred
    axes[0].scatter(y, p, s=12, alpha=0.7)
    lo = min(np.min(y), np.min(p))
    hi = max(np.max(y), np.max(p))
    axes[0].plot([lo, hi], [lo, hi], linewidth=1)  # y = x reference
    axes[0].set_title(f"{name} — True vs Predicted (MAE={mae:.2f}px)")
    axes[0].set_xlabel("True (pixels)")
    axes[0].set_ylabel("Predicted (pixels)")
    axes[0].grid(True, alpha=0.3)

    # Histogram: Errors (label - prediction)
    axes[1].hist(err, bins=40)
    axes[1].set_title(f"{name} — Error Histogram (label - prediction)")
    axes[1].set_xlabel("Error (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Save figures
    fig.savefig(f"model/{name.replace(' ', '_').lower()}_scatter_hist.png", dpi=150)

# Make both x1 and x2 plots
make_plots(0, preds_px, "x1")
make_plots(1, preds_px, "x2")


# In[ ]:





# In[ ]:





# In[35]:


import numpy as np
import pandas as pd

# ---------------- Load Data ----------------
y_true = np.load("model/y_true_array3_px.npy")   # shape: (N, 2)
y_pred = np.load("model/preds_array3_px.npy")    # shape: (N, 2)

# ---------------- Calculate Differences ----------------
diff_x1 = y_true[:, 0] - y_pred[:, 0]
diff_x2 = y_true[:, 1] - y_pred[:, 1]

# ---------------- Select 20 Samples ----------------
sample_idx = np.random.choice(len(y_true), 20, replace=False)

# Create DataFrame
df_samples = pd.DataFrame({
    "True x1": y_true[sample_idx, 0],
    "True x2": y_true[sample_idx, 1],
    "Pred x1": y_pred[sample_idx, 0],
    "Pred x2": y_pred[sample_idx, 1],
    "Diff x1": diff_x1[sample_idx],
    "Diff x2": diff_x2[sample_idx]
})

# Round values for readability
df_samples = df_samples.round(2)

# ---------------- Display ----------------
print("\n📊 Sample Predictions for Array3 (First 20 Samples)\n")
print(df_samples.to_string(index=False))


# In[ ]:





# In[ ]:





# In[ ]:




