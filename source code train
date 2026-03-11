import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Subtract, Activation, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from google.colab import drive

# --- 1. การตั้งค่า Path ---
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

BASE_PATH = '/content/drive/MyDrive/Colab_Notebooks/Contest_food5/'
CSV_1 = os.path.join(BASE_PATH, 'data_from_questionaire.csv')
CSV_2 = os.path.join(BASE_PATH, 'data_from_intragram.csv')
IMG_DIR_1 = os.path.join(BASE_PATH, 'Questionair Images')
IMG_DIR_2 = os.path.join(BASE_PATH, 'Intragram_Image')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_EPOCH = 100
# พาธสำหรับโหลดและเซฟโมเดล
checkpoint_path = os.path.join(BASE_PATH, 'Final_BestFoodRanker.keras')

# --- 2. ฟังก์ชันเตรียมข้อมูล (Matching ID) ---
insta_files_map = {}
if os.path.exists(IMG_DIR_2):
    for root, dirs, files in os.walk(IMG_DIR_2):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                prefix_id = f.split('_')[0]
                insta_files_map[prefix_id] = os.path.join(root, f)

def get_insta_path(img_id):
    if pd.isna(img_id): return None
    target_id = str(img_id).split('_')[0]
    return insta_files_map.get(target_id)

# --- 3. รวบรวม Dataset ---
df1 = pd.read_csv(CSV_1)
df1['Image 1'] = df1['Image 1'].apply(lambda x: os.path.join(IMG_DIR_1, str(x) if '.' in str(x) else f"{x}.jpg"))
df1['Image 2'] = df1['Image 2'].apply(lambda x: os.path.join(IMG_DIR_1, str(x) if '.' in str(x) else f"{x}.jpg"))

df2 = pd.read_csv(CSV_2)
df2['Image 1'] = df2['Image 1'].apply(get_insta_path)
df2['Image 2'] = df2['Image 2'].apply(get_insta_path)

df_total = pd.concat([df1, df2], ignore_index=True).dropna(subset=['Image 1', 'Image 2'])
df_train, df_val = train_test_split(df_total, test_size=0.15, random_state=42, shuffle=True)

# --- 4. ฟังก์ชันสร้างโมเดล (กรณีจะเริ่มสร้างmodelใหม่ตั้งแต่ต้น) ---
def build_siamese_model():
    input_node = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_node)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    score = Dense(1, name="Aesthetic_Score")(x)
    encoder = Model(inputs=input_node, outputs=score, name="Encoder")

    img_a, img_b = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)), Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    s1, s2 = encoder(img_a), encoder(img_b)
    diff = Subtract()([s2, s1])
    output = Activation('sigmoid')(diff)
    model = Model(inputs=[img_a, img_b], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 5. Data Generator ---
def pair_generator(df, batch_size, is_training=True):
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True if is_training else False,
        rotation_range=20 if is_training else 0,
        brightness_range=[0.95, 1.05] if is_training else None,
        fill_mode='reflect'
    )
    while True:
        working_df = df.sample(frac=1).reset_index(drop=True) if is_training else df
        for offset in range(0, len(working_df), batch_size):
            batch_df = working_df.iloc[offset : offset + batch_size]
            img1_list, img2_list, y_list = [], [], []
            for _, row in batch_df.iterrows():
                try:
                    img1_raw = img_to_array(load_img(row['Image 1'], target_size=IMAGE_SIZE))
                    img2_raw = img_to_array(load_img(row['Image 2'], target_size=IMAGE_SIZE))
                    winner = int(row['Winner'])
                    if is_training:
                        img1 = datagen.random_transform(img1_raw) / 255.0
                        img2 = datagen.random_transform(img2_raw) / 255.0
                        if np.random.random() > 0.5:
                            img1_list.append(img2); img2_list.append(img1)
                            y_list.append(1.0 if winner == 1 else 0.0)
                        else:
                            img1_list.append(img1); img2_list.append(img2)
                            y_list.append(1.0 if winner == 2 else 0.0)
                    else:
                        img1_list.append(img1_raw / 255.0); img2_list.append(img2_raw / 255.0)
                        y_list.append(1.0 if winner == 2 else 0.0)
                except: continue
            if img1_list:
                yield ((np.array(img1_list), np.array(img2_list)), np.array(y_list))

# --- 6. Dataset Setup ---
output_sig = ((tf.TensorSpec(shape=(None,224,224,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,224,224,3), dtype=tf.float32)),
               tf.TensorSpec(shape=(None,), dtype=tf.float32))

train_ds = tf.data.Dataset.from_generator(lambda: pair_generator(df_train, BATCH_SIZE, True), output_signature=output_sig).prefetch(2)
val_ds = tf.data.Dataset.from_generator(lambda: pair_generator(df_val, BATCH_SIZE, False), output_signature=output_sig).prefetch(2)

# --- 7. โหลดโมเดลเดิมมาเทรนต่อ ---
if os.path.exists(checkpoint_path):
    print(f"พบโมเดลเดิมที่: {checkpoint_path} กำลังโหลดมาเทรนต่อ...")
    siamese_model = load_model(checkpoint_path)
    # ลด Learning Rate ลงเหลือ 1e-5 เพื่อ Fine-tune อย่างระมัดระวัง
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                          loss='binary_crossentropy', metrics=['accuracy'])
else:
    print(" ไม่พบโมเดลเดิม กำลังเริ่มสร้างโมเดลใหม่...")
    siamese_model = build_siamese_model()

# --- 8. ตั้งค่า Callbacks และเริ่มเทรน ---
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

print(f"\n เริ่มทำการเทรนโมเดล...")
siamese_model.fit(
    train_ds,
    steps_per_epoch=len(df_train)//BATCH_SIZE,
    validation_data=val_ds,
    validation_steps=len(df_val)//BATCH_SIZE,
    epochs=MAX_EPOCH,
    callbacks=callbacks
)

print(f"\n การเทรนเสร็จสิ้น! โมเดลถูกบันทึกไว้ที่: {checkpoint_path}")
