import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import os
import random
from google.colab import drive

# --- 1. เชื่อมต่อ Google Drive ---
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- 2. ตั้งค่าพาธ (ตรวจสอบให้ตรงกับใน Drive ของคุณ) ---
BASE_SAVE_PATH = '/content/drive/MyDrive/Colab_Notebooks/Contest_food5/'
MODEL_PATH = os.path.join(BASE_SAVE_PATH, 'Final_BestFoodRanker.keras')
IMG_DIR = os.path.join(BASE_SAVE_PATH, 'Test Set Samples/Test Images/') # ไฟล์รูปภาพ
CSV_INPUT = os.path.join(BASE_SAVE_PATH, 'Test Set Samples/test.csv') # ชื่อไฟล์โจทย์คำถาม
CSV_OUTPUT = os.path.join(BASE_SAVE_PATH, 'test.csv')

IM_SIZE = (224, 224)

# --- 3. โหลดโมเดล ---
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"โหลดโมเดลสำเร็จ!")
else:
    print(f"ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")

# --- 4. ฟังก์ชันเตรียมภาพ ---
def prepare_img(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IM_SIZE, interpolation=cv2.INTER_AREA)
    return np.expand_dims(img / 255.0, axis=0).astype(np.float32)

# --- 5. เริ่มอ่าน CSV และทำนายผล ---
df = pd.read_csv(CSV_INPUT)
final_answers = []

print(f"กำลังเริ่มประมวลผล {len(df)} แถว...")

for index, row in df.iterrows():
    # ดึงชื่อรูปจากคอลัมน์ Image 1 และ Image 2
    img1_name = str(row['Image 1']).strip()
    img2_name = str(row['Image 2']).strip()

    img1_path = os.path.join(IMG_DIR, img1_name)
    img2_path = os.path.join(IMG_DIR, img2_name)

    i1 = prepare_img(img1_path)
    i2 = prepare_img(img2_path)

    if i1 is not None and i2 is not None:
        # ใช้โมเดลทำนาย
        prob = model.predict([i1, i2], verbose=0)[0][0]
        winner = 2 if prob > 0.5 else 1
    else:
        # กรณีถ้าหาไฟล์ไม่เจอ ให้สุ่ม 1 หรือ 2 (50/50)
        winner = random.choice([1, 2])
        print(f" แถวที่ {index}: ไม่พบไฟล์ -> สุ่มเลือก {winner}")

    final_answers.append(winner)

# --- 6. บันทึกผล ---
df['Winner'] = final_answers
df.to_csv(CSV_OUTPUT, index=False)

print("-" * 30)
print(f" เสร็จเรียบร้อย! บันทึกไฟล์ที่: {CSV_OUTPUT}")
display(df.head())
