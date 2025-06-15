# label_captcha.py
import os
import cv2

raw_dir     = 'data/real/raw'
labeled_dir = 'data/real/labeled'
os.makedirs(labeled_dir, exist_ok=True)

files = sorted(os.listdir(raw_dir))
for fname in files:
    path = os.path.join(raw_dir, fname)
    img  = cv2.imread(path)
    cv2.imshow('CAPTCHA (nhấn Esc để thoát)', img)
    # bấm ESC (27) để bỏ qua, hoặc bất kỳ phím nào để gán nhãn
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    label = input(f"Nhập text cho {fname}: ").strip().upper()
    if label:
        new_name = f"{label}.png"
        os.rename(path, os.path.join(labeled_dir, new_name))
    else:
        print("Bỏ qua:", fname)
