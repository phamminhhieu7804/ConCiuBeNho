import os
import string
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from captcha.image import ImageCaptcha
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Configuration
WIDTH, HEIGHT, N_LEN = 170, 80, 4
CHARS = string.digits + string.ascii_uppercase
N_CLASS = len(CHARS)
BATCH_SIZE = 64
STEPS_PER_EPOCH = 1000
EPOCHS = 30

# 1) Data generator using ImageCaptcha
class CaptchaSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, steps):
        self.batch_size = batch_size
        self.steps = steps
        self.generator = ImageCaptcha(width=WIDTH, height=HEIGHT)

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, HEIGHT, WIDTH, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, N_CLASS), dtype=np.uint8) for _ in range(N_LEN)]
        for i in range(self.batch_size):
            text = ''.join(random.choice(CHARS) for _ in range(N_LEN))
            img = np.array(self.generator.generate_image(text)) / 255.0
            X[i] = img
            for j, ch in enumerate(text):
                y[j][i, CHARS.find(ch)] = 1
        return X, y

# 2) Build CNN model with multiple outputs
def build_model():
    inp = layers.Input(shape=(HEIGHT, WIDTH, 3))
    x = inp
    # Block1
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Block2
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Block3
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Flatten
    x = layers.Flatten()(x)
    outputs = []
    for _ in range(N_LEN):
        outputs.append(layers.Dense(N_CLASS, activation='softmax')(x))
    model = models.Model(inputs=inp, outputs=outputs)
    return model

# 3) Training function
def train():
    model = build_model()
    losses = {f'dense_{i+1}_': 'categorical_crossentropy' for i in range(N_LEN)}
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=['categorical_crossentropy']*N_LEN,
        metrics=['accuracy']
    )
    seq = CaptchaSequence(BATCH_SIZE, STEPS_PER_EPOCH)
    model.fit(seq, epochs=EPOCHS)
    model.save('captcha_cnn.h5')
    print('Model saved to captcha_cnn.h5')

# 4) Inference on real CAPTCHA
def infer():
    # Capture with Selenium
    options = Options()
    options.add_argument(r"--load-extension=D:\ADMIN\file DTU 3 in 1 test")
    driver = webdriver.Chrome(options=options)
    driver.get("https://mydtu.duytan.edu.vn/Signin.aspx")
    time.sleep(3)
    elem = driver.find_element('xpath', '//img[contains(@src,"Captcha")]')
    elem.screenshot('real_captcha.png')
    driver.quit()

    # Preprocess
    img = cv2.imread('real_captcha.png')
    img = cv2.resize(img, (WIDTH, HEIGHT)) / 255.0
    X = np.expand_dims(img, axis=0)

    # Load model
    model = models.load_model('captcha_cnn.h5')
    preds = model.predict(X)
    text = ''.join(CHARS[np.argmax(p)] for p in preds)
    print('Predicted CAPTCHA:', text)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    else:
        infer()
