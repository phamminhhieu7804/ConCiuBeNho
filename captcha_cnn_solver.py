import os
import time
import string
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.mixed_precision import set_global_policy
from captcha.image import ImageCaptcha
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Default Configuration
WIDTH, HEIGHT, N_LEN = 170, 80, 4
CHARS = string.digits + string.ascii_uppercase
N_CLASS = len(CHARS)
DEFAULT_BATCH_SIZE = 64
DEFAULT_STEPS = 1000
DEFAULT_EPOCHS = 30

# Paths
WEIGHTS_PATH = 'captcha_cnn.weights.h5'
REAL_RAW_DIR = 'data/real/raw'
REAL_LABELED_DIR = 'data/real/labeled'

# Set mixed precision policy
if tf.config.list_physical_devices('GPU'):
    set_global_policy('mixed_float16')
    print('Mixed precision policy: mixed_float16 enabled')
else:
    print('Mixed precision policy: GPU not found, using default float32')

# Optimizer factory
def get_optimizer(lr=1e-4):
    lr_schedule = ExponentialDecay(initial_learning_rate=lr,
                                   decay_steps=1000,
                                   decay_rate=0.5,
                                   staircase=True)
    return Adam(learning_rate=lr_schedule, clipnorm=1.0)

# Model builder
def build_model():
    inp = layers.Input(shape=(HEIGHT, WIDTH, 3))
    x = inp
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    outputs = [layers.Dense(N_CLASS, activation='softmax')(x) for _ in range(N_LEN)]
    return models.Model(inputs=inp, outputs=outputs)

# Synthetic data pipeline
def make_dataset(batch_size):
    def gen():
        generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
        while True:
            text = ''.join(random.choice(CHARS) for _ in range(N_LEN))
            img = np.array(generator.generate_image(text)) / 255.0
            labels = [tf.one_hot(CHARS.index(ch), N_CLASS) for ch in text]
            yield img, tuple(labels)

    output_signature = (
        tf.TensorSpec((HEIGHT, WIDTH, 3), tf.float32),
        tuple([tf.TensorSpec((N_CLASS,), tf.float32)] * N_LEN)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0'))
    return ds

# Real data pipeline
def make_real_dataset(dir_path, batch_size, shuffle=True):
    files = tf.data.Dataset.list_files(os.path.join(dir_path, '*.png'), shuffle=shuffle)
    def parse_fn(fp):
        img = tf.io.read_file(fp)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [HEIGHT, WIDTH]) / 255.0
        fname = tf.strings.split(fp, os.sep)[-1]
        label_str = tf.strings.regex_replace(fname, r'\.png$', '')
        chars = tf.strings.unicode_split(label_str, 'UTF-8')
        labels = tf.stack([tf.one_hot(CHARS.index(c.numpy().decode()), N_CLASS) for c in chars])
        labels = tuple(labels[i] for i in range(N_LEN))
        return img, labels
    ds = files.map(lambda fp: tf.py_function(parse_fn, [fp], [tf.float32]+[tf.float32]*N_LEN), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda img, *lbl: (img, tuple(lbl)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Collect utility
def collect_captcha(n=2000):
    os.makedirs(REAL_RAW_DIR, exist_ok=True)
    driver = webdriver.Chrome(options=Options())
    for i in range(n):
        driver.get('https://mydtu.duytan.edu.vn/Signin.aspx')
        time.sleep(0.5)
        elem = driver.find_element('xpath', '//img[contains(@src,"Captcha")]')
        elem.screenshot(f'{REAL_RAW_DIR}/{i:04d}.png')
    driver.quit()
    print(f'Collected {n} images to {REAL_RAW_DIR}')

# Train on synthetic
def train_model(batch_size, steps, epochs):
    model = build_model()
    # Resume if weights exist
    if os.path.exists(WEIGHTS_PATH):
        print('Loading model weights…')
        model.load_weights(WEIGHTS_PATH)
    else:
        print('Training new model from scratch…')
    optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=['categorical_crossentropy']*N_LEN,
                  metrics=['accuracy']*N_LEN)
    ds = make_dataset(batch_size)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS_PATH, save_weights_only=True, save_best_only=True, monitor='loss')
    ]
    model.fit(ds, steps_per_epoch=steps, epochs=epochs, callbacks=callbacks, verbose=1)
    print('Synthetic training complete')

# Infer single CAPTCHA
def infer_captcha():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    driver = webdriver.Chrome(options=Options())
    driver.get('https://mydtu.duytan.edu.vn/Signin.aspx')
    time.sleep(3)
    elem = driver.find_element('xpath', '//img[contains(@src,"Captcha")]')
    elem.screenshot('real_captcha.png')
    driver.quit()
    img = cv2.imread('real_captcha.png')
    img = cv2.resize(img, (WIDTH, HEIGHT)) / 255.0
    preds = model.predict(np.expand_dims(img, 0))
    text = ''.join(CHARS[np.argmax(p)] for p in preds)
    print('Predicted CAPTCHA:', text)

# Test synthetic model on real site
def test_real_captcha(n, username, password):
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    opts = Options()
    driver = webdriver.Chrome(options=opts)
    success = 0
    for _ in range(n):
        driver.get('https://mydtu.duytan.edu.vn/Signin.aspx')
        time.sleep(1)
        elem = driver.find_element('xpath', '//img[contains(@src,"Captcha")]')
        elem.screenshot('tmp.png')
        img = cv2.imread('tmp.png')
        img = cv2.resize(img, (WIDTH, HEIGHT)) / 255.0
        preds = model.predict(np.expand_dims(img, 0))
        text = ''.join(CHARS[np.argmax(p)] for p in preds)
        driver.find_element('id','txtUser').send_keys(username)
        driver.find_element('id','txtPass').send_keys(password)
        driver.find_element('id','txtCaptcha').send_keys(text)
        driver.find_element('id','btnLogin').click()
        time.sleep(0.5)
        if 'Signin.aspx' not in driver.current_url:
            success += 1
    driver.quit()
    print(f'Tested {n}, success {success}/{n} = {success/n*100:.2f}%')

# Fine-tune on real data
def train_real(real_dir, val_dir, epochs):
    model = build_model()
    if os.path.exists(WEIGHTS_PATH):
        print('Loading base weights for fine-tune…')
        model.load_weights(WEIGHTS_PATH)
    else:
        print('No base weights found, training from scratch…')
    optimizer = get_optimizer(5e-5)
    model.compile(optimizer=optimizer,
                  loss=['categorical_crossentropy']*N_LEN,
                  metrics=['accuracy']*N_LEN)
    train_ds = make_real_dataset(real_dir, batch_size=32)
    val_ds   = make_real_dataset(val_dir,  batch_size=32, shuffle=False)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS_PATH, save_weights_only=True, save_best_only=True, monitor='val_loss')
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    print('Fine-tune complete')

# Evaluate on real test set
def eval_real(test_dir):
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    ds = make_real_dataset(test_dir, batch_size=32, shuffle=False)
    loss, *accs = model.evaluate(ds, verbose=1)
    print('Per-char validation accuracy:', accs[-1])
    correct = total = 0
    for imgs, lbls in ds:
        preds = model.predict(imgs)
        for i in range(imgs.shape[0]):
            true = ''.join(CHARS[np.argmax(lbls[j][i])] for j in range(N_LEN))
            pred = ''.join(CHARS[np.argmax(preds[j][i])] for j in range(N_LEN))
            if true == pred: correct += 1
            total += 1
    print(f'Full-string accuracy: {correct/total:.4f}')

# CLI interface
def main():
    parser = argparse.ArgumentParser(description='Captcha CNN Tool')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('collect', help='Collect real CAPTCHA images').add_argument('--n', type=int, default=2000)
    tr = sub.add_parser('train', help='Train on synthetic data')
    tr.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    tr.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    tr.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    tr_real = sub.add_parser('train_real', help='Fine-tune on real data')
    tr_real.add_argument('--real_dir', type=str, required=True)
    tr_real.add_argument('--val_dir',  type=str, required=True)
    tr_real.add_argument('--epochs',   type=int, default=5)
    sub.add_parser('infer', help='Infer a single CAPTCHA')
    testp = sub.add_parser('test', help='Test synthetic model on real site')
    testp.add_argument('--n', type=int, default=100)
    testp.add_argument('--user',   type=str, required=True)
    testp.add_argument('--pass', dest='passwd', type=str, required=True)
    ev = sub.add_parser('eval_real', help='Evaluate on real test dataset')
    ev.add_argument('--test_dir', type=str, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        collect_captcha(args.n)
    elif args.command == 'train':
        train_model(args.batch_size, args.steps, args.epochs)
    elif args.command == 'train_real':
        train_real(args.real_dir, args.val_dir, args.epochs)
    elif args.command == 'infer':
        infer_captcha()
    elif args.command == 'test':
        test_real_captcha(args.n, args.user, args.passwd)
    elif args.command == 'eval_real':
        eval_real(args.test_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
