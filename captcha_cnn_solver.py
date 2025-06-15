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
# import tensorflow_addons as tfa  # removed due to compatibility issues


# Default Configuration
WIDTH, HEIGHT, N_LEN = 170, 80, 4
CHARS = string.digits + string.ascii_uppercase
N_CLASS = len(CHARS)
DEFAULT_BATCH_SIZE = 64
DEFAULT_STEPS = 1000
DEFAULT_EPOCHS = 30

# Paths  
WEIGHTS_PATH = 'captcha_cnn.weights.h5'  
# directories for real data  
REAL_RAW_DIR    = 'data/real/raw'  
REAL_LABELED_DIR = 'data/real/labeled'


# Set mixed precision
if tf.config.list_physical_devices('GPU'):
    set_global_policy('mixed_float16')
    print('Mixed precision policy: mixed_float16 enabled')
else:
    print('Mixed precision policy: GPU not found, using default float32')

# Optimizer
def get_optimizer():
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.5,
        staircase=True
    )
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
            labels = [tf.one_hot(CHARS.find(ch), N_CLASS) for ch in text]
            yield img, tuple(labels)
    output_signature = (
        tf.TensorSpec((HEIGHT, WIDTH, 3), tf.float32),
        tuple([tf.TensorSpec((N_CLASS,), tf.float32)] * N_LEN)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0'))
    return ds

# Real data pipeline
def make_real_dataset(dir_path, batch_size, shuffle=True):
    # List files
    files = tf.data.Dataset.list_files(os.path.join(dir_path, '*.png'), shuffle=shuffle)
    def parse_fn(fp):
        img = tf.io.read_file(fp)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [HEIGHT, WIDTH]) / 255.0
        # Extract label
        fname = tf.strings.split(fp, os.sep)[-1]
        label_str = tf.strings.regex_replace(fname, r'\.png$', '')
        chars = tf.strings.unicode_split(label_str, 'UTF-8')
        def onehot(ch):
            ch = ch.numpy().decode('utf-8')
            idx = CHARS.index(ch)
            return np.eye(N_CLASS, dtype=np.float32)[idx]
        labels = tf.py_function(lambda x: [onehot(c) for c in x], [chars], Tout=tf.float32)
        labels = tuple(tf.reshape(labels[i], (N_CLASS,)) for i in range(N_LEN))
        return img, labels
    def augment(img, lbl):
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # rotation removed for compatibility
        return img, lbl
    ds = files.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
    ds = files.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Collection utility
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

# Training synthetic
def train_model(batch_size, steps, epochs):
    # ... existing synthetic train code ...
    pass

# Inference single CAPTCHA
def infer_captcha():
    # ... existing infer code ...
    pass

# Testing synthetic on real site
def test_real_captcha(n, username, password):
    # ... existing test code ...
    pass

# Fine-tune on real data
def train_real(real_dir, val_dir, epochs):
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    model.compile(optimizer=Adam(5e-5),
                  loss=['categorical_crossentropy']*N_LEN,
                  metrics=['accuracy']*N_LEN)
    train_ds = make_real_dataset(real_dir, batch_size=32)
    val_ds   = make_real_dataset(val_dir,  batch_size=32, shuffle=False)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS_PATH, save_weights_only=True, save_best_only=True, monitor='val_loss')
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print('Fine-tune complete')

# Evaluate on real test set
def eval_real(test_dir):
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    ds = make_real_dataset(test_dir, batch_size=32, shuffle=False)
    loss, *accs = model.evaluate(ds)
    print('Per-char validation accuracy:', accs[-1])
    # Full-string
    correct = total = 0
    for imgs, lbls in ds:
        preds = model.predict(imgs)
        for i in range(imgs.shape[0]):
            true = ''.join(CHARS[int(tf.argmax(lbls[j][i]))] for j in range(N_LEN))
            pred = ''.join(CHARS[int(np.argmax(preds[j][i]))] for j in range(N_LEN))
            if true == pred: correct += 1
            total += 1
    print(f'Full-string accuracy: {correct/total:.4f}')

# Main CLI
def main():
    parser = argparse.ArgumentParser(description='Captcha CNN Tool')
    sub = parser.add_subparsers(dest='command')

    # collect
    collect = sub.add_parser('collect', help='Collect real CAPTCHA images')
    collect.add_argument('--n', type=int, default=2000)

    # synthetic train
    train = sub.add_parser('train', help='Train on synthetic data')
    train.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    train.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)

    # fine-tune on real
    train_real_cmd = sub.add_parser('train_real', help='Fine-tune on real data')
    train_real_cmd.add_argument('--real_dir', type=str, required=True)
    train_real_cmd.add_argument('--val_dir',  type=str, required=True)
    train_real_cmd.add_argument('--epochs',   type=int, default=5)

    # infer single
    sub.add_parser('infer', help='Infer a single CAPTCHA')

    # test synthetic on real
    test = sub.add_parser('test', help='Test synthetic model on real site')
    test.add_argument('--n', type=int, default=100)
    test.add_argument('--user', type=str, required=True)
    test.add_argument('--pass', dest='passwd', type=str, required=True)

    # evaluate on real test set
    evalp = sub.add_parser('eval_real', help='Evaluate on real test dataset')
    evalp.add_argument('--test_dir', type=str, required=True)

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
