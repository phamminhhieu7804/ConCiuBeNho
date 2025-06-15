# ConCiuBeNho

git clone https://github.com/phamminhhieu7804/ConCiuBeNho.git
cd ConCiuBeNho
conda env create -f environment.yml
conda activate captcha

pip install gdown
gdown https://drive.google.com/uc?id=1MI8IKAZxJ-k6945JxiYxBQgo_FSSjQLj -O real_data.zip
unzip real_data.zip -d data/real

pip install gdown
gdown https://drive.google.com/uc?id=1HAMsL3TPmIiETE4DIvxK-EuHTPUNkK2v -O captcha_weights.zip
unzip captcha_weights.zip

## Các lệnh chính

1. **Thu thập ảnh thật**  
   `python captcha_cnn_solver.py collect --n 2000`

2. **Gán nhãn thủ công**  
   `python label_captcha.py`

3. **Chia dataset**  
   `python split_real.py`

4. **Train synthetic**  
   `python captcha_cnn_solver.py train --epochs 30`

5. **Fine-tune trên ảnh thật**  
    `python captcha_cnn_solver.py train_real \
--real_dir data/real/train \
--val_dir  data/real/val \
--epochs   5`

6. **Đánh giá test set thật**  
   `python captcha_cnn_solver.py eval_real --test_dir data/real/test`

7. **Test end-to-end**  
   `python captcha_cnn_solver.py test --n 100 --user YOUR_USER --pass YOUR_PASS`
