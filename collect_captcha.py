from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time, os

driver = webdriver.Chrome(options=Options())
os.makedirs('data/real/raw', exist_ok=True)
for i in range(2000):
    driver.get("https://mydtu.duytan.edu.vn/Signin.aspx")
    time.sleep(0.5)
    elem = driver.find_element('xpath', '//img[contains(@src,"Captcha")]')
    elem.screenshot(f"data/real/raw/{i:04d}.png")
driver.quit()
