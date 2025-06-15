from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Khởi Chrome với extension
service = Service(executable_path="chromedriver")  # chromedriver đã có trong PATH
options = Options()
options.add_argument(r"--load-extension=D:\ADMIN\DTU 3 in 1")

driver = webdriver.Chrome(service=service, options=options)
driver.get("https://mydtu.duytan.edu.vn/Signin.aspx")
time.sleep(5)
driver.save_screenshot("dtulogin.png")
driver.quit()
