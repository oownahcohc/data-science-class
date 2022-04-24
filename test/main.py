import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())
CoffeeBean_URL = "https://www.coffeebeankorea.com/store/store.asp"

driver.get(CoffeeBean_URL)
time.sleep(1)
