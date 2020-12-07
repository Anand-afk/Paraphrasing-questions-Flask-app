import pandas as pd
from selenium import webdriver

df = pd.read_csv("questions- alex.csv")

ques= df['Questions '][0]


driver = webdriver.Chrome("C:\\Users\\arane\\Anand Projects\\Paraphrasing-questions-Flask-app\\chromedriver.exe")
driver.get("http://17a9d3252a9f.ngrok.io")
driver.find_element_by_name("question").send_keys(ques)
driver.find_element_by_id("submit").click()