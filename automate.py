import pandas as pd
from selenium import webdriver
import re
import os

driver = webdriver.Chrome("C:\\Users\\arane\\Anand Projects\\Paraphrasing-questions-Flask-app\\chromedriver.exe")
driver.get("https://kindsaola-cac17e.bots.teneo.ai/general_enquiry_536803kc3081vrdatgv2gftaxs/")

for filename in os.listdir("Questions"):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join("Questions/","%s")%filename)
        df['Response']=""
        for i in range(len(df['Questions'])):
            driver.find_element_by_id("inputTextBox").send_keys(df['Questions'][i])
            driver.find_element_by_id("inputButton").click()
            driver.implicitly_wait(10)
            df['Response'][i] = re.sub("^Text ","",(driver.find_element_by_class_name("outputText")).text)
            driver.find_element_by_id("endSessionButton").click()
            driver.implicitly_wait(10)
        
        df.to_csv(os.path.join("Questions/","%s") %filename)
    else:
        continue