import pandas as pd
import os

for filename in os.listdir("Questions"):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join("Questions/","%s")%filename)
        df['Response']=""
        for i in range(1,len(df)):
            print(df[i])
        