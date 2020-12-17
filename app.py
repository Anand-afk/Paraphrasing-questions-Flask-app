from flask import Flask, redirect, request, url_for,render_template, send_file
import re,csv
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from werkzeug.utils import secure_filename
import os
from selenium import webdriver
import math
from collections import Counter

WORD = re.compile(r"\w+")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

def generator(sentence):
    text = "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs

def findmatches(pattern, phrase):
    for p in patterns:
        match = re.findall(p, phrase)
    return match[0]

patterns = ['[^!.?]+']

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

@app.route('/')
def home():
    return render_template('question.html')

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')

res_file = ""
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        res_file = secure_filename(f.filename)
        f.save(os.path.join("Input_csv/",res_file))
        print(res_file)

    df = pd.read_csv(f.filename)
    for i,expected in zip(df['Questions '],df['Expected Response']):
        final_outputs = []
        if "|" in i:
            x = i.split('|')
            for j in x:
                final_outputs.append(generator(j))

            final_outputs = [item for sublist in final_outputs for item in sublist]
            output = {}
            for c, final_output in enumerate(final_outputs):
                output[c] = final_output
            result = pd.DataFrame(list(output.items()),columns = ['No','Questions'])
            result['Expected Response']=expected
            filename = findmatches(patterns, i)
            result.to_csv(os.path.join("Questions/","%s.csv") %filename)

        else:
            final_outputs = generator(i)
            output = {}
            for c, final_output in enumerate(final_outputs):
                output[c] = (final_output)

            result = pd.DataFrame(list(output.items()),columns = ['No','Questions'])
            result['Expected Response']=expected
            filename = findmatches(patterns, i)
            result.to_csv(os.path.join("Questions/","%s.csv") %filename)

    # return redirect(url_for('downloadFile'))
    return render_template('download.html')

@app.route('/download')
def downloadFile():
    driver = webdriver.Chrome("C:\\Users\\arane\\Anand Projects\\Paraphrasing-questions-Flask-app\\chromedriver.exe")
    driver.get("https://kindsaola-cac17e.bots.teneo.ai/general_enquiry_536803kc3081vrdatgv2gftaxs/")
    
    for mainfile in os.listdir("Input_csv"):
        if mainfile.endswith(".csv"):
            res = pd.read_csv(os.path.join("Input_csv/","%s")%mainfile)
    j=0
    res['Rate'] = ""
    for filename in os.listdir("Questions"):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join("Questions/","%s")%filename)
            # res = pd.read_csv("Input_csv/questions-alex.csv")
            
            df['Returned Response']= ""
            df['Similarity Score']= ""
            df['Result'] = ""
            
            for i in range(len(df['Questions'])):
                driver.find_element_by_id("inputTextBox").send_keys(df['Questions'][i])
                driver.find_element_by_id("inputButton").click()
                driver.implicitly_wait(10)
                df['Returned Response'][i] = re.sub("^Text ","",(driver.find_element_by_class_name("outputText")).text)

                match = re.findall(r"<a[^>]*>([^<]+)<\/a>", df['Returned Response'][i])
                if match:
                    df['Returned Response'][i]= re.sub(r"(<a[^>]*>([^<]+)<\/a>)",match[0],df['Returned Response'][i])
                    df['Returned Response'][i] = re.sub(r"<br>","",df['Returned Response'][i])
                expected_vec=text_to_vector(df['Expected Response'][i])
                returned_vec=text_to_vector(df['Returned Response'][i])
                df['Similarity Score'][i] = (get_cosine(expected_vec, returned_vec))*100
                if float(df['Similarity Score'][i]) > 80.0:
                    df['Result'][i]="Success"
                else:
                    df['Result'][i]="Fail"

                driver.find_element_by_id("endSessionButton").click()
                driver.implicitly_wait(10)

            res['Rate'][j] = "{0} -/- {1}".format(len(df[df['Result']=="Success"]),len(df['Result']))
            j+=1
            df.to_csv(os.path.join("Questions/","%s") %filename)
        else:
            continue
    
    res.to_csv("Output_csv/result.csv")
    driver.close()      
    path = "Output_csv/result.csv"
    return send_file(path, as_attachment=True)

@app.route('/question', methods = ['POST'])
def question():
    if request.method == 'POST':
        question = request.form['question']
        final_out = generator(question) 
        return render_template('question.html', output=final_out)
    else:
        return 'hello'


    #For windows you need to use drive name [ex: F:/Example.pdf]
    

if __name__ == "__main__":
    app.run(debug=True)






