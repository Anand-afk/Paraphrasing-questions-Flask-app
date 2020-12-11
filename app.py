from flask import Flask, redirect, request, url_for,render_template
import re,csv
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

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


@app.route('/')
def home():
    return render_template('question.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("Input_csv/",secure_filename(f.filename)))

    df = pd.read_csv(f.filename)
    for i in df['Questions ']:
        final_outputs = []
        if "|" in i:
            x = i.split('|')
            for j in x:
                final_outputs.append(generator(j))

            final_outputs = [item for sublist in final_outputs for item in sublist]
            output = {}
            for c, final_output in enumerate(final_outputs):
                output[c] = final_output
                filename = findmatches(patterns, i)
                with open("%s.csv" % filename, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in output.items():
                        writer.writerow([key, value])
        else:
            final_outputs = generator(i)
            output = {}
            for c, final_output in enumerate(final_outputs):
                output[c] = final_output

            result = pd.DataFrame(list(output.items()),columns = ['No','Questions']) 
            filename = findmatches(patterns, i)
            result.to_csv(os.path.join("Questions/","%s.csv") %filename)
            
    return "testing out"


@app.route('/question', methods = ['POST'])
def question():
    if request.method == 'POST':
        question = request.form['question']
        final_out = generator(question) 
        return render_template('question.html', output=final_out)
    else:
        return 'hello'

if __name__ == "__main__":
    app.run(debug=True)