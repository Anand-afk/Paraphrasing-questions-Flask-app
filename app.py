from flask import Flask, redirect, request, url_for,render_template
import pickle
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

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

@app.route('/')
def home():
    return render_template('question.html')

@app.route('/question', methods = ['POST'])
def question():
    if request.method == 'POST':
        question = request.form['question']
        text =  "paraphrase: " + question + " </s>"

        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
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

        final_outputs =[]
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != question.lower() and sent not in final_outputs:
                final_outputs.append(sent)

        return render_template('question.html', output='Paraphrased Questions: ${}'.format(final_outputs))

    else:
        return 'hello'

if __name__ == "__main__":
    app.run(debug=True)