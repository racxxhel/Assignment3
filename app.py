#import necessary libraries
import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel
import string
import re

#initialize the Flask application so flask know where to get html and css files
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# These constants make the script easier to configure and read.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = "distilbert-base-uncased"
LORA_MODEL_PATH = "./backend/results_lora_final/" 
IA3_MODEL_PATH = "./backend/results_ia3_final/"

#helper functions
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace."""
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in exclude)
    exclude = set(string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0: return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def load_model(peft_path):
    """Loads a PEFT model for inference."""
    base_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    return peft_model.to(DEVICE).eval()

def get_answer(context, query, model, tokenizer):
    """Runs inference to get a prediction."""
    inputs = tokenizer.encode_plus(query, context, return_tensors='pt', max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
    
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    if answer == tokenizer.cls_token or answer == "":
        return "[No answer found in context]"
    return answer

#load both models
print("Loading models... This may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
lora_model = load_model(LORA_MODEL_PATH)
ia3_model = load_model(IA3_MODEL_PATH)
print("All models loaded successfully.")

#routing
@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    context = ""
    question = ""
    true_answer = ""
    
    if request.method == 'POST':
        context = request.form['context']
        question = request.form['question']
        true_answer = request.form.get('true_answer', '') 

        if context and question:
            lora_prediction = get_answer(context, question, lora_model, tokenizer)
            ia3_prediction = get_answer(context, question, ia3_model, tokenizer)
            
            # Prepare results
            predictions = {
                'lora': {'text': lora_prediction},
                'ia3': {'text': ia3_prediction}
            }
            
            # If a true answer was provided, calculate and add scores to the dictionary
            if true_answer:
                predictions['lora']['em'] = compute_exact_match(lora_prediction, true_answer)
                predictions['lora']['f1'] = compute_f1(lora_prediction, true_answer)
                predictions['ia3']['em'] = compute_exact_match(ia3_prediction, true_answer)
                predictions['ia3']['f1'] = compute_f1(ia3_prediction, true_answer)

    return render_template('index.html', context=context, question=question, true_answer=true_answer, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)