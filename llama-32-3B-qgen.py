import torch
import sys
import time 

import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from prepare_dataset_llama import gq_prompt_generate


HUGGING_FACE_TOKEN='hf_{}'

MAX_LEN_SEQ = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  entry = entry.strip()
  return entry

def read_slice(source_file): 
    slice_df = pd.read_csv(source_file, keep_default_na=False)
    return slice_df

def write_slice(df, destination_file): 
  df.to_csv(destination_file, index=False)


def create_dict_obj(entry, decode):
  return {
      'context': entry['context'], 'sequence': entry['sequence'],
      'question': decode, 
      }

def qgen_split(entry, tokenizer, model):
  prompt = gq_prompt_generate(entry)
  model_inputs =  tokenizer(prompt, return_tensors="pt", padding='longest', truncation=True, max_length=2048*2).to(device)
  
  generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=300,
    pad_token_id=128002,
  )

  to_decode = generated_ids[0][len(model_inputs.input_ids[0]):]
  decoded = tokenizer.decode(to_decode, skip_special_tokens=True)
  decoded = normalize(decoded)
  return decoded


def apply_qgen(model_name, source_dataset):

  tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, padding_side="left")
  model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, device_map="auto", torch_dtype=torch.bfloat16)

  source_dataset['question'] = source_dataset.apply(
    lambda e : qgen_split(e, tokenizer, model), axis=1) 
 
  return source_dataset


def main(start, end, slice):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))
  filename = 'fairytale_dataset/original/input_{}.csv'.format(slice)
  model_name='claudiapreda/llama32-3b_qgen_ft'
  destination = 'fairytale_dataset/QG/' + '{}/qg_{}_'.format(slice, slice) + str(end) +'.csv'

  source_dataset = read_slice(filename)

  if end == 'end':
    print('hello')
    source_dataset = source_dataset.iloc[int(start):]
  else: 
    source_dataset = source_dataset.iloc[int(start):int(end)]

  dataset = apply_qgen(model_name, source_dataset)

  write_slice(dataset, destination)

  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))

  return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])