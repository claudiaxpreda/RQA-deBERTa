import pandas as pd
from datasets import load_dataset

DATASET_FT = 'readerbench/fairytale_qgen_contest'

def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  return entry

def qg_prompt_learn(entry):
    return f"Generate a question based on the context and the answer.\nContext: {entry['context']}\nAnswer: {entry['answer']}\n### Response: {entry['question']}"

def qa_prompt_learn(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry['question']}\n### Response: {entry['answer']}"

def ag_prompt_learn(entry): 
    return f"Select an answer from the context that can be used to generate a question.\nContext: {entry['context']}\n### Response: {entry['answer']}"

def qa_prompt_learn_processed(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry['question']}\n### Response: {entry['sequence']}"

def gq_prompt_generate(entry):
    return f"Generate a question based on the context and the answer.\nContext: {entry['context']}\nAnswer: {entry['sequence']}\n### Response:"

def  qa_prompt_generate(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry['question']}\n### Response:"

def ag_prompt_generate(entry):
    return f"Select an answer from the context that can be used to generate a question.\nContext: {entry['context']}\n### Response:"

def get_input_qg(split, token): 
    fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': qg_prompt_learn(e)})

    return fairytale_dataset

def get_input_qa(split, token): 
    fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': qa_prompt_learn(e)})

    return fairytale_dataset

def get_input_ag(split, token): 
    fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': ag_prompt_learn(e)})

    return fairytale_dataset

def main():
    test = get_input_ag('validation', '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')
    print(test['input_prompt'][1])


if __name__ == "__main__":
    main()