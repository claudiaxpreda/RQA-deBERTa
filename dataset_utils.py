import os
import sys 
import pandas as pd
import ast

from transformers import BertTokenizerFast
import prepare_input as pi 

MAX_LEN_SEQ = 512

# def read_slice(source_file): 
#     slice_df = pd.read_csv(source_file, keep_default_na=False, 
#                            converters={"indexes": ast.literal_eval})
    
#     return slice_df


def read_slice(source_file): 
  slice_df = pd.read_csv(source_file, keep_default_na=False, 
                           converters={"indexes": ast.literal_eval, 
                                       "token_indexes": ast.literal_eval})

  return slice_df

def write_slice(path, slice, df): 
  destination_file = path + 'input_' + slice + '.csv'
  df.to_csv(destination_file, index=False)


def concat_files(list_of_files, path, write_output=False, target_name=''):
  df_list = []

  for file in list_of_files: 
    df = pd.read_csv(path + file, index_col=None, header=0)
    df_list.append(df)
    print(len(df_list))
  
  df_concat = pd.concat(df_list, ignore_index=True)
  print('Before removing duplicates: {}\n'.format(df_concat.size))
  df_concat.drop_duplicates(inplace=True)
  print('After removing duplicates: {}\n'.format(df_concat.size))

  if write_output == True: 
    df_concat.to_csv(target_name, index=False)
  
  return df_concat

def concat_dfs(): 
  path = 'fairytale_dataset/QAL/train/'
  list_of_files = [
    "qloss_train_50000.csv", "qloss_train_100000.csv", 
    "qloss_train_150000.csv", "qloss_train_200000.csv", 
    "qloss_train_250000.csv",  "qloss_train_300000.csv"
  ]

  target_name = 'fairytale_dataset/final/qloss_train_all_v1.csv'
  concat_files(list_of_files, path, write_output=True, target_name=target_name)



def get_avg_len_pad():
  test_dataset_path = 'dataset/final/gl_test_all.csv'
  val_dataset_path = 'dataset/final/gl_val_all.csv'
  train_dataset_path = 'dataset/final/ql_train_all.csv'
  tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")  
  max_len_pad = '100'
  
  x_train, y_train, gt = pi.prepare_input(read_slice(train_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 1: Generated data for trainig\n")

  x_val, y_val, gv = pi.prepare_input(read_slice(val_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 2: Generated data for val\n")

  x_test, y_test, gtt= pi.prepare_input(read_slice(test_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 3: Generated data for test\n")

  print("Train: " + str(gt.count().mean()['token_indexes']))
  print("\nVal: " + str(gv.count().mean()['token_indexes']))
  print("\nTest: " + str(gv.count().mean()['token_indexes']))


  print(gt.ngroups)
  print(gv.ngroups)
  print(gtt.ngroups)

if __name__ == "__main__":
  get_avg_len_pad()