import sys 
import pandas as pd
import ast
import time
import prepare_input as pi 


import tensorflow as tf

import wandb
from wandb.integration.keras import WandbMetricsLogger

from transformers import AutoTokenizer, TFDebertaV2Model, TFBertModel
from tensorflow import keras
from tensorflow.keras import layers
from huggingface_hub import push_to_hub_keras

MAX_LEN_SEQ = 512
MAX_LEN_BERT = 768
HUGGING_TOKEN = "hf_{}"
API_KEY_WB = "{}"
LR = 1e-5


def read_slice(source_file): 
  slice_df = pd.read_csv(source_file, keep_default_na=False, 
                           converters={"indexes": ast.literal_eval, 
                                       "token_indexes": ast.literal_eval})
    
  return slice_df


def create_model(encoder_model_name):

  #encoder = TFDebertaV2Model.from_pretrained("microsoft/deberta-v3-base", output_hidden_states=True)
  #encoder = TFBertModel.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True)
  if encoder_model_name == 'google-bert/bert-base-uncased':
    encoder = TFBertModel.from_pretrained(encoder_model_name, output_hidden_states=True)
  else: 
    encoder =  TFDebertaV2Model.from_pretrained(encoder_model_name, output_hidden_states=True)



  context_ids = layers.Input(shape=(MAX_LEN_SEQ,), dtype=tf.int32)
  attention_mask = layers.Input(shape=(MAX_LEN_SEQ,), dtype=tf.int32)
  questions_ids = layers.Input(shape=(None, MAX_LEN_SEQ), dtype=tf.float32)
  embedding = encoder(
      context_ids, attention_mask=attention_mask
  )[0]

  #embedding: B, 512, 768
  #question_ids: B, MAX_LEN_PAD, 512
  questions_ids_t = tf.transpose(questions_ids, perm=[0, 2, 1])
  #print('Question Mask Shape: {}\n}.format(questions_ids_t.shape))
  
  product_step_a=tf.matmul(questions_ids, embedding)
  #print('E x Q Mask shape: {}\n}.format(product_step_a.shape))
  product_step_b = tf.math.count_nonzero(questions_ids, 2, keepdims=True,  dtype=tf.dtypes.float32)
  #print('Count of 1 Mask shape: {}\n}.format(product_step_b.shape))

  product_step_c = tf.math.divide(product_step_a, tf.maximum(product_step_b, 1))
  #print('E x Q Mask normalized shape: {}\n}.format(product_step_c.shape))

  layer1 = layers.TimeDistributed(layers.Dense(256, name="DenseLayer1", activation="relu"))(product_step_c, mask=tf.squeeze(product_step_b > 0, axis=-1))

  layer2 = layers.Dense(64, name="DneseLayer2")(layer1)

  layer3 = layers.Dense(1, name="DneseLayer3")(layer2)

  output = layer3

  model = keras.Model(
      inputs=[context_ids, attention_mask, questions_ids],
      outputs= [output],
  )

  optimizer = keras.optimizers.Adam(learning_rate=LR)
  model.compile(loss='mean_squared_error', optimizer=optimizer,metrics='mean_squared_error')

  return model


 
def main(answer_selection_model, max_len_pad, epochs, encoder_model_name):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))

  train_dataset_path = 'fairytale_dataset/final/qloss_train_all_v1.csv'
  test_dataset_path = 'fairytale_dataset/QAL/test/qloss_test_15000.csv'

  tokenizer=AutoTokenizer.from_pretrained(encoder_model_name)
  # tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")  

  input_df = read_slice(train_dataset_path)

  x_train, y_train, x_val, y_val = pi.prepare_input_with_split(input_df, tokenizer, int(max_len_pad))
  print("Checkpoint 1: Generated data\n")


  model = create_model(encoder_model_name)


  print("Checkpoint 4: Generated model\n")

  wandb.login(key = API_KEY_WB)

  
  configs_wb = {
      "learning_rate": LR,
      "architecture": encoder_model_name,
      "dataset": "FairytaleQALoss",
      "epochs": epochs,
    }
    
  run = wandb.init(
    project = "answer-selection",
    config = configs_wb, 
    notes = answer_selection_model,
  )

  model.fit(
    x_train,
    y_train,
    epochs= int(epochs),  # For demonstration, 3 epochs are recommended
    verbose=2,
    batch_size=48, 
    validation_data=(x_val, y_val),
    callbacks = [WandbMetricsLogger()]
  )

  test_df= read_slice(test_dataset_path)
  x_test, y_test, _ = pi.prepare_input(test_df, tokenizer, int(max_len_pad))

  results = model.evaluate(x_test, y_test, batch_size=128)
  print("test loss, test metric:", results)

  run.finish()

  push_to_hub_keras(model,
    "claudiapreda/" + answer_selection_model, token = HUGGING_TOKEN,

  )
  tokenizer.push_to_hub('claudiapreda/' + answer_selection_model, token=HUGGING_TOKEN)

  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))
  return 0

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
