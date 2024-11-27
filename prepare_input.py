
import tensorflow as tf
from sklearn.model_selection import train_test_split 


MAX_LEN_SEQ = 512
MAX_LEN_BERT = 768

def rectangular(n):
    lengths = {len(i) for i in n}
    return len(lengths) == 1

def split_in_chunks(l, n):
  x = [l[i:i + n] for i in range(0, len(l), n)] 
  return x

def get_max_len(nested_list): 
    return len(max(nested_list, key=len))

def get_pad_len(postitions): 
  max_len = -1
  for pos in postitions:
    print('start: {}, end: {}\n'.format(pos[0], pos[1]))
    if pos[1] - pos[0] + 1 > max_len:
      max_len = pos[1] - pos[0] + 1
  
  return max_len

def prepare_input(slice, tokenizer, max_len_pad_set):
  grouped_slice = slice.groupby('context')
  # len(list_contexts) == B 
  #list_contexts = slice['context'].unique().tolist()
  list_contexts = []
  # len(context_masks) == B
  context_masks = []
  labels = []

  max_len_pad =grouped_slice.count().max()['token_indexes']
  max_len_pad = max_len_pad_set if max_len_pad > max_len_pad_set else max_len_pad

  for group_name, df_group in grouped_slice:
    current_labels = df_group['loss'].to_list()
    #print('Meow' + str(len(current_labels)))
    current_labels = split_in_chunks(current_labels, max_len_pad) 
    #print('Meow2' + str(len(current_labels)))

    labels.extend(current_labels)
    #print(current_labels)

    mask_list_entry = []

    #current_tokens = df_group['token_indexes'].to_list()
    # current_tokens = split_in_chunks(current_tokens, max_len_pad)

    for pos in df_group['token_indexes'].to_list():
      mask = [0] * MAX_LEN_SEQ
      if pos[0] != pos[1] and pos[1] > 0 and pos[0] > 0:
        mask[pos[0]:pos[1]] = [1] * (pos[1] - pos[0])
      else: 
        #mask = pos[0] if pos[0] > 0 else pos[1]
        mask[pos[0]] = 1
      mask_list_entry.append(mask)

    mask_list_entry = split_in_chunks(mask_list_entry, max_len_pad)
    list_contexts.extend([group_name for i in range(len(mask_list_entry))])


    if len(mask_list_entry[-1]) < max_len_pad: 
      padding = [[0]*MAX_LEN_SEQ for x in range(max_len_pad - len(mask_list_entry[-1]))]
      mask_list_entry[-1].extend(padding)
  
    # if len(mask_list_entry) < max_len_pad: 
    #   padding = [[0]*MAX_LEN_SEQ for x in range(max_len_pad - len(mask_list_entry))]
    #   mask_list_entry.extend(padding)
    
    context_masks.extend(mask_list_entry)
  
  context_masks_tf = tf.convert_to_tensor(context_masks)

  for entry in labels:  
    current_len = len(entry)
    if current_len < max_len_pad:
      entry.extend([0 for x in range(max_len_pad - current_len)])


  labels_tf =  tf.convert_to_tensor(labels)
  labels_tf = tf.expand_dims(labels_tf, -1)

  #print('Labesl shape: {}\n}.format(labels_tf.shape))

    #Size (B, 512)
  inputs=tokenizer(list_contexts, padding="max_length", truncation=True, max_length=512, return_tensors='tf')

  context_ids = inputs['input_ids']
  attention_mask =inputs['attention_mask']

  inputs_tf = [
      context_ids,
      attention_mask,
      context_masks_tf
    ]
  
  return inputs_tf, labels_tf , grouped_slice

def prepare_input_with_split(input_df, tokenizer, max_len_pad):
  train_df, val_df = train_test_split(input_df, random_state=42, test_size=0.2) 
  print(train_df.shape, val_df.shape)
  x_train, y_train, _ = prepare_input(train_df, tokenizer, max_len_pad)
  x_val, y_val, _ = prepare_input(val_df, tokenizer, max_len_pad)

  return x_train, y_train, x_val, y_val




def preppare_input_max(slice, tokenizer):
  grouped_slice = slice.groupby('context')
  # len(list_contexts) == B 
  list_contexts = slice['context'].unique().tolist()

  # len(context_masks) == B
  context_masks = []
  labels = []

  max_len_pad =grouped_slice.count().max()['token_indexes']

  for group_name, df_group in grouped_slice:
    current_labels = df_group['loss'].to_list()
    labels.append(current_labels)

    mask_list_entry = []


    for pos in df_group['token_indexes'].to_list():
      mask = [0] * MAX_LEN_SEQ
      if pos[0] != pos[1] and pos[1] > 0 and pos[0] > 0:
        mask[pos[0]:pos[1]] = [1] * (pos[1] - pos[0])
      else: 
        mask[pos[0]] = 1
      mask_list_entry.append(mask)

    if len(mask_list_entry) < max_len_pad: 
      padding = [[0]*MAX_LEN_SEQ for x in range(max_len_pad - len(mask_list_entry))]
      mask_list_entry.extend(padding)
    
    context_masks.append(mask_list_entry)
  
  context_masks_tf = tf.convert_to_tensor(context_masks)

  for entry in labels:  
    current_len = len(entry)
    if current_len < max_len_pad:
      entry.extend([0 for x in range(max_len_pad - current_len)])


  labels_tf =  tf.convert_to_tensor(labels)
  labels_tf = tf.expand_dims(labels_tf, -1)

    #Size (B, 512)
  inputs=tokenizer(list_contexts, padding="max_length", truncation=True, max_length=512, return_tensors='tf')

  context_ids = inputs['input_ids']
  attention_mask =inputs['attention_mask']

  inputs_tf = [
      context_ids,
      attention_mask,
      context_masks_tf
    ]
  
  return inputs_tf, labels_tf , grouped_slice