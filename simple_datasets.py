from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

from transformers import (BertConfig,
                          BertForTokenClassification,
                          BertTokenizer)
import torch
import pandas as pd
from tqdm import tqdm
import string

import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import ast
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from pyg_until import SequenceEncoder, GenresEncoder, load_node_csv, load_edge_csv, node_range
import pickle 

# remove stopwords, keep the negations
keeped_words = ['without', 'within', 'w/o', 'against', 'no', 'not', 'nt', "n't", 'against', "aren't", 'but',
                "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                "needn't", 'no', 'nor', 'not', 'now', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                "won't", 'wouldn', "wouldn't", 'after', 'again', 'aren', 'before', 'couldn', 'didn', 'doesn', 'don',
                'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', "mustn't", 'needn', 'shouldn']
stop_words = set([w for w in stopwords.words('english') if w not in keeped_words] + ["'s", "s'", "please",'refills:*0', 'po','sig','bid', 'tablet', 'mg'])

exclude_dep_type = ['ROOT','punct','quantmod','mwe','dative', 'amod@nmod', 'meta', 'preconj', 'cop','pobj','det:predet']
exclude_pos_type = ['PUNCT']

def exclude_words(x):
  x=x.strip()
  if x in stop_words or x.isdigit() or x in string.punctuation:
    return True
  else:
    return False

def process_conpound(data):
  # data['new_sentence_attribute'] = None
  data['new_mt'] = None
  data['new_dependency'] = None
  data['mt_token_indice'] = None
  data['start2mt_indices'] = None
  #print('exclude_pos_type',exclude_pos_type)
  for d_index,row in data.iterrows():

    sentence_attribute = ast.literal_eval(row['sentence_attribute'])

    new_mt = []
    new_dep = []
    new_mt_tok = []
    start2mt_indices = {}
    for sent_info in sentence_attribute:
      # all_mts = ast.literal_eval(sent_info[3])
      all_mts = sent_info[3]
      new_dependencies = {}
      mt_indices2start = {}
      sent_start = sent_info[1]
      mt_indices = []
      if sent_info[4] != None:
        all_dependencies = sent_info[4]
        mt_indices = sent_info[5]
        for k, mt in enumerate(all_mts):
          start2mt_indices[mt[2][0]] = [j for j in mt[2]]
          for mt_i in range(mt[2][0], mt[2][1]):
            mt_indices2start[mt_i] = (mt[2][0], mt[0])
        for index, dep in all_dependencies.items():
          index = index + sent_start
          if index not in mt_indices:
            temp = all_dependencies[index - sent_start]
            new_temp = []
            for d in temp:
              if d[0] in exclude_dep_type or d[5] in exclude_pos_type or d[6] in exclude_pos_type or exclude_words(d[1]) or exclude_words(d[2]):
                continue
              start = d[3] + sent_start
              end = d[4] + sent_start
              if start == end:
                    continue
              a = mt_indices2start[start][0] if start in mt_indices2start else start
              b = mt_indices2start[end][0] if end in mt_indices2start else end
              if a == b: continue
              a_word = mt_indices2start[start][1] if start in mt_indices2start else d[1]
              b_word = mt_indices2start[end][1] if end in mt_indices2start else d[2]
              if a != start:
                a_pos = 'COMPOUND'
              else:
                a_pos = d[5]
              if b != end:
                b_pos = 'COMPOUND'
              else:
                b_pos = d[6]
              new_temp.append([d[0], a_word, b_word, a, b, a_pos, b_pos])
            new_dependencies[index] = new_temp
          else:
            if index in start2mt_indices:
              temp = []
              cur_indices = start2mt_indices[index]
              cur_indices_list = list(range(cur_indices[0], cur_indices[1]))
              for idx in cur_indices_list:
                cur_dep = all_dependencies[idx - sent_start]
                for dep in cur_dep:
                  if dep[0] in exclude_dep_type or dep[5] in exclude_pos_type or dep[6] in exclude_pos_type or exclude_words(dep[1]) or exclude_words(dep[2]):
                    continue
                  start = dep[3] + sent_start
                  end = dep[4] + sent_start
                  if start == end:
                    continue
                  if start in cur_indices_list and dep[4] in cur_indices_list:
                    continue
                  a = mt_indices2start[start][0] if start in mt_indices2start else start
                  b = mt_indices2start[end][0] if end in mt_indices2start else end
                  if a == b: continue
                  a_word = mt_indices2start[start][1] if start in mt_indices2start else dep[1]
                  b_word = mt_indices2start[end][1] if end in mt_indices2start else dep[2]
                  if a != start:
                    a_pos = 'COMPOUND'
                  else:
                    a_pos = dep[5]
                  if b != end:
                    b_pos = 'COMPOUND'
                  else:
                    b_pos = dep[6]
                  if [dep[0], dep[1], dep[2], a, b, dep[5], dep[6]] not in temp:
                    temp.append([dep[0], a_word, b_word, a, b, a_pos, b_pos])
              new_dependencies[index] = temp
        for d_i, d_d in new_dependencies.items():
          for each_d in d_d:
            if each_d not in new_dependencies[each_d[4]]:
              new_dependencies[each_d[4]].append(each_d)
      sent_info.append(new_dependencies)
      sent_info.append(start2mt_indices)
      sent_info.append(mt_indices)
      new_mt.append(sent_info[3])
      new_dep.append(sent_info[6])
      new_mt_tok.append(sent_info[8])
    data.at[d_index, 'new_mt'] = new_mt
    data.at[d_index, 'new_dependency'] = json.dumps(new_dep)
    data.at[d_index, 'mt_token_indice'] = new_mt_tok
    data.at[d_index, 'start2mt_indices'] = {str(k):v for k,v in start2mt_indices.items()}

  return data

def get_dep_pos_types(args,data,train_or_test):
  record_dep_types = defaultdict(list)
  pos_types = []
  for index,row in data.iterrows():
    all_sent_dep = json.loads(row['new_dependency'])
    for sent_info in all_sent_dep:
      for _,dep in sent_info.items():
        for d in dep:
          if d[0] in ['ROOT','punct']:
            continue
          record_dep_types[d[0]].append('|'.join(sorted((d[1],d[2]))))
          if d[-2] not in pos_types:
            pos_types.append(d[-2])
          if d[-1] not in pos_types:
            pos_types.append(d[-1])

  record_dep_types = {k:Counter(v) for k,v in record_dep_types.items()}
  record_dep_types = {k:dict(sorted(v.items(), key=lambda item: item[1],reverse=True)[:20]) for k,v in record_dep_types.items()}

  # with open(args.output_dir+train_or_test+'_dep_types.pkl', 'wb') as f:
  #   pickle.dump(record_dep_types, f)
  dep_types = list(record_dep_types.keys())
  return pos_types, dep_types

def get_edge_type(row, mt_l, mod_l):
  if row['sources'] in mt_l and row['dest'] in mt_l:
    return 0 # mt-mt
  elif row['sources'] in mt_l and row['dest'] in mod_l:
    return 1 # mt-mod
  elif row['sources'] in mod_l and row['dest'] in mt_l:
    return 2 # mt-mod
  else:
    return 3 # mod-mod

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()

def find_adj_nodes(node, dependencies, start2mt_indices, idx_mapping, max_seq_length, skip = [], directed = False):
  s_word = []
  d_word = []
  sources = []
  dest = []
  sources_pos = []
  dest_pos = []
  dep_type = []

  s_idx = []
  d_idx = []
  mt_info =[]
  for adj_dep in dependencies[str(node)]:
    if adj_dep[3] not in idx_mapping or adj_dep[4] not in idx_mapping:
      continue
    if str(adj_dep[3]) in start2mt_indices:
      if start2mt_indices[str(adj_dep[3])][1] - 1 not in idx_mapping:
        continue
    if str(adj_dep[4]) in start2mt_indices:
      if start2mt_indices[str(adj_dep[4])][1] - 1 not in idx_mapping:
        continue
    if str(adj_dep[3]) in skip or str(adj_dep[4]) in skip:
      continue

    if directed:
      if str(adj_dep[3]) == node:
        d_word.append(adj_dep[2])
        dest.append(str(adj_dep[4]))
        # dest.append(str(adj_dep[3]))
        dest_pos.append(adj_dep[6])
        # dest_pos.append(adj_dep[5])
        dep_type.append(adj_dep[0])
        # if str(adj_dep[3]) in start2mt_indices:
        #   d_idx.append(start2mt_indices[str(adj_dep[3])])
        # else:
        #   d_idx.append((adj_dep[3], adj_dep[3]+1))
        if str(adj_dep[4]) in start2mt_indices:
          # idx_mapping
          # s_idx.append((idx_mapping[start2mt_indices[str(adj_dep[4])][0]][0],idx_mapping[start2mt_indices[str(adj_dep[4])][1]][1]))
          d_idx.append(start2mt_indices[str(adj_dep[4])])

        else:
          d_idx.append((adj_dep[4], adj_dep[4]+1))
    else:

      s_word.append(adj_dep[1])
      d_word.append(adj_dep[2])

      sources.append(str(adj_dep[3]))
      dest.append(str(adj_dep[4]))

      if len(adj_dep[1].split())>1:
        s_pos = 'COMPOUND'
      else:
        s_pos = adj_dep[5]
      if len(adj_dep[2].split())>1:
        d_pos = 'COMPOUND'
      else:
        d_pos = adj_dep[6]
      sources_pos.append(s_pos)
      dest_pos.append(d_pos)
      dep_type.append(adj_dep[0])

      if str(adj_dep[3]) in start2mt_indices:
        s_idx.append(start2mt_indices[str(adj_dep[3])])
      else:
        s_idx.append((adj_dep[3], adj_dep[3]+1))
      if str(adj_dep[4]) in start2mt_indices:
        d_idx.append(start2mt_indices[str(adj_dep[4])])
      else:
        d_idx.append((adj_dep[4], adj_dep[4]+1))

      if str(adj_dep[3]) == node:
        mt_info = [adj_dep[1], str(adj_dep[3]), s_pos, s_idx]
      elif str(adj_dep[4]) == node:
        mt_info = [adj_dep[2], str(adj_dep[4]), d_pos, d_idx]

      # s_word.extend([adj_dep[1], adj_dep[2]])
      # d_word.extend([adj_dep[2], adj_dep[1]])

      # sources.extend([adj_dep[3], adj_dep[4]])
      # dest.extend([adj_dep[4], adj_dep[3]])
      # sources_pos.extend([adj_dep[5], adj_dep[6]])
      # dest_pos.extend([adj_dep[6], adj_dep[5]])
      # dep_type.extend([adj_dep[0],adj_dep[0]])
  if directed:
    return d_word, dest, dest_pos, dep_type, d_idx
  else:
    return s_word, d_word, sources, dest, sources_pos, dest_pos, dep_type, s_idx, d_idx, mt_info

class MIMIC_Depparsed_Dataset(Dataset):
  def __init__(self, data, dep_types, pos_types, args, train_or_test):
    self.data = data
    self.args = args
    self.dep_types = dep_types
    self.pos_types = pos_types
    self.formatted_features = []
    self.train_or_test = train_or_test
    self.all_exclude = ['mm','cc']

    self.mt_per_chunk = []
    self.mod_per_mt = []
    self.graph_per_note = []
    self.nodes_per_graph = []
    self.edges_per_graph = []

    if self.args.load_graph:
      if train_or_test == 'train':
        self.graphs = torch.load(self.args.graph_dir + 'train_graph.pt')
      elif train_or_test == 'test':
        self.graphs = torch.load(self.args.graph_dir + 'test_graph.pt')
    else:
      self.graphs = []
    self.dataset = self.process_examples()
    if not self.args.load_graph:
      torch.save(self.graphs, self.args.output_dir + train_or_test +'_graph.pt')

  def map_index_from_raw_2_tokenized(self, doc):
    idx_mapping = {}
    tokenized_words = []
    doc_tokens =self.args.tokenizer.tokenize(doc.text) 
    for sent_i, sent in enumerate(doc.sents): 
      for w_i, word in enumerate(sent):
        word_t = word.text
        word_tokens = self.args.tokenizer.tokenize(word_t)
        if len(word_tokens) == 0:
          continue
        if len(tokenized_words) + len(word_tokens)+1 > self.args.max_seq_length - 2:
          break
        idx_mapping[word.i] = (len(tokenized_words)+1, len(tokenized_words) + len(word_tokens)+1)
        tokenized_words.extend(word_tokens)
      if len(tokenized_words) + len(word_tokens)+1 > self.args.max_seq_length - 2:
        break
    return idx_mapping

  def encode(self,examples):
    tokenized_note = self.args.tokenizer(examples["TEXT"], padding="max_length",max_length=self.args.max_seq_length, truncation=True)
    return tokenized_note

  def __len__(self):
        return len(self.formatted_features)
  def get_all_items(self):
    out = []
    # if self.args.pure_bert:
    #   out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), torch.tensor(data['Label'])])
    for idx in tqdm(range(len(self.dataset)),desc='get all items'):
        data = self.dataset[idx]
        if not self.args.pure_bert:
          graph = self.graphs[idx]
          out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), graph, torch.tensor(data['Label'])])
        else:
          out.append([torch.tensor(data['ID']), torch.tensor(data['input_ids']), torch.tensor(data['token_type_ids']), torch.tensor(data['attention_mask']), torch.tensor(data['Label'])])
    return out

  def get_range(self, sentence, start, end):
        pre = sentence[:start]
        tokenized_pre = self.args.tokenizer.tokenize(pre)
        word = sentence[start:end]
        tokenized_word = self.args.tokenizer.tokenize(word)
        return torch.tensor([len(tokenized_pre), len(tokenized_pre) + len(tokenized_word)])
  def get_statistics(self):
    # self.mt_per_chunk = []
    # self.mod_per_mt = []
    # self.graph_per_note = []
    # self.nodes_per_graph = []
    # self.edges_per_graph = []
    print(self.train_or_test)
    # print('Ave # of MT per chunk:',sum(self.mt_per_chunk)/len(self.mt_per_chunk))
    # print('Ave # of modifier per MT:',sum(self.mod_per_mt)/len(self.mod_per_mt))
    # print('Ave # of graph per note:',sum(self.graph_per_note)/len(self.graph_per_note))
    # print('Ave # of nodes per graph:',sum(self.nodes_per_graph)/len(self.nodes_per_graph))
    # print('Ave # of edges per graph:',sum(self.edges_per_graph)/len(self.edges_per_graph))

    print('# of MT per chunk: min, max, mean:',min(self.mt_per_chunk),max(self.mt_per_chunk),sum(self.mt_per_chunk)/len(self.mt_per_chunk))
    print('# of modifier per MT: min, max, mean:',min(self.mod_per_mt), max(self.mod_per_mt), sum(self.mod_per_mt)/len(self.mod_per_mt))
    print('# of graph per note: min, max, mean:',min(self.graph_per_note), max(self.graph_per_note), sum(self.graph_per_note)/len(self.graph_per_note))
    print('# of nodes per graph: min, max, mean:',min(self.nodes_per_graph), max(self.nodes_per_graph), sum(self.nodes_per_graph)/len(self.nodes_per_graph))
    print('# of edges per graph: min, max, mean:',min(self.edges_per_graph), max(self.edges_per_graph), sum(self.edges_per_graph)/len(self.edges_per_graph))
  def build_empty_graph(self):
    data = HeteroData()
    data['mt'].x = torch.empty(0, len(self.pos_types)+2, dtype=torch.long)
    data['mod'].x = torch.empty(0, len(self.pos_types)+2, dtype=torch.long)

    data['mt', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mt', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mt', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mt', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mod', 'dep', 'mt'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mod', 'dep', 'mt'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    data['mod', 'dep', 'mod'].edge_index = torch.empty(2, 0, dtype=torch.long)
    data['mod', 'dep', 'mod'].edge_label = torch.empty(0, len(self.dep_types), dtype=torch.long)
    return data

  def construct_graph(self, note_mts, note_dependencies, note_mt_token_indices, nlp, start2mt_indices, idx_mapping):
    # Implementation of graph construction (simplified for this version)
    # Return an empty graph for pure_bert mode
    return self.build_empty_graph()

  def process_examples(self):
    # Process the dataset - since we're in pure_bert mode, we just need to tokenize the text
    tokenized_data = []
    
    # For each row in the dataframe
    for idx, row in tqdm(self.data.iterrows(), desc='Processing examples'):
        # Tokenize the text
        tokens = self.args.tokenizer(
            row['TEXT'], 
            padding="max_length",
            max_length=self.args.max_seq_length, 
            truncation=True,
            return_tensors="pt"
        )
        
        # Create a new data point
        data_point = {
            'ID': row['ID'],
            'input_ids': tokens['input_ids'][0],
            'token_type_ids': tokens['token_type_ids'][0],
            'attention_mask': tokens['attention_mask'][0],
            'Label': row['Label']
        }
        
        tokenized_data.append(data_point)
        
        # If in pure_bert mode, we don't need to create graphs
        if not self.args.pure_bert and not self.args.load_graph:
            self.graphs.append(self.build_empty_graph())
            
    return tokenized_data