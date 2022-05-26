from argparse import ArgumentParser

parser = ArgumentParser(prog='word2vec.py', description='Script que cria um word-2-vec utilizando algoritmo CBOW, através de um dataset que foi baixado de artigos da internet.')
parser.add_argument('--edim', type=int, default=32, help='Embedding DIM')
parser.add_argument('--e', type=int, default=5, help='Número de épocas de treinamento')
parser.add_argument('--b', type=int, default=64, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learming Rate')
parser.add_argument('--rsw', type=str, default=False, help='Remover STOPWORDS')
parser.add_argument('--s', type=str, default=False, help="Considerar Stemming")
parser.add_argument('--fmin', type=int, default=2, help='Frequência Mínima para considerar dentro do vocabulário')
parser.add_argument('--nw', type=int, default=3, help='Representa o número de vizinhos antes e depois da palavra central para analizar.')

args = parser.parse_args()
print (args)

from datetime import datetime

def log(frase):
    print (f"[{datetime.strftime(datetime.now(), '%H:%M:%S')}] - {frase}")

log('Importando módulos para executar treinamento...')

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from collections import Counter
from random import choice, random, randint
import os
import re
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from unicodedata import normalize

EMBEDDING_DIM = args.edim
N_EPOCHS = args.e
BATCH_SIZE = args.b
LEARNING_RATE = args.lr
RETIRAR_STOP_WORDS = True if args.rsw == 'True' else False
ESTEMIZAR = True if args.s == 'True' else False
FREQUENCIA_MINIMA = args.fmin
N_WINDOW = args.nw

nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')

log ('Listando todos arquivos...') # ===========================================================

texts_dir = './artigos'
texts_list = [f'{texts_dir}/{file}' for file in  os.listdir(texts_dir)]
log (f'Total de arquivos: {len(texts_list)}')

todas_sentencas = []
for text in texts_list:
    arquivo = open(text, 'r', encoding='utf-8')
    sentencas = arquivo.readlines()
    arquivo.close()

    todas_sentencas.append(sentencas)

todas_sentencas = np.array(todas_sentencas, dtype=np.object)
todas_sentencas = np.hstack(todas_sentencas)

df = pd.DataFrame(todas_sentencas, columns=['sentencas'])
log (f'Total de sentenças do corpus: {len(df)}.')

log ('Limpando as sentenças...') # ===========================================================

stemmer = nltk.stem.RSLPStemmer() # Objeto para "stemização" em português

def remover_acentos(frase):
    return normalize('NFKD', frase).encode('ASCII', 'ignore').decode('ASCII')

if (RETIRAR_STOP_WORDS):
    stopwords = [remover_acentos(sw) for sw in  nltk.corpus.stopwords.words('portuguese')]
else:
    stopwords = []

def normalizar_frase(frase):
    frase = str(frase).lower()
    frase = remover_acentos(frase)
    frase = re.sub(r'\w*[0-9]+\w*', ' ', frase)
    frase = re.sub(r'[\[\]º!"#$%&\'()*+,-./:;<=>?@^_`~]+', ' ', frase) # remove caracteres especiais
    
    tokens = [stemmer.stem(tk) if ESTEMIZAR else tk for tk in word_tokenize(frase) if tk not in stopwords] # removendo stopwords e estemizando

    return tokens

df['tokens'] = df['sentencas'].apply(normalizar_frase)

log ('Criando vocabulário e Dicionário') # ===========================================================

counter = Counter()
for tokens in df['tokens'].values:
    counter.update(tokens)

vocab_words = [tk for tk,freq in list(counter.items()) if freq >= FREQUENCIA_MINIMA]
words2idx = {w:n for n,w in enumerate(vocab_words)}
idx2words = {n:w for n,w in enumerate(vocab_words)}
log (f'Vocab size: {len(vocab_words)}')

log ('Refatorando as sentenças e pegando somente as palavras do vocabulário...') # ===========================================================

sentencas_final = []
for sentenca in tqdm(df['tokens'].values, ncols=100):
    vetor = [words2idx[w] for w in sentenca if w in vocab_words]
    sentencas_final.append(vetor)

log ('Preparando o dataset...') # ===========================================================

class custom_dataset(Dataset):

    def __init__(self, n_window, lista_sentencas):
        self.lista_sentencas = lista_sentencas
        self.tensor_x, self.tensor_y = [], []
        for sentenca in tqdm(self.lista_sentencas, ncols=100):
            for ind_tk in range(len(sentenca)-2*n_window):
                vetorx = sentenca[ind_tk:ind_tk+2*n_window+1]
                vetory = vetorx.pop(n_window)
                self.tensor_x.append(vetorx)
                self.tensor_y.append(vetory)
        
        self.tensor_x = torch.tensor(self.tensor_x, dtype=torch.long)
        self.tensor_y = torch.tensor(self.tensor_y)

    def __getitem__(self, index):
        return self.tensor_x[index], self.tensor_y[index]
    
    def __len__(self):
        return len(self.tensor_x)

dataset = custom_dataset(n_window=N_WINDOW, lista_sentencas=sentencas_final)
log (f'Total de linhas do dataset: {len(dataset)}')

log ('Criando a classe modelo...') # ===========================================================

class Modelo_CBOW(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super(Modelo_CBOW, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, max_norm=1)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(axis=1)
        return self.linear(x)


log ('Treinamento...') # ===========================================================

# ======================== FUNÇÃO QUE SALVA AS WORD EMBEDDING ======================
def salvar_word_embedding(epoch):
    vetor = list(model.parameters())[0].detach().cpu().numpy() # retirando os pesos da camada embedding
    vetor = np.hstack([np.array(vocab_words).reshape(-1, 1), vetor]) # preparando a matriz colocando as vocab_words na primeira coluna
    colunas = np.hstack(['word', [f'dim_{k}' for k in range(EMBEDDING_DIM)]]) # preparando os nomes das colunas da matriz
    df = pd.DataFrame(vetor, columns=colunas).set_index('word') # criando o df e setando seu índice
    df.to_csv(f'word_embedding_{epoch}.csv') # salvando
# ======================== FUNÇÃO QUE SALVA AS WORD EMBEDDING ======================

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Modelo_CBOW(embedding_dim=EMBEDDING_DIM, vocab_size=len(vocab_words))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)
model.train()

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(N_EPOCHS):
    loss_epoch, total_itens = 0, 0
    for x, y_real in tqdm(dataloader, ncols=70):
        x, y_real = x.to(DEVICE), y_real.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y_real)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        total_itens += len(x)
    
    loss_epoch = loss_epoch/total_itens
    log (f'Epoch: {epoch:02} train_loss: {round(loss_epoch,4)}')
    salvar_word_embedding(epoch)