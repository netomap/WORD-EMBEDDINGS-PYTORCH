# Word-Embedding

## Coleta de Dados


- Este repositório é uma treinamento/aprendizado na criação de word-embedding utilizando [Pytorch](https://pytorch.org/).

- O corpus atodado foram mais de 3.000 artigos captados de uma fonte de informação [BBC](https://www.bbc.com/portuguese).  

- Inicialmente, foi criado um script que percorre por todas as páginas de cada subtema na página inicial da BBC e pegou-se os links que estão nelas. Para cada subtema foram inseridos os links no arquivo [arquivo](links.txt).

- Depois de eliminados os links repetidos, para cada link foram pegos todos os parágrafos de cada artigo e seu título. Esse conteúdo foi salvo em cada arquivo individual na pasta [artigos](./artigos/). 

- **TODO O CONTEÚDO AQUI LISTADO NO DIRETÓRIO FOI PEGO NA PÁGINA DA [BBC](https://www.bbc.com/portuguese)**

---

## Escolha do modelo

- O script para criação da word-embedding está no arquivo [Word2Vec](word2vec.ipynb).  

- A escolha do algoritmo para treinamento e criação de **word-embedding** do corpus escohido foi um [CBOW](https://en.wikipedia.org/wiki/Bag-of-words_model#CBOW).