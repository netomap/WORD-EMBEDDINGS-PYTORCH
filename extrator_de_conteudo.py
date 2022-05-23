import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser

def obter_conteudo_pagina(url_pagina):
    r'''
    Função que faz o request da página e retorna um list. Na primeira linha é o título do artigo.  
    Já da segunda linha adiante são os parágrafos do artigo.  

    Args:  
        - url_da_pagina.  
    
    Returns:  
        - Lista de parágrafos. 
    '''
    conteudo = requests.get(url_pagina).text
    soup = BeautifulSoup(conteudo, 'html.parser')
    try:
        titulo = soup.find(id='content').text
    except:
        titulo = '-'
    
    paragrafos = [titulo]
    for paragrafo in soup.find_all('p'):
        if ('dir' in paragrafo.attrs and paragrafo['dir'] == 'ltr'):
            paragrafos.append(paragrafo.text)
    
    return paragrafos

def obter_links_do_arquivo(file_path):
    r'''
    Retorna uma lista onde o primeiro item de cada linha é o nome e o segundo representa a url do artigo.
    '''
    arquivo = open(file_path, 'r')
    linhas = arquivo.readlines()
    arquivo.close()

    linhas = [str(link).strip() for link in linhas if str(link).strip() != ''] # remove os \n de cada url e urls vazias

    linhas = list(set(linhas)) # remove as urls duplicadas

    links = []
    for url in linhas:
        pbarra = str(url).rfind('/')
        nome = url[pbarra+1:]
        links.append([nome, url])
    
    return links

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--fp', type=str, help='Nome do arquivo que tem todos os links.')
    parser.add_argument('--dir', type=str, help='Nome do diretório que vai salvar os arquivos.')

    args = parser.parse_args()
    FILE_PATH = args.fp
    DIR_NAME = args.dir

    links = obter_links_do_arquivo(FILE_PATH)
    total_links = len(links)

    for k, (nome_arquivo, url) in enumerate(links):

        conteudo = obter_conteudo_pagina(url)
        with open(f'./{DIR_NAME}/{nome_arquivo}.txt', 'w') as file:
            file.write('\n'.join(conteudo))
        
        print (f'[{k:04}/{total_links}] arquivo {nome_arquivo} salvo com {len(conteudo)} paragrafos.')