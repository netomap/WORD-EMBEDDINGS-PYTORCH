import requests
import re
from bs4 import BeautifulSoup
from argparse import ArgumentParser

def listar_links_dos_artigos(url_pagina_pesquisa):
    r'''
    Esta função retora um vetor com as urls de cada artigo listado na página.  
    Args:  
        - Url da página com os artigos
    Returns:  
        - Lista com os links dos artigos.
    '''
    conteudo = requests.get(url_pagina_pesquisa).text
    soup = BeautifulSoup(conteudo, 'html.parser')
    links_da_pagina = []
    for link in soup.find_all('a'):
        if ('class' in link.attrs and 'qa-heading-link' in link['class']):
            links_da_pagina.append('https://www.bbc.com' + link['href'])
    
    return links_da_pagina

def obter_links_e_salvar(topico, n_pagina):
    url = f'{topico}{n_pagina}'
    links = listar_links_dos_artigos(url)
    with open('links.txt', 'a') as file:
        file.write('\n'.join(links) + '\n')
    
    return len(links)

if (__name__ == '__main__'):

    parser = ArgumentParser()
    parser.add_argument('--n', type=int, help='Número final de páginas a procurar links.')
    parser.add_argument('--t', type=str, help='Link básico do tópico, já incluso o page/')

    args = parser.parse_args()
    N = args.n
    TOPICO = args.t

    for n in range(1, N, 1):
        n_links = obter_links_e_salvar(TOPICO, n)
        print (f'Pagina {n} com {n_links} links.')