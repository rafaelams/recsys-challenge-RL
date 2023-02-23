# RecSys Challenge RL

Deverá ser criado um modelo de Deep Learning para geração de embeddings de itens que será utilizado em um sistema de recomendação por similaridade. O objetivo é que seja utilizado o máximo de features e combinações possíveis (a criatividade é o limite) na criação da representação do item (embedding), de modo a otimizar uma função de recomendação baseada em similaridade a partir dos embeddings criados.

Ref: https://docs.google.com/document/d/1RxrNXsfzq4WlpQBSOXuczdHUXt5SXrQ6hrwilwqKIQg/edit?usp=sharing

## Dataset
https://www.yelp.com/dataset

O datastet da Yelp é composto por 6 arquivos contendo diferentes informações:
* business.json: Contém dados de negócios, incluindo dados de localização, atributos e categorias.
* review.json: Contém dados completos do texto da resenha, incluindo o user_id que escreveu a resenha e o business_id para o qual a resenha foi escrita.
* user.json: User data including the user's friend mapping and all the metadata associated with the user.
* checking.json: Check-ins em um negócio.
* tip.json:  Dicas escritas por um usuário em um negócio. As dicas são mais curtas do que as avaliações e tendem a transmitir sugestões rápidas.
* photo.json: Contains photo data including the caption and classification (one of "food", "drink", "menu", "inside" or "outside").

O objetivo do trabalho é que sejam criados os embeddings dos bussiness, que no caso são os restaurantes a serem recomendados a partir da similaridade. Todas as informações no dataset podem ser utilizadas para criação do embedding, embora nem todas sejam úteis.

### Transforma o Dataset `.json` em `.csv`

O script abaixo transforma o dataset em `.json` em `.csv`.

```bash
python scripts/json_to_csv_converter.py data/yelp_dataset/yelp_academic_dataset_business.json
```

## Baseline Models

É possível utilizar dois modelos baselines para teste, um modelo randomico que consiste em gerar embeddings aleatórios e um modelo baseado em embeddings de texto. 

### Random Model

Script que gera embeddings randomicos para valiadação.

#### Exportar embeddings

script:
```
baseline/extract_random_embedding.py <csv_file> <output_path> 

Params:

csv_file: The csv file to extract the embeddings.
output_path: The output path to save the embeddings and metadata.
```

Example: 
```bash
python baseline/extract_random_embedding.py data/dataset/yelp_dataset/yelp_academic_dataset_business.csv data/models/random
```

Após extrair os embeddings utilizando o script `extract_random_embedding.py` serão criados os arquivos:

- **embeddings.txt**:  Contém apenas os embeddings dos itens
- **metadados.csv**: Contém todos os metadados utilizados junto com o business_id para identificar o embedding do item. Os dois arquivos devem estar na mesma ordem pra dar match.

### Text Embedding

O Base line consiste em utilizar um modelo de linguagem do huggingface para extrair os embeddings dos nomes dos restaurantes utilizando a biblioteca https://github.com/huggingface/transformers


#### Exportar embeddings

script:
```
baseline/extract_text_embedding.py <model_base> <csv_file> <output_path> 

Params:

model_base: The model base to extract the embeddings.
csv_file: The csv file to extract the embeddings.
output_path: The output path to save the embeddings and metadata.
```

Example: 
```bash
python baseline/extract_text_embedding.py bert-base-uncased data/dataset/yelp_dataset/yelp_academic_dataset_business.csv data/output/text_emb
```

Após extrair os embeddings utilizando o script `extract_text_embedding.py` serão criados os arquivos:

- **embeddings.txt**:  Contém apenas os embeddings dos itens
- **metadados.csv**: Contém todos os metadados utilizados junto com o business_id para identificar o embedding do item. Os dois arquivos devem estar na mesma ordem pra dar match.

## Avaliação

A avaliação seguirá a seguinte lógica:

- Iremos separar um determinado grupo de usuários (user.json) para servir como teste.
- Para cada usuário desse grupo, escolheremos o business com a review (review.json) de maior nota para servir como groudtruth da preferência do usuário criando assim o perfil do usuário.
- As demais reviews do usuário (n) servirão como business groudtruth (itens que o usuário interagiu) a serem avaliados em um conjunto com outros k business escolhidos aleatoriamente no dataset (itens que o usuário não interagiu).
- A ordenação dos itens será realizada a partir da similaridade de cosseno em relação ao perfil do usuário, desse modo todos os k+n itens serão ranqueados onde espera-se que os itens groudtruth sejam melhor ranqueados.
- A métrica utilizada para avaliar a lista de ranqueada será o NDCG@5. 

### Dataset de avaliação

`data/evaluation/eval_users.csv`, subsete de 1000 usuários extraidos do `user.json` contendo: 

- **user_id**: Id do usuário
- **user_perfil**: Bussines com 5 estrelas  dado pelo usuário, utilizado para montar o perfil do usuário
- **gt_reclist**: Total de bussines com 4 e 5 estrelas interagido pelo usuário
- **reclist**: Lista contento os bussines do gt_reclist e alguns (10) bussiness aleatórios que devem ser ordenados para avaliação

### Executando a Avaliação

script:
```
evaluation/evaluation.py <embedding_path> <metadados_path>

Params:

embedding_path: Arquivo de embeddings.
metadados_path: Arquivo de metadados
```

Example:

```bash
python evaluation/evaluation.py data/models/random/embeddings.txt data/models/random/metadados.csv

> Avaliação de Embeddings
> Embeddings:  data/models/random/embeddings.txt
> Total Users:  1000
> NDCG@5:  0.4490953192454955
> NDCG@10:  0.5098859573979446
```