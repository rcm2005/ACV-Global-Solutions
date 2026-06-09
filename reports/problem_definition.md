# Definicao do problema

Data da definicao: 2026-06-01

## Base utilizada

Esta definicao foi produzida a partir da inspecao real do dataset `./EuroSAT_RGB`, registrada em `reports/EuroSAT_RGB_dataset_inspection.md`.

Resumo verificado:

- dataset com 27.000 imagens validas;
- imagens `.jpg`, RGB, com dimensao `64x64` pixels;
- estrutura plana com uma pasta por classe;
- ausencia de split previo de treino, validacao e teste;
- ausencia de notebooks, scripts, modelos, pesos, logs ou metadados prontos dentro de `EuroSAT_RGB`;
- imagens compativeis com cenas de sensoriamento remoto/cobertura do solo.

## Nome do problema

Classificacao de uso e cobertura do solo em imagens RGB de sensoriamento remoto.

## Tipo de tarefa

A tarefa e classificacao supervisionada de imagens.

Cada entrada sera uma imagem RGB `64x64` proveniente do dataset `EuroSAT_RGB`. A saida esperada sera uma unica classe de uso ou cobertura do solo entre as 10 classes detectadas na estrutura de pastas do dataset.

Esta e uma tarefa de classificacao de imagens porque:

- cada arquivo de imagem possui um rotulo explicito derivado da pasta em que esta armazenado;
- o modelo deve aprender padroes visuais de textura, forma, cor e organizacao espacial;
- para cada imagem, ha uma unica categoria correta;
- a predicao final deve escolher uma classe discreta entre categorias fechadas.

## Classes consolidadas

| Classe | Imagens | Interpretacao operacional |
|---|---:|---|
| AnnualCrop | 3.000 | areas de cultivo anual |
| Forest | 3.000 | areas florestais |
| HerbaceousVegetation | 3.000 | vegetacao herbacea |
| Highway | 2.500 | rodovias e infraestrutura linear |
| Industrial | 2.500 | zonas industriais |
| Pasture | 2.000 | pastagens |
| PermanentCrop | 2.500 | cultivos permanentes |
| Residential | 3.000 | areas residenciais |
| River | 2.500 | rios e cursos d'agua |
| SeaLake | 3.000 | mares, lagos e grandes corpos d'agua |
| **Total** | **27.000** | 10 classes |

## Justificativa da escolha

O dataset e adequado para este problema porque ja esta organizado por classe, possui imagens validas e padronizadas em tamanho, modo de cor e extensao, e representa categorias diretamente relacionadas a uso e cobertura do solo. A ausencia de split previo exige que o projeto crie uma divisao propria entre treino, validacao e teste, preferencialmente estratificada por classe e com semente fixa.

A tarefa tambem e adequada para CNNs treinadas do zero, pois as classes dependem de padroes visuais locais e espaciais, como textura de vegetacao, organizacao urbana, formas lineares de rodovias e rios, e diferencas entre areas agricolas, naturais e construidas.

## Conexao com a Industria Espacial

A conexao com a Industria Espacial e tecnicamente plausivel porque o dataset contem imagens RGB de sensoriamento remoto/cobertura do solo. Em fluxos reais de observacao da Terra, imagens orbitais ou aereas podem ser processadas por modelos de visao computacional para classificar automaticamente regioes do territorio.

Uma solucao desse tipo pode apoiar:

- monitoramento de mudancas de uso e cobertura do solo;
- acompanhamento de areas agricolas, florestais, urbanas e industriais;
- analise de infraestrutura e expansao urbana;
- identificacao de corpos d'agua e cursos fluviais;
- triagem automatica de grandes volumes de imagens coletadas por plataformas de observacao da Terra.

Como a inspecao local nao identificou metadados adicionais dentro de `EuroSAT_RGB`, a documentacao deve tratar a aplicacao como compativel com observacao da Terra e sensoriamento remoto, sem depender de afirmacoes sobre sensor, orbita ou satelite especifico que nao tenham sido verificadas no diretorio.

## ODS relacionados

### ODS 13 - Acao Climatica

O problema pode apoiar monitoramento ambiental e analise de mudancas de cobertura do solo, incluindo vegetacao, areas agricolas, florestas, agua e ocupacao urbana. Esse tipo de classificacao contribui para sistemas de acompanhamento territorial usados em estudos climaticos e ambientais.

### ODS 15 - Vida Terrestre

As classes `Forest`, `HerbaceousVegetation`, `Pasture`, `AnnualCrop` e `PermanentCrop` permitem analisar vegetacao, agricultura, pastagens e areas naturais. A classificacao automatica pode apoiar conservacao, gestao territorial e acompanhamento de pressoes sobre ecossistemas terrestres.

### ODS 9 - Industria, Inovacao e Infraestrutura

Como ODS secundario, a solucao se relaciona ao uso de tecnologia e inovacao aplicada a sensoriamento remoto, automacao de analise geoespacial e apoio a sistemas de decisao para infraestrutura, industria e planejamento urbano.

## Usuario-alvo

Usuarios provaveis:

- equipes de sensoriamento remoto e geoprocessamento;
- analistas ambientais;
- equipes de planejamento urbano e territorial;
- pesquisadores em agricultura, uso do solo e monitoramento ambiental;
- empresas ou orgaos que operam sistemas de observacao da Terra.

## Uso em sistema real

Em um sistema real, o modelo poderia receber recortes RGB padronizados de imagens de observacao da Terra e classificar cada recorte em uma categoria de uso/cobertura do solo. As predicoes poderiam alimentar mapas tematicos, paineis de monitoramento, alertas de mudanca territorial ou etapas de triagem para revisao humana.

Fluxo previsto:

1. receber imagens RGB ou recortes extraidos de uma cena maior;
2. aplicar o mesmo pre-processamento usado no treinamento;
3. executar uma CNN treinada do zero no projeto;
4. retornar a classe prevista e a confianca da predicao;
5. registrar exemplos corretos, erros e possiveis ambiguidades para auditoria.

## Riscos tecnicos

- As imagens possuem baixa resolucao (`64x64`), o que limita detalhes finos.
- Classes agricolas e de vegetacao podem ser visualmente semelhantes, especialmente `AnnualCrop`, `PermanentCrop`, `Pasture` e `HerbaceousVegetation`.
- Classes com estruturas lineares ou bordas podem gerar confusao, como `River`, `Highway` e algumas areas urbanas ou agricolas.
- A classe `Pasture` tem menos imagens que as classes maiores, com 2.000 exemplos contra 3.000 nas maiores classes.
- A meta de 88% de acuracia em teste foi validada experimentalmente com split estratificado, matriz de confusao e analise de erros.

## Decisoes implementadas

- Proporcao final definida em `70%` treino, `15%` validacao e `15%` teste.
- Split estratificado registrado em CSV dentro de `data/processed/`, sem copia fisica das imagens.
- Classes mantidas com os nomes originais em ingles para consistencia com `ImageFolder`.
- Transforms de treino, validacao, teste e inferencia definidos em `src/transforms.py`, com normalizacao calculada no treino.
- Duas arquiteturas proprias implementadas e treinadas do zero: `BaselineCNN` e `ImprovedCNN`.

## Decisao registrada

O problema final do projeto fica definido como classificacao de uso e cobertura do solo em imagens RGB de sensoriamento remoto, usando as 10 classes reais detectadas em `./EuroSAT_RGB`.
