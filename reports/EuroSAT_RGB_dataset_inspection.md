# Inspecao do dataset EuroSAT_RGB

Data da inspecao: 2026-06-01

## Objetivo

Inspecionar o diretorio `./EuroSAT_RGB` antes da definicao final do problema de classificacao de imagens, conforme as politicas do projeto.

## Estrutura encontrada

O dataset esta organizado em uma estrutura plana de pastas por classe:

```text
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

Nao foi encontrado split previo de `train`, `val` e `test` dentro de `EuroSAT_RGB`.

## Quantidade de imagens por classe

| Classe | Imagens |
|---|---:|
| AnnualCrop | 3.000 |
| Forest | 3.000 |
| HerbaceousVegetation | 3.000 |
| Highway | 2.500 |
| Industrial | 2.500 |
| Pasture | 2.000 |
| PermanentCrop | 2.500 |
| Residential | 3.000 |
| River | 2.500 |
| SeaLake | 3.000 |
| **Total** | **27.000** |

## Formato dos arquivos

- Total de arquivos encontrados: 27.000.
- Total de imagens validas: 27.000.
- Extensoes encontradas: `.jpg` em 27.000 arquivos.
- Extensoes nao-imagem: nenhuma.
- Modos de cor: `RGB` em 27.000 imagens.
- Dimensoes: `64x64` pixels em 27.000 imagens.

Todas as classes possuem imagens com dimensao `64x64` e modo `RGB`.

## Amostras verificadas

Foram verificados caminhos de exemplo por classe:

| Classe | Exemplos |
|---|---|
| AnnualCrop | `EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg`, `EuroSAT_RGB/AnnualCrop/AnnualCrop_10.jpg`, `EuroSAT_RGB/AnnualCrop/AnnualCrop_100.jpg` |
| Forest | `EuroSAT_RGB/Forest/Forest_1.jpg`, `EuroSAT_RGB/Forest/Forest_10.jpg`, `EuroSAT_RGB/Forest/Forest_100.jpg` |
| HerbaceousVegetation | `EuroSAT_RGB/HerbaceousVegetation/HerbaceousVegetation_1.jpg`, `EuroSAT_RGB/HerbaceousVegetation/HerbaceousVegetation_10.jpg`, `EuroSAT_RGB/HerbaceousVegetation/HerbaceousVegetation_100.jpg` |
| Highway | `EuroSAT_RGB/Highway/Highway_1.jpg`, `EuroSAT_RGB/Highway/Highway_10.jpg`, `EuroSAT_RGB/Highway/Highway_100.jpg` |
| Industrial | `EuroSAT_RGB/Industrial/Industrial_1.jpg`, `EuroSAT_RGB/Industrial/Industrial_10.jpg`, `EuroSAT_RGB/Industrial/Industrial_100.jpg` |
| Pasture | `EuroSAT_RGB/Pasture/Pasture_1.jpg`, `EuroSAT_RGB/Pasture/Pasture_10.jpg`, `EuroSAT_RGB/Pasture/Pasture_100.jpg` |
| PermanentCrop | `EuroSAT_RGB/PermanentCrop/PermanentCrop_1.jpg`, `EuroSAT_RGB/PermanentCrop/PermanentCrop_10.jpg`, `EuroSAT_RGB/PermanentCrop/PermanentCrop_100.jpg` |
| Residential | `EuroSAT_RGB/Residential/Residential_1.jpg`, `EuroSAT_RGB/Residential/Residential_10.jpg`, `EuroSAT_RGB/Residential/Residential_100.jpg` |
| River | `EuroSAT_RGB/River/River_1.jpg`, `EuroSAT_RGB/River/River_10.jpg`, `EuroSAT_RGB/River/River_100.jpg` |
| SeaLake | `EuroSAT_RGB/SeaLake/SeaLake_1.jpg`, `EuroSAT_RGB/SeaLake/SeaLake_10.jpg`, `EuroSAT_RGB/SeaLake/SeaLake_100.jpg` |

A inspecao visual de amostras confirmou imagens RGB de sensoriamento remoto/cobertura do solo, incluindo areas agricolas, vegetacao, rodovias, areas industriais, areas residenciais, rios e corpos d'agua.

## Integridade

- Imagens corrompidas: 0.
- Duplicatas exatas por hash SHA-256: 0 grupos encontrados.
- Arquivos vazios ou fora do padrao de imagem: nenhum encontrado pela varredura.

## Artefatos prontos encontrados

Foi feita busca dentro de `EuroSAT_RGB` por:

- notebooks: `*.ipynb`;
- scripts: `*.py`;
- modelos/pesos: `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.keras`, `*.onnx`;
- artefatos serializados: `*.pkl`, `*.joblib`;
- metadados/logs: `*.json`, `*.csv`, `*.yaml`, `*.yml`, `*.log`.

Resultado: nenhum notebook, script, modelo, peso, log ou metadado foi encontrado dentro de `EuroSAT_RGB`.

## Uso com ImageFolder

O dataset pode ser carregado diretamente com `torchvision.datasets.ImageFolder`, pois possui uma pasta por classe e imagens diretamente dentro de cada pasta.

Como nao existe split previo, o projeto deve criar uma divisao propria de treino, validacao e teste. Opcoes adequadas:

- usar `ImageFolder` sobre `EuroSAT_RGB` e criar indices estratificados para `train`, `val` e `test`;
- ou gerar uma copia/split organizado em `data/processed/train`, `data/processed/val` e `data/processed/test`.

Para reprodutibilidade academica, o split deve ser estratificado por classe e usar semente fixa.

## Adequacao para classificacao

O dataset e adequado para classificacao supervisionada de imagens:

- as classes estao explicitas nos nomes das pastas;
- ha 10 classes de uso/cobertura do solo;
- todas as imagens possuem o mesmo tamanho, extensao e modo de cor;
- nao ha imagens corrompidas detectadas;
- a estrutura e compativel com PyTorch e `ImageFolder`.

## Balanceamento

Existe desequilibrio leve a moderado:

- maior quantidade por classe: 3.000 imagens;
- menor quantidade por classe: 2.000 imagens em `Pasture`;
- razao menor/maior: 0,67.

O desequilibrio nao impede o treinamento, mas recomenda split estratificado e acompanhamento da matriz de confusao por classe.

## Riscos para atingir 88% de acuracia

Riscos principais:

- imagens pequenas de `64x64`, com limite de detalhe espacial;
- classes visualmente proximas, especialmente `AnnualCrop`, `PermanentCrop`, `Pasture` e `HerbaceousVegetation`;
- possivel confusao entre `River`, `Highway` e bordas de areas urbanas/agricolas por estruturas lineares;
- desequilibrio relativo em `Pasture`.

Mesmo com esses riscos, o dataset parece viavel para buscar 88% de acuracia no teste com CNNs proprias treinadas do zero, data augmentation moderado e validacao adequada. Caso a meta nao seja atingida, a justificativa tecnica deve considerar baixa resolucao, similaridade visual entre classes, capacidade da arquitetura, overfitting/underfitting e distribuicao das classes.

## Problema de classificacao sugerido

Nome sugerido: classificacao de uso e cobertura do solo em imagens RGB de sensoriamento remoto.

Tarefa: classificar uma imagem RGB `64x64` em uma das 10 classes detectadas:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River` e `SeaLake`.

Conexao com Industria Espacial: o problema e tecnicamente plausivel para fluxos de observacao da Terra, nos quais imagens orbitais ou aereas sao classificadas para monitoramento territorial, planejamento urbano, agricultura, infraestrutura e analise ambiental.

ODS relacionados:

- ODS 13 - Acao Climatica, por apoiar monitoramento ambiental e mudancas de cobertura do solo;
- ODS 15 - Vida Terrestre, por apoiar analise de vegetacao, florestas, pastagens e uso do solo;
- ODS 9 - Industria, Inovacao e Infraestrutura, como apoio secundario a sistemas de sensoriamento remoto e decisao baseada em dados.

## Proximos arquivos a criar ou alterar

Etapas recomendadas apos esta inspecao:

- criar/atualizar documento de definicao do problema;
- criar pipeline de split estratificado em `data/processed/`;
- criar `src/config.py`, `src/dataset.py` e `src/transforms.py`;
- criar duas arquiteturas proprias em `src/models/baseline_cnn.py` e `src/models/improved_cnn.py`;
- criar notebooks de treinamento e avaliacao conforme o brief;
- atualizar `README.md` com reproducibilidade, problema, dataset, ODS e regras de uso.

