# Plano do pipeline de dataset PyTorch

Data do plano: 2026-06-01

## Base verificada

Este plano usa somente informacoes verificadas nos relatorios:

- `reports/EuroSAT_RGB_dataset_inspection.md`
- `reports/problem_definition.md`

Resumo operacional validado:

- dataset oficial: `./EuroSAT_RGB`;
- 27.000 imagens validas;
- imagens `.jpg`, RGB, `64x64`;
- 10 classes organizadas em uma pasta por classe;
- ausencia de split previo de treino, validacao e teste;
- ausencia de notebooks, scripts, modelos, pesos, logs ou metadados prontos dentro de `EuroSAT_RGB`;
- estrutura compativel com `torchvision.datasets.ImageFolder`.

O problema definido e classificacao de uso e cobertura do solo em imagens RGB de sensoriamento remoto.

## Decisao geral

O pipeline deve usar `torchvision.datasets.ImageFolder` como mecanismo principal de leitura das imagens, com splits estratificados persistidos em metadados dentro de `data/processed/`.

Nao sera criada copia fisica das 27.000 imagens em `data/processed/` nesta etapa. O diretorio `EuroSAT_RGB` continua sendo a fonte unica das imagens, e `data/processed/` guarda os arquivos de controle necessarios para reproduzir o split, as classes e a normalizacao.

Motivos:

- a estrutura original ja esta adequada para `ImageFolder`;
- evita duplicacao desnecessaria de 27.000 imagens;
- mantem rastreabilidade por caminho relativo;
- permite aplicar transforms diferentes para treino, validacao e teste usando tres instancias de `ImageFolder` apontando para a mesma raiz e filtradas pelos indices do split.

## Estrategia de split

Proporcao definida:

- treino: 70%;
- validacao: 15%;
- teste: 15%.

Regras:

- split estratificado por classe;
- seed fixa: `42`;
- ordenacao deterministica dos caminhos antes do embaralhamento;
- nenhuma imagem pode aparecer em mais de um split;
- as classes devem manter os nomes originais em ingles para consistencia com `ImageFolder`;
- as duas CNNs futuras devem usar exatamente o mesmo split para comparacao justa.

Contagens esperadas por classe:

| Classe | Total | Treino | Validacao | Teste |
|---|---:|---:|---:|---:|
| AnnualCrop | 3.000 | 2.100 | 450 | 450 |
| Forest | 3.000 | 2.100 | 450 | 450 |
| HerbaceousVegetation | 3.000 | 2.100 | 450 | 450 |
| Highway | 2.500 | 1.750 | 375 | 375 |
| Industrial | 2.500 | 1.750 | 375 | 375 |
| Pasture | 2.000 | 1.400 | 300 | 300 |
| PermanentCrop | 2.500 | 1.750 | 375 | 375 |
| Residential | 3.000 | 2.100 | 450 | 450 |
| River | 2.500 | 1.750 | 375 | 375 |
| SeaLake | 3.000 | 2.100 | 450 | 450 |
| **Total** | **27.000** | **18.900** | **4.050** | **4.050** |

Como todas as quantidades por classe sao divisiveis pela proporcao escolhida, nao deve haver arredondamento ambiguo.

## Estrutura planejada de `data/processed/`

Estrutura prevista:

```text
data/
└── processed/
    ├── class_to_idx.json
    ├── dataset_manifest.csv
    ├── split_config.json
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── train_stats.json
```

Conteudo planejado:

- `dataset_manifest.csv`: uma linha por imagem com `relative_path`, `class_name`, `class_idx`, `filename`, `width`, `height`, `mode`.
- `train.csv`, `val.csv`, `test.csv`: subconjuntos do manifesto com `relative_path`, `class_name`, `class_idx`, `split`.
- `class_to_idx.json`: mapeamento gerado pelo `ImageFolder`, preservando a ordem deterministica das classes.
- `split_config.json`: raiz do dataset, seed, proporcoes, data de geracao, totais por split e total por classe.
- `train_stats.json`: media e desvio padrao RGB calculados somente no split de treino.

O plano nao exige `data/processed/train/<classe>/...`; os caminhos das imagens continuam apontando para `EuroSAT_RGB/<classe>/<arquivo>.jpg`.

## Escolha entre `ImageFolder` e `Dataset` customizado

Decisao: usar `ImageFolder` com indices persistidos.

Implementacao planejada para etapa futura:

1. Instanciar um `ImageFolder(root="EuroSAT_RGB")` sem transform para descobrir `samples`, `classes` e `class_to_idx`.
2. Gerar e salvar o manifesto e os CSVs de split.
3. Para treino, validacao e teste, criar instancias separadas de `ImageFolder(root="EuroSAT_RGB", transform=...)`.
4. Filtrar cada instancia com `torch.utils.data.Subset` ou helper equivalente a partir dos indices definidos nos CSVs.
5. Criar `DataLoader` separado para `train`, `val` e `test`.

Nao ha necessidade de `Dataset` customizado completo porque:

- os rotulos ja estao codificados pela estrutura de pastas;
- o dataset tem modo, tamanho e extensao consistentes;
- nao existem metadados adicionais obrigatorios para compor o rotulo;
- `ImageFolder` reduz codigo proprio e facilita auditoria academica.

Um helper proprio em `src/dataset.py` pode ser criado apenas para carregar os CSVs, resolver os indices e montar os DataLoaders.

## Tamanho de entrada

Tamanho definido: `64x64`.

Justificativa:

- todas as imagens originais ja possuem `64x64`;
- redimensionar para cima nao adicionaria informacao real;
- manter `64x64` reduz custo computacional e e adequado para CNNs treinadas do zero;
- o mesmo tamanho deve ser usado nas duas arquiteturas para comparacao justa.

As transforms ainda devem incluir `Resize((64, 64))` ou validacao equivalente para proteger o pipeline contra imagens futuras fora do padrao.

## Normalizacao

Decisao: calcular media e desvio padrao por canal usando somente o split de treino.

Regras:

- nao usar estatisticas de validacao ou teste para evitar vazamento de informacao;
- nao usar normalizacao do ImageNet, pois modelos pre-treinados sao proibidos e o dominio e sensoriamento remoto RGB;
- salvar os valores finais em `data/processed/train_stats.json`;
- reutilizar exatamente os mesmos valores em treino, validacao, teste e notebook demonstrativo.

Formato planejado:

```json
{
  "mean": [0.0, 0.0, 0.0],
  "std": [1.0, 1.0, 1.0],
  "source_split": "train",
  "num_images": 18900,
  "input_size": [64, 64]
}
```

Os valores `mean` e `std` acima sao apenas placeholders de formato. A etapa de implementacao deve calcular os valores reais a partir das imagens de treino.

## Augmentations e transforms

As transforms devem ser separadas por split.

### Treino

Transformacoes planejadas:

1. `Resize((64, 64))`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomVerticalFlip(p=0.5)`
4. `RandomRotation(degrees=15)`
5. `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)` com probabilidade moderada
6. `ToTensor()`
7. `Normalize(mean=train_mean, std=train_std)`

Justificativa:

- flips e pequenas rotacoes sao plausiveis para recortes de sensoriamento remoto, pois a orientacao absoluta da imagem nao deve definir a classe;
- variacoes moderadas de cor e contraste ajudam robustez sem descaracterizar classes pequenas;
- augmentations agressivas devem ser evitadas por causa da baixa resolucao `64x64` e da similaridade entre classes agricolas.

### Validacao

Transformacoes planejadas:

1. `Resize((64, 64))`
2. `ToTensor()`
3. `Normalize(mean=train_mean, std=train_std)`

### Teste

Transformacoes planejadas:

1. `Resize((64, 64))`
2. `ToTensor()`
3. `Normalize(mean=train_mean, std=train_std)`

Validacao e teste nao devem usar augmentations aleatorias.

## DataLoaders

Configuracao inicial planejada:

- batch size inicial: `64`;
- `shuffle=True` apenas no treino;
- `shuffle=False` em validacao e teste;
- `num_workers`: configuravel, com valor seguro inicial `2`;
- `pin_memory=True` quando o device for CUDA;
- labels como `torch.long`;
- imagens como tensor `float32` no formato `[batch, 3, 64, 64]`.

O device deve ser detectado automaticamente em etapa futura: `cuda`, `mps` ou `cpu`.

## Criterios de validacao do dataset processado

A etapa de implementacao do pipeline so deve ser considerada correta se todos os criterios abaixo forem satisfeitos:

1. `data/processed/` existe com todos os metadados planejados.
2. `class_to_idx.json` contem exatamente 10 classes.
3. As classes batem com as pastas reais de `EuroSAT_RGB`.
4. `dataset_manifest.csv` possui exatamente 27.000 linhas de imagens.
5. Todos os caminhos relativos do manifesto existem no disco.
6. `train.csv`, `val.csv` e `test.csv` somam 27.000 imagens.
7. As contagens por split sao exatamente 18.900, 4.050 e 4.050.
8. As contagens por classe seguem a tabela planejada.
9. Nao existe sobreposicao de `relative_path` entre treino, validacao e teste.
10. Todas as imagens verificadas abrem como RGB `64x64`.
11. A media e o desvio padrao sao calculados somente a partir do treino.
12. Um batch de treino possui shape `[batch, 3, 64, 64]`.
13. Um batch de validacao e teste possui shape `[batch, 3, 64, 64]`.
14. Os labels ficam no intervalo `[0, 9]`.
15. A geracao do split com a mesma seed produz os mesmos CSVs.

## Arquivos previstos para a implementacao futura

A proxima etapa podera criar:

- `src/dataset.py`: funcoes para carregar manifesto/splits e construir datasets/DataLoaders;
- `src/transforms.py`: factories de transforms de treino, validacao, teste e inferencia;
- `src/data/prepare_dataset.py` ou `scripts/prepare_dataset.py`: geracao dos CSVs e estatisticas;
- `data/processed/`: metadados gerados;
- `reports/dataset_pipeline_implementation.md`: relatorio da implementacao.

Esta etapa nao deve criar treinamento, arquiteturas CNN, pesos, metricas de modelo ou notebooks de avaliacao.

## Decisoes registradas

- Split: estratificado 70/15/15 com seed `42`.
- Leitura: `ImageFolder` com indices persistidos em CSV.
- Imagens processadas: sem copia fisica; usar caminhos relativos para `EuroSAT_RGB`.
- Entrada: RGB `64x64`.
- Normalizacao: media e desvio padrao calculados apenas no treino.
- Augmentation: apenas no treino, moderado e compativel com sensoriamento remoto.
- Validacao: contagens, classes, ausencia de vazamento entre splits e batches PyTorch.
