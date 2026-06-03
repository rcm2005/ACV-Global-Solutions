# Implementacao do pipeline de dataset PyTorch

Data da implementacao: 2026-06-01

## Escopo

Esta etapa implementou o pipeline de dataset para o problema de classificacao de uso e cobertura do solo em imagens RGB de sensoriamento remoto com `EuroSAT_RGB`.

Nao houve treinamento de CNNs, criacao de pesos, metricas de modelo ou notebooks de avaliacao nesta etapa.

## Arquivos criados ou atualizados

- `src/data/prepare_dataset.py`: script de preparacao do manifesto, splits estratificados e estatisticas de normalizacao.
- `src/dataset.py`: helpers para carregar CSVs, montar subsets de `torchvision.datasets.ImageFolder` e construir `DataLoader`s.
- `src/transforms.py`: factories de transforms de treino, validacao, teste e inferencia.
- `src/__init__.py` e `src/data/__init__.py`: inicializacao dos pacotes Python.
- `data/processed/class_to_idx.json`
- `data/processed/dataset_manifest.csv`
- `data/processed/split_config.json`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/train_stats.json`
- `reports/dataset_pipeline_implementation.md`

## Estrategia implementada

O pipeline usa o diretorio `EuroSAT_RGB` como fonte unica das imagens. Nenhuma imagem foi copiada para `data/processed/`.

Os arquivos em `data/processed/` armazenam apenas metadados:

- manifesto com uma linha por imagem;
- splits estratificados;
- mapeamento de classes;
- configuracao de split;
- media e desvio padrao RGB calculados somente no treino.

O split usa:

- treino: 70%;
- validacao: 15%;
- teste: 15%;
- seed: 42;
- ordenacao deterministica de caminhos antes do embaralhamento;
- estratificacao por classe.

## Classes

`class_to_idx.json` contem 10 classes na mesma ordem deterministica esperada pelo `ImageFolder`:

```text
AnnualCrop: 0
Forest: 1
HerbaceousVegetation: 2
Highway: 3
Industrial: 4
Pasture: 5
PermanentCrop: 6
Residential: 7
River: 8
SeaLake: 9
```

## Contagens geradas

`dataset_manifest.csv` contem 27.000 imagens, todas com modo `RGB` e tamanho `64x64`.

| Split | Imagens |
|---|---:|
| Treino | 18.900 |
| Validacao | 4.050 |
| Teste | 4.050 |
| **Total** | **27.000** |

Contagens por classe:

| Classe | Treino | Validacao | Teste |
|---|---:|---:|---:|
| AnnualCrop | 2.100 | 450 | 450 |
| Forest | 2.100 | 450 | 450 |
| HerbaceousVegetation | 2.100 | 450 | 450 |
| Highway | 1.750 | 375 | 375 |
| Industrial | 1.750 | 375 | 375 |
| Pasture | 1.400 | 300 | 300 |
| PermanentCrop | 1.750 | 375 | 375 |
| Residential | 2.100 | 450 | 450 |
| River | 1.750 | 375 | 375 |
| SeaLake | 2.100 | 450 | 450 |

## Normalizacao

As estatisticas foram calculadas somente a partir das 18.900 imagens do split de treino.

```json
{
  "mean": [0.3438271126, 0.3799947054, 0.407612037],
  "std": [0.2024812323, 0.1369397151, 0.1156126249],
  "source_split": "train",
  "num_images": 18900,
  "num_pixels": 77414400,
  "input_size": [64, 64],
  "resized_images_for_stats": 0
}
```

## Transforms

Treino:

1. `Resize((64, 64))`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomVerticalFlip(p=0.5)`
4. `RandomRotation(degrees=15)`
5. `RandomApply([ColorJitter(...)], p=0.5)`
6. `ToTensor()`
7. `Normalize(mean=train_mean, std=train_std)`

Validacao, teste e inferencia:

1. `Resize((64, 64))`
2. `ToTensor()`
3. `Normalize(mean=train_mean, std=train_std)`

## Validacoes executadas

Comandos executados:

```bash
python3 -m src.data.prepare_dataset
python3 -m compileall src
python3 -m src.data.prepare_dataset --output-dir /tmp/eurosat_processed_check
diff -q data/processed/dataset_manifest.csv /tmp/eurosat_processed_check/dataset_manifest.csv
diff -q data/processed/train.csv /tmp/eurosat_processed_check/train.csv
diff -q data/processed/val.csv /tmp/eurosat_processed_check/val.csv
diff -q data/processed/test.csv /tmp/eurosat_processed_check/test.csv
diff -q data/processed/train_stats.json /tmp/eurosat_processed_check/train_stats.json
diff -q data/processed/class_to_idx.json /tmp/eurosat_processed_check/class_to_idx.json
```

Resultados:

- `data/processed/` foi criado com todos os metadados planejados.
- `class_to_idx.json` contem exatamente 10 classes.
- As classes batem com as pastas reais de `EuroSAT_RGB`.
- `dataset_manifest.csv` possui 27.000 linhas de imagens, mais cabecalho.
- Todos os caminhos relativos do manifesto existem no disco.
- `train.csv`, `val.csv` e `test.csv` somam 27.000 imagens.
- As contagens por split sao 18.900, 4.050 e 4.050.
- As contagens por classe seguem a tabela planejada.
- O script validou ausencia de sobreposicao entre treino, validacao e teste.
- Todas as imagens registradas no manifesto abriram com Pillow.
- Todas as imagens verificadas no manifesto sao `RGB` e `64x64`.
- A media e o desvio padrao foram calculados somente a partir do treino.
- A reexecucao com a mesma seed gerou CSVs, `class_to_idx.json` e `train_stats.json` identicos.
- A compilacao Python dos novos modulos passou.

## Validacao de DataLoader

A validacao real de batch com `torch.utils.data.DataLoader` foi implementada em `src/dataset.py`, mas nao foi executada neste ambiente porque as dependencias PyTorch nao estao instaladas.

Tentativa executada:

```bash
python3 - <<'PY'
from src.dataset import build_dataloaders, validate_dataloader_batch
loaders = build_dataloaders(num_workers=0)
for split, loader in loaders.items():
    print(split, validate_dataloader_batch(loader))
PY
```

Resultado:

```text
ModuleNotFoundError: torch is required for device detection and DataLoaders. Install project dependencies before using src.dataset.
```

Quando `torch` e `torchvision` estiverem instalados, o mesmo comando deve validar:

- imagens como tensor `float32` no formato `[batch, 3, 64, 64]`;
- labels no intervalo `[0, 9]`;
- `shuffle=True` apenas no treino;
- `shuffle=False` em validacao e teste;
- `pin_memory=True` apenas quando o device detectado for CUDA.

## Status

Pipeline de metadados, split, normalizacao, transforms e helpers PyTorch implementado.

Unico item nao validado em runtime: batch real de `DataLoader`, bloqueado por ausencia local de `torch` e `torchvision`.
