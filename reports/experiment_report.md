# Relatorio de treinamento e avaliacao das CNNs

Data de execucao: 2026-06-01T14:23:40

## Configuracao

- Dataset: `EuroSAT_RGB`
- Split processado: `data/processed`
- Seed: 42
- Device: `cpu`
- Entrada: `64x64` RGB
- Batch size: 256
- Num workers: 0
- Funcao de loss: `CrossEntropyLoss`
- Otimizador: `AdamW`
- Meta de referencia no teste: 88%

## Classes e splits

| Split | Imagens |
|---|---:|
| train | 18900 |
| val | 4050 |
| test | 4050 |

Classes: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`

## Resultados finais

| Modelo | Parametros | Melhor epoca | Val acc | Test loss | Test acc | Meta 88% |
|---|---:|---:|---:|---:|---:|---|
| BaselineCNN | 620,362 | 8 | 0.8422 | 0.5041 | 0.8242 | nao atingida |
| ImprovedCNN | 1,207,402 | 11 | 0.9230 | 0.2400 | 0.9153 | atingida |

## Melhor modelo

O melhor modelo foi `ImprovedCNN`, com accuracy de teste 0.9153.
A meta de referencia de 88% foi atingida no conjunto de teste.

## Historico por epoca

### BaselineCNN

| Epoca | Train loss | Train acc | Val loss | Val acc | LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4179 | 0.4622 | 1.0200 | 0.6375 | 0.001000 |
| 2 | 0.9402 | 0.6591 | 0.8259 | 0.7069 | 0.001000 |
| 3 | 0.7991 | 0.7104 | 0.7104 | 0.7452 | 0.001000 |
| 4 | 0.7584 | 0.7286 | 0.6459 | 0.7696 | 0.001000 |
| 5 | 0.7006 | 0.7486 | 0.5979 | 0.7822 | 0.001000 |
| 6 | 0.6444 | 0.7697 | 0.6235 | 0.7694 | 0.001000 |
| 7 | 0.6527 | 0.7708 | 0.4925 | 0.8331 | 0.001000 |
| 8 | 0.5910 | 0.7899 | 0.4810 | 0.8422 | 0.001000 |

### ImprovedCNN

| Epoca | Train loss | Train acc | Val loss | Val acc | LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.2389 | 0.5637 | 0.7507 | 0.7190 | 0.001000 |
| 2 | 0.8407 | 0.7078 | 0.5808 | 0.7879 | 0.001000 |
| 3 | 0.7401 | 0.7464 | 0.5177 | 0.8170 | 0.001000 |
| 4 | 0.6410 | 0.7834 | 0.4763 | 0.8304 | 0.001000 |
| 5 | 0.5886 | 0.8041 | 0.3758 | 0.8704 | 0.001000 |
| 6 | 0.5315 | 0.8238 | 0.3433 | 0.8842 | 0.001000 |
| 7 | 0.4901 | 0.8390 | 0.3453 | 0.8830 | 0.001000 |
| 8 | 0.4513 | 0.8513 | 0.2706 | 0.9091 | 0.001000 |
| 9 | 0.4151 | 0.8672 | 0.2409 | 0.9225 | 0.001000 |
| 10 | 0.3951 | 0.8729 | 0.2795 | 0.9074 | 0.001000 |
| 11 | 0.3673 | 0.8792 | 0.2192 | 0.9230 | 0.001000 |
| 12 | 0.3482 | 0.8839 | 0.2154 | 0.9202 | 0.001000 |

## Relatorio de classificacao

### BaselineCNN

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| AnnualCrop | 0.7759 | 0.9156 | 0.8400 | 450 |
| Forest | 0.9506 | 0.9400 | 0.9453 | 450 |
| HerbaceousVegetation | 0.8527 | 0.6044 | 0.7074 | 450 |
| Highway | 0.5652 | 0.6587 | 0.6084 | 375 |
| Industrial | 0.9105 | 0.9493 | 0.9295 | 375 |
| Pasture | 0.8270 | 0.7967 | 0.8115 | 300 |
| PermanentCrop | 0.7399 | 0.6373 | 0.6848 | 375 |
| Residential | 0.9113 | 0.9822 | 0.9455 | 450 |
| River | 0.7241 | 0.7840 | 0.7529 | 375 |
| SeaLake | 0.9764 | 0.9200 | 0.9474 | 450 |

### ImprovedCNN

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| AnnualCrop | 0.9460 | 0.8956 | 0.9201 | 450 |
| Forest | 0.9669 | 0.9733 | 0.9701 | 450 |
| HerbaceousVegetation | 0.7379 | 0.9511 | 0.8311 | 450 |
| Highway | 0.9251 | 0.9227 | 0.9239 | 375 |
| Industrial | 0.9354 | 0.9653 | 0.9501 | 375 |
| Pasture | 0.8765 | 0.9467 | 0.9103 | 300 |
| PermanentCrop | 0.9247 | 0.6880 | 0.7890 | 375 |
| Residential | 0.9842 | 0.9711 | 0.9776 | 450 |
| River | 0.9533 | 0.8160 | 0.8793 | 375 |
| SeaLake | 0.9632 | 0.9889 | 0.9759 | 450 |

## Comparacao tecnica

`BaselineCNN` usa tres blocos convolucionais simples com ReLU e MaxPool, seguido por classificador denso com dropout. Ela funciona como referencia mais compacta e menos regularizada.

`ImprovedCNN` aprofunda a extracao de caracteristicas com quatro estagios, duas convolucoes por estagio, BatchNorm2d, Dropout2d e pooling adaptativo. A maior capacidade e a normalizacao tendem a melhorar a generalizacao em classes visualmente parecidas, mas com maior custo computacional.

## Analise de erros

### BaselineCNN

| Classe real | Classe predita | Ocorrencias |
|---|---|---:|
| PermanentCrop | Highway | 57 |
| HerbaceousVegetation | Highway | 52 |
| River | Highway | 50 |
| HerbaceousVegetation | PermanentCrop | 48 |
| Highway | River | 41 |
| Highway | AnnualCrop | 31 |
| PermanentCrop | AnnualCrop | 28 |
| HerbaceousVegetation | River | 21 |

### ImprovedCNN

| Classe real | Classe predita | Ocorrencias |
|---|---|---:|
| PermanentCrop | HerbaceousVegetation | 100 |
| River | Highway | 19 |
| River | Pasture | 19 |
| Highway | Industrial | 14 |
| AnnualCrop | SeaLake | 13 |
| River | HerbaceousVegetation | 13 |
| AnnualCrop | PermanentCrop | 11 |
| AnnualCrop | Pasture | 10 |

Exemplos de acertos do melhor modelo:

- `EuroSAT_RGB/AnnualCrop/AnnualCrop_100.jpg`: real `AnnualCrop`, predito `AnnualCrop`, confianca 0.9910.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1005.jpg`: real `AnnualCrop`, predito `AnnualCrop`, confianca 0.9853.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1010.jpg`: real `AnnualCrop`, predito `AnnualCrop`, confianca 0.9505.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1021.jpg`: real `AnnualCrop`, predito `AnnualCrop`, confianca 0.9932.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1024.jpg`: real `AnnualCrop`, predito `AnnualCrop`, confianca 0.9078.

Exemplos de erros do melhor modelo:

- `EuroSAT_RGB/AnnualCrop/AnnualCrop_101.jpg`: real `AnnualCrop`, predito `Pasture`, confianca 0.6558.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1234.jpg`: real `AnnualCrop`, predito `PermanentCrop`, confianca 0.7386.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1266.jpg`: real `AnnualCrop`, predito `Highway`, confianca 0.5015.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1294.jpg`: real `AnnualCrop`, predito `River`, confianca 0.5968.
- `EuroSAT_RGB/AnnualCrop/AnnualCrop_1355.jpg`: real `AnnualCrop`, predito `PermanentCrop`, confianca 0.9209.

## Artefatos gerados

- `models/baseline_best.pt`
- `models/improved_best.pt`
- `models/best_model.pt`
- `models/class_names.json`
- `models/metrics.json`
- `reports/figures/baseline_accuracy_loss.png`
- `reports/figures/improved_accuracy_loss.png`
- `reports/confusion_matrices/baseline_confusion_matrix.png`
- `reports/confusion_matrices/improved_confusion_matrix.png`
- `reports/predictions/correct_predictions.png`
- `reports/predictions/wrong_predictions.png`
