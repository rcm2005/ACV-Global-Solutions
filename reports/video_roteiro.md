# Roteiro do vídeo final

Tempo alvo: até 3 minutos.

## 0:00-0:20 - Abertura e problema

Olá, eu vou apresentar a solução de visão computacional da Global Solution 2026 da FIAP, no tema Indústria Espacial.

O problema é classificar uso e cobertura do solo em imagens RGB de sensoriamento remoto. Cada entrada tem `64x64` pixels e deve ser classificada em uma das 10 categorias do `EuroSAT_RGB`, como agricultura, floresta, rodovia, indústria, área residencial, rio ou mar e lago.

## 0:20-0:45 - Conexão espacial e ODS

A conexão com a Indústria Espacial está na observação da Terra: imagens orbitais ou aéreas podem ser recortadas e classificadas automaticamente para apoiar monitoramento ambiental, agricultura, infraestrutura e planejamento urbano.

Os ODS principais são o ODS 13, Ação Climática, e o ODS 15, Vida Terrestre. Também há relação com o ODS 9, por usar inteligência artificial e sensoriamento remoto como apoio à análise territorial.

## 0:45-1:15 - Dataset e preparação

O dataset usado foi o `EuroSAT_RGB`: 27.000 imagens `.jpg`, RGB, `64x64`, organizadas em 10 pastas de classe.

Não havia split pronto, nem notebooks, scripts, modelos ou pesos dentro do dataset. Criamos um split estratificado com seed 42: 18.900 imagens de treino, 4.050 de validação e 4.050 de teste.

A normalização foi calculada somente no treino, e o treinamento usou data augmentation moderado, com flips, rotação e variação leve de cor.

## 1:15-1:50 - CNNs treinadas do zero

Foram implementadas duas CNNs próprias em PyTorch, sem modelos pré-treinados e sem pesos externos.

A `BaselineCNN` é a referência simples: três blocos `Conv2d`, `ReLU` e `MaxPool2d`, mais camadas densas com dropout. Ela tem 620.362 parâmetros.

A `ImprovedCNN` é mais robusta: quatro estágios, duas convoluções por estágio, `BatchNorm2d`, `Dropout2d` e pooling adaptativo global. Ela tem 1.207.402 parâmetros.

## 1:50-2:25 - Resultados e comparação

Na avaliação final, a `BaselineCNN` atingiu 82,42% de accuracy no teste, abaixo da meta de 88%.

A `ImprovedCNN` foi o melhor modelo, com test loss de 0,2400 e test accuracy de 91,53%, acima da meta.

Aqui eu mostro as curvas de accuracy e loss, a matriz de confusão e a comparação. A principal confusão restante foi entre `PermanentCrop` e `HerbaceousVegetation`, coerente com a semelhança visual entre áreas agrícolas e vegetação em imagens RGB pequenas.

## 2:25-2:50 - Demonstração no notebook

No notebook `notebooks/04_model_evaluation_demo.ipynb`, a demonstração carrega `models/best_model.pt` e `models/class_names.json`, recria a arquitetura salva e aplica o mesmo pré-processamento do teste.

O notebook não retreina. Ele executa predição em imagem nova ou amostra de teste, mostra classe prevista, confiança, top-3 probabilidades, acertos e erros.

## 2:50-3:00 - Fechamento

Como conclusão, o projeto entrega uma solução reproduzível com CNNs treinadas do zero. O melhor modelo atingiu 91,53% no teste e pode apoiar observação da Terra, monitoramento ambiental e análise territorial.

## Checklist visual para gravação

- Mostrar o README na seção de problema e conexão espacial.
- Mostrar a tabela de classes e quantidade de imagens.
- Mostrar os arquivos `src/models/baseline_cnn.py` e `src/models/improved_cnn.py`.
- Mostrar a tabela de resultados no README ou em `reports/experiment_report.md`.
- Mostrar `reports/confusion_matrices/improved_confusion_matrix.png`.
- Mostrar `reports/predictions/correct_predictions.png` e `reports/predictions/wrong_predictions.png`.
- Executar ou navegar pelo notebook `notebooks/04_model_evaluation_demo.ipynb`.
