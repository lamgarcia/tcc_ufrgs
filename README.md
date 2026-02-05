# Trabalho de Conclusão do Curso  - UFRGS 

Este espaço estão os códigos, notebooks e demais artefatos utilizados no Trabalho de Conclusão para o curso de Especialização em Engenheria de Software para Aplicações de Ciência de Dados, UFRGS, 2024-2026.

Aluno: Luiz Antônio Marques Garcia
Orientador: Joel Luís Carbonera

## 📁 Estrutura do Projeto

```bash
├── README.md                  # documentação do projeto
├── run_exp.py                 # executa um experimento 
├── params.yaml                # parâmetros para geração arquivos individuais de experimentos
├── gera\_configs.py           # cria vários arquivos de parâmetros em \configs a partir da leitura de params.yaml
├── run\_all.py                # executa os experimentos configurados nos .yaml em \configs
├── lote_run_all.bat           # executa vários run_all.py
├── configs                    # pasta com arquivos de parâmetros de cada experimento (.yaml)
├── datasets
│   ├── adult_sklearn          # pasta com dataset principal, Adult Income do ScitLearn.
│   │   └── adult_sklearn.csv         # dataset principal sem split
│   │   └── adult_sklearn_test.csv    # dataset split teste  
│   │   └── adult_sklearn_train.csv   # dataset split treintamento
│   │   └── adult_sklearn_val.csv     # dataset split validação
│   ├── adult_uci         # datasets Adult Income da fonte UCI (não foi utilizado nas análises)
│   ├── german_sklearn    # datasets com German Score do Scikit Learn (não foi utilizado nas análises)
├── src
    ├── datasets                        # código de download dos datasets
    │   ├── download_adult_sklearn.py   # download e split do Adult Income do Scikit-learn
    │   └── download_adult_uci.py       # download do Adult Income do UCI
    │   └── download_german_sklearn.py  # download do German Score
	├── models                          # modelos utilziados
    │   └── bernoulli_nb.py
	│   └── decision_tree.py
	│   └── logistic_regression.py
	│   └── neural_network.py
	│   └── random_forest.py
	│   └── svm.py
	│   └── xgboost.py
	├── mitigation
        ├── pre                                # mitigadores pré-processamento
			└── disparate_impact_remover.py
			└── reweighing.py
        ├── in                                 # mitigadores in-processing
        ├── post	                           # mitigadores pós-processamento
			└── equalized_odds_postprocessing.py
			└── reject_option_classification.py
	├── metrics
    │   └── evaluate_fairness.py        # calcula métricas de fairness dos experimentos
    │   └── evaluate_performance.py     # calcula métricas de desempenho preditivo dos experimientos
    └── results                         # codigo para gerar gráficos que devem ser ajustado os box manualmente
├── TCC_experimentos                    # Saídas (runs) dos experimentos utilizados no TCC
├── TCC_imagens                         # Imagens utilizadas em Latex
├── TCC_notebooks                       # Noteboks Jupyter (Google Colab) para criação dos gráficos do TCC
```

## Executar apenas um experimento
```bash
python run_exp.py .\\configs\\adult__bernoulli_nb__pre-none__in-none__post-none.yaml
```
Se não passar o arquivo .yaml com os parâmetros, irá buscar os parâmetros de config.yaml.
A execução será salva em runs_adult.csv.


## Executar vários experimentos

### Gerar os arquivos individuais de experimentos a partir dos params.yaml
```bash
python gera\_configs.py
```
Os arquivos de parâmetros .yaml de cada experimento serão salvos em \configs.

### Executar todos os epxerimentos criados em \\configs\\

```bash
pyhton run\_all.py
```
As execuções serão salvas em runs_adult.csv.

## Rodar lotes de execução de vários experimentos

lote\_run\_all.bat
As execuções serão salvas em runs_adult.csv.






## \###No warnings

set TF\_ENABLE\_ONEDNN\_OPTS=0

