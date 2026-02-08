# Trabalho de Conclusão do Curso  - UFRGS 

Este espaço estão os códigos, notebooks e demais artefatos utilizados no Trabalho de Conclusão para o curso de Especialização em Engenheria de Software para Aplicações de Ciência de Dados, UFRGS, 2024-2026.

Aluno: Luiz Antônio Marques Garcia
Orientador: Joel Luís Carbonera

## 📁 Estrutura do Projeto

```bash
├── README.md                  # documentação do projeto
├── requirements.txt           # bibliotecas utilizadas no projeto
├── run_exp.py                 # executa um experimento 
├── params.yaml                # parâmetros para geração arquivos individuais de experimentos
├── gera_configs.py            # cria vários arquivos de parâmetros em \configs a partir da leitura de params.yaml
├── run_all.py                 # executa os experimentos configurados nos .yaml em \configs
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
    │   └── evaluate_fairness.py        # métricas de fairness dos experimentos
    │   └── evaluate_performance.py     # métricas de desempenho preditivo dos experimientos
    └── results                         # pasta com codigos auxiliares
 	│   └── evaluate_fairness.py
 	│   └── results_mean_std.py         # retorna CSV agregado por médio e devio padrão
├── TCC_experimentos                    # pasta com saídas (runs) dos experimentos utilizados no TCC
	├── dfs_10x                         # log dos datasets entra e saída dos mitigadores para conferências
	└── runs_adult_10x_1.csv            # Saída do lote de 10 execuções dos experimentos com parâmetros em \configs
	└── runs_adult_10x_1_mean_std.csv   # O runs_adult_10x_1.csv agregado por média e desvio padrão (resultado de (results_mean_std.py))
├── TCC_imagens                         # Imagens utilizadas em Latex
├── TCC_notebooks                       # Noteboks Jupyter (Google Colab) para criação dos gráficos do TCC
```

## Arquivo de parâmetro do experimento

 Para execução de um experimento _run_exp.py_ é preciso passar um arquivo de configuração no formato _.yaml_ no formato abaixo.
 Nele estão as informações utilizadas no experimento sobre.

```yaml
"dataset":
  "name": "adult"                                                                           # nome do dataset
  "path": "datasets/adult_sklearn/adult_sklearn.csv"                                        # dataset completo
  "path_train": "datasets/adult_sklearn/adult_sklearn_train.csv"                            # dataset com split de treino
  "path_val": "datasets/adult_sklearn/adult_sklearn_val.csv"                                # dataset com split de validação
  "path_test": "datasets/adult_sklearn/adult_sklearn_test.csv"                              # dataset  com split  de teste   
  "cols_exclude": ["fnlwgt"]                                                                # colunas que se deseja excluir
  "cols_cat": ["workclass", "education", "marital-status", "occupation", "relationship",    
    "race", "native-country"]     # colunas categóricas para one-hot enconding
  "target": "income"       		  # atributo alvo                                        
  "sensitive": "sex"              # atributo sensível
  "privileged": ["Male"]          # valor do atributo sensível que indica o grupo privilegiado
  "unprivileged": ["Female"]      # valor do atributo sensível que indica o grupo desprivilegiado
  "favorable": ">50K"             # valor do atributo alvo que é favorável 
  "unfavorable": "<=50K"          # valor do atributo alvo que não é favorável
"model":
  "name": "xgboost"                 # modelo utilizado, utilize o mesmo nome do arquivo .ý em \src\models    .
  "params":                         # hiperâmetros do modelo 
    "objective": "binary:logistic"
    "n_estimators": 200
    "max_depth": 6
    "learning_rate": 0.1
    "subsample": 0.8
    "colsample_bytree": 0.8
"mitigation":
  "pre":                                      # mitigação pré utilizada, mesmo nome do .py em \src\mitigation\pre
    "name": "reweighing"
    "params": {}
  "in":                                 	  # mitigação in utilizada, mesmo nome do arquivo em \src\mitigation\in  
    "name": "none"                            # none indica que não se quer utilizar mitigação naquela fase.
    "params": {}
  "post":
    "name": "equalized_odds_postprocessing"   # mitigação pós utilizada, mesmo nome do arquivo em \src\mitigation\post
    "params": {}
```

## Geração de vários arquivos de configuração de experimentos

O arquivo params.yaml tem formato similar a um arquivo de configuração de um experimento. Porém, ele aceita a inclusão de mais de modelos e mais mitigadores por fase.
Desta forma, o codigo _gera_config.py_ lê _params.yaml_ e realiza a combinação dos modelos e mitigadores gerando arquivos individuais de experimentos.
Os arquivos resultantes da combinação de modelos e mitigadores ficam armazenados em _\configs_.
Ao executar _run_all.py_ serão feitas execuções de run_exp.py para cada um dos arquivos de experimentos armazenados em _\configs_.

 
## Executar experimentos

### Executar apenas um experimento
```bash
python run_exp.py .\\configs\\adult__bernoulli_nb__pre-none__in-none__post-none.yaml
```
Se não passar o arquivo _.yaml_ com os parâmetros, irá buscar os parâmetros de _config.yaml_.
A execução será salva em _runs_adult.csv_.


### Executar vários experimentos

#### Gerar os arquivos individuais de experimentos a partir dos params.yaml
```bash
python gera_configs.py
```
Os arquivos de parâmetros _.yaml_ de cada experimento serão salvos em _\configs_.

#### Executar todos os epxerimentos criados em \\configs\\

```bash
pyhton run_all.py
```
As execuções serão salvas em _runs_adult.csv_.

#### Rodar lotes de execução de vários experimentos
```bash
lote_run_all.bat
```
As execuções serão salvas em _runs_adult.cs_v.


#### No warnings
set TF\_ENABLE\_ONEDNN\_OPTS=0

