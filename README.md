# Trabalho de Conclusão do Curso  - UFRGS 

Este espaço estão os códigos, notebooks e demais artefatos utilizados no Trabalho de Conclusão para o curso de Especialização em Engenheria de Software para Aplicações de Ciência de Dados, UFRGS, 2024-2026.

Aluno: Luiz Antônio Marques Garcia
Orientador: Joel Luís Carbonera

## Estrutura do Projeto

```bash
├── README.md                  # documentação do projeto
├── run_exp.py                 # executa um experimento 
├── config_pipeline.json       # configurações principais do pipeline
├── run_training_serving.py    # run mlflow e codigo de treinamento e serving 
├── run_simulation_drift.py    # run simula inferencia, monitor de drift e trigger
├── stop_mlflow.py             # codigo auxiliar para parar mlflow se preciso
├── mlflow.db                  # base dados do mlflow criada na execução do mlflow

├── data                       
│   ├── raw                    # dataset principal do modelo
│   │   └── credit_data.csv
│   └── inferences             # dataset com inferências simuladas
│       └── credit_data_inferences_log.csv
 
├── src
    ├── experiments
    │   ├── credit_model_experiments.py # experimentos de treinamento dos models
    │   └── credit_model_promote.py     # promove modelo campeão a produção
	├── serve
    │   └── credit_model_serve.py       # sobe serviço de api com modelo campeão 
	├── monitor
    │   └── monitor_drift.py            # monitora drifts e salva em \reports
    ├── simulation
    │   └── simulation.py               # cria dataset de inferencias simuladas
    └── triggers
        └── retraining_trigger.py       # verifica \reports e aciona retreinamento

├── reports                     # pasta com reports de drift do evidently
│   ├── report_classdrift.html
│   ├── report_classdrift.json  # class drifts em json para a trigger
│   ├── report_datadrift.html
│   └── report_datadrift.json   # data drifts em json para a trigger

├── runs   
│   └── runs.log                # logs dos pythons executados no pipeline (\src)
        
├── mlruns/                    # runs do mlflow, criado após inicialização
```

## Executar apenas um experimento

python run\_exp.py .\\configs\\adult\_\_bernoulli\_nb\_\_pre-none\_\_in-none\_\_post-none.yaml


## \###Parametros dos experimentos gerais

params.yaml

## 

## \### Gerar os arquivos individuais de experimentos a partir dos param.yaml

python gera\_configs.py # serão salvos em \\configs\\

## 

## \###Rodas todos os epxerimentos em \\configs\\

pyhton run\_all.py



## \###rodar várias execuções de run\_all

lote\_run\_all.bat

# 





## \###No warnings

set TF\_ENABLE\_ONEDNN\_OPTS=0

