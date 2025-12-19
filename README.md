# TCC - Ufrgs

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

## \###rodar apenas um experimento

python run\_exp.py .\\configs\\adult\_\_bernoulli\_nb\_\_pre-none\_\_in-none\_\_post-none.yaml



## \###No warnings

set TF\_ENABLE\_ONEDNN\_OPTS=0

