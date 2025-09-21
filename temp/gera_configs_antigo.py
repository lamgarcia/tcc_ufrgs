import os
import yaml

# === Caminhos das pastas ===
MODELS_DIR = "src/models"
PRE_DIR = "src/mitigation/pre"
IN_DIR = "src/mitigation/in"
POST_DIR = "src/mitigation/post"

OUTPUT_DIR = "configs_automaticas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Dataset fixo ===
dataset_cfg = {
    "name": "adult",
    "path": "datasets/adult_sklearn/adult_sklearn.csv",
    "path_train": "datasets/adult_sklearn/adult_sklearn_train.csv",
    "path_test": "datasets/adult_sklearn/adult_sklearn_test.csv",
    "target": "income",
    "sensitive": "sex",
    "privileged": ["Male"],       #1
    "unprivileged": ["Female"],   #0
    "favorable": ">50K",          #1
    "unfavorable": "<=50K"        #0
}

# === Custom Dumper para forçar aspas em strings e listas inline ===
class QuotedDumper(yaml.SafeDumper):
    pass

def represent_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

def represent_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

QuotedDumper.add_representer(str, represent_str)
QuotedDumper.add_representer(list, represent_list)

# === Função auxiliar para listar arquivos .py (ignora __init__.py) ===
def list_py_files(path):
    if not os.path.isdir(path):
        return []
    files = [
        os.path.splitext(f)[0]
        for f in os.listdir(path)
        if f.endswith(".py") and f != "__init__.py"
    ]
    return sorted(files)

# === Descobrir modelos e métodos de mitigação, sempre incluindo "none" para mitigações ===
models = list_py_files(MODELS_DIR)
if not models:
    models = ["none"]  # se não há modelos, usa 'none' como modelo

pre_methods = ["none"] + list_py_files(PRE_DIR)
in_methods = ["none"] + list_py_files(IN_DIR)
post_methods = ["none"] + list_py_files(POST_DIR)

# remover duplicatas possíveis (por precaução)
pre_methods = list(dict.fromkeys(pre_methods))
in_methods = list(dict.fromkeys(in_methods))
post_methods = list(dict.fromkeys(post_methods))

print("Modelos encontrados:", models)
print("Pre methods:", pre_methods)
print("In methods:", in_methods)
print("Post methods:", post_methods)

# === Gerar combinações ===
count = 0
for model in models:
    for pre in pre_methods:
        for in_ in in_methods:
            for post in post_methods:
                config = {
                    "dataset": dataset_cfg,
                    "model": {
                        "name": model,
                        "params": {}
                    },
                    "mitigation": {
                        "pre": {"name": pre, "params": {}},
                        "in": {"name": in_, "params": {}},
                        "post": {"name": post, "params": {}}
                    }
                }

                # Nome do arquivo
                filename = f"{model}__pre-{pre}__in-{in_}__post-{post}.yaml"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Salvar YAML com o Dumper customizado
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, sort_keys=False, Dumper=QuotedDumper, allow_unicode=True)

                count += 1

print(f"✅ {count} arquivos YAML gerados em '{OUTPUT_DIR}'")
