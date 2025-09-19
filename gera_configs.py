import os
import yaml

# === Caminhos das pastas ===
OUTPUT_DIR = "configs_automaticas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Custom Dumper para forçar aspas em strings e listas inline ===
class QuotedDumper(yaml.SafeDumper):
    pass

def represent_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

def represent_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

QuotedDumper.add_representer(str, represent_str)
QuotedDumper.add_representer(list, represent_list)

# === Ler params.yaml ===
with open("params.yaml", "r", encoding="utf-8") as f:
    params_yaml = yaml.safe_load(f)

dataset_cfg = params_yaml.get("dataset", {})

# Extrair modelos e mitigações como listas de dicionários {name, params}
models_cfg = []
mitigations_cfg = {"pre": [], "in": [], "post": []}

# Modelos
if "model" in params_yaml:
    # Aceita tanto lista quanto múltiplos blocos repetidos
    if isinstance(params_yaml["model"], list):
        models_cfg = params_yaml["model"]
    else:
        # Caso "params.yaml" esteja no formato repetido (sem lista explícita)
        # converte manualmente: cada item é dict com "name"/"params"
        temp = params_yaml["model"]
        if isinstance(temp, dict) and "name" in temp:
            models_cfg.append(temp)
        elif isinstance(temp, dict):
            # fallback: varre por chaves
            for k, v in temp.items():
                if isinstance(v, dict) and "name" in v:
                    models_cfg.append(v)

# Mitigations
for stage in ["pre", "in", "post"]:
    if stage in params_yaml.get("mitigation", {}):
        items = params_yaml["mitigation"][stage]
        if isinstance(items, list):
            mitigations_cfg[stage] = items
        else:
            # Mesmo truque para blocos repetidos
            if isinstance(items, dict) and "name" in items:
                mitigations_cfg[stage].append(items)
            elif isinstance(items, dict):
                for k, v in items.items():
                    if isinstance(v, dict) and "name" in v:
                        mitigations_cfg[stage].append(v)

print("Modelos carregados:", [m["name"] for m in models_cfg])
for stage in ["pre", "in", "post"]:
    print(f"{stage} methods:", [m["name"] for m in mitigations_cfg[stage]])

# === Gerar combinações ===
count = 0
for model in models_cfg:
    for pre in mitigations_cfg["pre"]:
        for in_ in mitigations_cfg["in"]:
            for post in mitigations_cfg["post"]:
                config = {
                    "dataset": dataset_cfg,
                    "model": {
                        "name": model["name"],
                        "params": model.get("params", {})
                    },
                    "mitigation": {
                        "pre": {"name": pre["name"], "params": pre.get("params", {})},
                        "in": {"name": in_["name"], "params": in_.get("params", {})},
                        "post": {"name": post["name"], "params": post.get("params", {})}
                    }
                }

                # Nome do arquivo
                filename = f"{model['name']}__pre-{pre['name']}__in-{in_['name']}__post-{post['name']}.yaml"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Salvar YAML
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, sort_keys=False, Dumper=QuotedDumper, allow_unicode=True)

                count += 1

print(f"✅ {count} arquivos YAML gerados em '{OUTPUT_DIR}'")
