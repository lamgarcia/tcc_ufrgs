import os
import glob
import subprocess
import sys
from datetime import datetime

def main():
    configs_dir = ".\configs"

    if not os.path.exists(configs_dir):
        print(f"Erro: Pasta '{configs_dir}' não encontrada.")
        sys.exit(1)

    yaml_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    if not yaml_files:
        print(f"No .yaml files found in '{configs_dir}'.")
        return

    print(f"Found {len(yaml_files)} YAML files.\n")


    for yaml_file in yaml_files:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{dt} - Run: python run_exp.py {yaml_file}")
        try:
            # Executa o comando
            result = subprocess.run(
                ["python", "run_exp.py", yaml_file],
                check=True,  
                capture_output=True,
                text=True
            )
            print(f"Sucess: {yaml_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error running {yaml_file}:")
            print(e.stderr)
        except Exception as e:
            print(f"Error  unexpected with  {yaml_file}: {e}")

        print("-" * 60)

if __name__ == "__main__":
    main()