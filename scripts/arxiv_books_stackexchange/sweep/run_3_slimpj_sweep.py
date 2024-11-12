import os 
import numpy as np
import stat

seed = 0 # replace with [1, 2, 3, 4]
n=9

file = "./run.sh"
# file = "./dynamic/run_parent.sh"

weight_file = f"../../../dirichlet_weights/k_3_n_{n}_seed_{seed}.txt"

file_name = file.split("/")[-1].split(".")[0]

with open(weight_file, 'r') as f:
    all_weights = f.readlines()


for i, weights in enumerate(all_weights):
    weights = " ".join(weights.strip().split(","))

    with open(file, "r") as f:
        script = f.read()
    proportions = script.split("--proportions ")[-1].split("\\")[0].strip()
    new_script = script.replace(f"--proportions {proportions}", f"--proportions {weights}")
    new_script = new_script.replace("${SEED}", f"{seed}")
    new_file = f'{"/".join(file.split("/")[:-1])}/{file_name}_{i}_seed_{seed}.sh'

    with open(new_file, "w") as f:
        f.write(new_script)
    st = os.stat(new_file)
    os.chmod(new_file, st.st_mode | stat.S_IEXEC)
