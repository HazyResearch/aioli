import os 
import numpy as np
import stat

seed = 0 
n=50

file = f"./dynamic/run_child.sh"

weight_file = f"../../../dirichlet_weights/k_9_n_{n}_seed_{seed}.txt"

file_name = file.split("/")[-1].split(".")[0]

with open(weight_file, 'r') as f:
    all_weights = f.readlines()


for i, weights in enumerate(all_weights):

    weights = " ".join(weights.strip().split(","))

    with open(file, "r") as f:
        script = f.read()
    proportions = script.split("--proportions ")[-1].split("\\")[0].strip()
    new_script = script.replace(f"--proportions {proportions}", f"--proportions {weights}")


    for j, weights in enumerate(all_weights):

        if j >= 10 or j < 0:
            continue 

        weights_no_trailing_zeros = "".join([str(float(num)) for num in weights.split(",")])

        current_saved_model_proportions = new_script.split("1000_break_500_mixture_weights_")[-1].split("_static")[0]


        new_new_script = new_script.replace(f"1000_break_500_mixture_weights_{current_saved_model_proportions}_static",
                                        f"1000_break_500_mixture_weights_{weights_no_trailing_zeros}_static")

        new_new_script = new_new_script.replace("${SEED}", f"{seed}")
        new_file = f'{"/".join(file.split("/")[:-1])}/{file_name}_{i}_{j}_n_{n}_seed_{seed}.sh'


        print(new_new_script)
        with open(new_file, "w") as f:
            f.write(new_new_script)
        st = os.stat(new_file)
        os.chmod(new_file, st.st_mode | stat.S_IEXEC)
