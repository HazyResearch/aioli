import os 
import numpy as np
import stat


file = "./run.sh" 
# file = "./dynamic/run_parent.sh"
# file = "./run_child.sh"


n_grid_points = 10
weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

file_name = file.split("/")[-1].split(".")[0]

for i, weight in enumerate(weights):

    with open(file, "r") as f:
        script = f.read()

    proportions = script.split("--proportions ")[-1].split("\\")[0].strip()
    new_proportions = f"{weight} {n_grid_points-weight}"

    new_script = script.replace(f"--proportions {proportions}", f"--proportions {new_proportions}")


    new_file = f'{"/".join(file.split("/")[:-1])}/{file_name}_{weight}_{n_grid_points-weight}.sh'
    with open(new_file, "w") as f:
        f.write(new_script)
    st = os.stat(new_file)
    os.chmod(new_file, st.st_mode | stat.S_IEXEC)

