import glob
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import softmax
import fire
import logging


def load_weights_from_folder(folder_path, seed, t):
    """
    Load weights array from a .pkl file in a folder for a specific iteration.

    args:
    - folder_path (str): Path to the directory containing .pkl files.
    - seed (int): Seed value used in the file naming.
    - t (int): Specific iteration number to load.

    returns:
    - Tuple: Contains iteration number and corresponding weights array.
    """
    print(folder_path)
    if seed is None:
        # match the first file in the directory
        loss_file_name = glob.glob(folder_path + f"test_seed_*_checkpoint-{t}.pkl")[0]
    else:
        loss_file_name = f"test_seed_{seed}_checkpoint-{t}.pkl"

    with open(loss_file_name, "rb") as file:
        loss = pickle.load(file)["task_loss"].to_numpy()

    return loss


def load_weight_diff_from_folder(folder_path, seed, start_t, end_t):
    """
    Load weights array from a .pkl file in a folder for a specific iteration.
    Relevant for fitting dynamic mixing laws.

    args:
    - folder_path (str): Path to the directory containing .pkl files.
    - seed (int): Seed value used in the file naming.
    - t (int): Specific iteration number to load.

    returns:
    - Tuple: Contains iteration number and corresponding weights array.
    """
    if seed is None:
        # match the first file in the directory
        start_loss_file_name = glob.glob(
            folder_path + f"val_seed_*_checkpoint-{start_t}.pkl"
        )[0]
        end_loss_file_name = glob.glob(
            folder_path + f"val_seed_*_checkpoint-{end_t}.pkl"
        )[0]
    else:
        start_loss_file_name = folder_path + f"val_seed_{seed}_checkpoint-{start_t}.pkl"
        end_loss_file_name = folder_path + f"val_seed_{seed}_checkpoint-{end_t}.pkl"

    with open(start_loss_file_name, "rb") as file:
        start_loss = pickle.load(file)["task_loss"].to_numpy()

    with open(end_loss_file_name, "rb") as file:
        end_loss = pickle.load(file)["task_loss"].to_numpy()

    return end_loss - start_loss


def collate(dir, seed, max_t, step, save=False):
    # ts, losses = load_weights_from_folder(dir, seed, max_t, step)
    ts = []
    losses = []
    for t in range(0, max_t, step):
        loss = load_weights_from_folder(dir, seed, t)
        ts.append([t, seed])
        losses.append(loss)
    ts = np.array(ts)
    losses = np.array(losses)

    data = np.hstack((ts, losses))

    df = pd.DataFrame(
        data, columns=["t", "seed"] + [f"l{i}" for i in range(losses.shape[1])]
    )

    if save:
        df.to_csv(dir + "collated.csv")
    return df


def diagonal_model(x, *params):
    # Assuming a, b, m are vectors
    n_params = len(params) // 3  # Number of elements in each vector (a, b, m)

    a = np.array(params[:n_params])
    b = np.array(params[n_params : 2 * n_params])
    m = np.array(params[2 * n_params :])

    result = a * np.exp(x * m) + b

    # Ensure the result is a 1D array of floats
    return result.flatten()


def model(x, *params):
    """
    The mixing law assumed by Ye et al. (2024) "Data Mixing Laws."
    """
    n_params = x.shape[-1]

    a = np.array(params[:n_params])
    b = np.array(params[n_params : 2 * n_params])
    m = np.array(params[2 * n_params :]).reshape(n_params, -1)

    result = a * np.exp(x @ m) + b

    return result.flatten()


def linear_model(x, *params):
    n_params = x.shape[-1]
    a = np.array(params).reshape(n_params, -1)

    result = x @ a

    return result.flatten()


def fit_model(x, y, model, initial_guess):
    # Use curve_fit to find the best-fitting parameters
    popt, pcov = curve_fit(model, x, y.flatten(), p0=initial_guess, maxfev=100000)

    # Calculate the predicted values using the optimized parameters
    y_pred = model(x, *popt)

    # Calculate the mean squared error
    mse = np.mean((y.flatten() - y_pred) ** 2)

    return popt, pcov, mse, y_pred, y


def linreg(run_dir, proportions_file, min_t=0, max_t=5000, loss_diff=True):
    np.set_printoptions(precision=3, suppress=True)
    logging.getLogger().setLevel(logging.INFO)

    # read lines in proportions file
    with open(proportions_file, "r") as f:
        lines = f.readlines()
        # remove commas
        lines = [line.strip().split(",") for line in lines]

    xs = []
    ys = []

    for proportions in lines:
        proportion_str = "".join(proportions)
        proportions = np.array([float(p) for p in proportions])
        dir_regex = (
            run_dir
            + "slimpj_pythia-160m_from_scratch_40000_mixture_arxiv_book_c4_cc_github_stackexchange_wikipedia_weights_"
            + proportion_str[:6]
            + "*/"
        )
        files = glob.glob(dir_regex)
        if len(files) == 0:
            logging.info(f"Directory {dir_regex} not found.")
            continue
        current_run_dir = files[0]

        if loss_diff:
            loss = load_weight_diff_from_folder(
                current_run_dir, min_t, max_t, seed=None
            )
        else:
            loss = load_weights_from_folder(current_run_dir, None, max_t)
        xs.append(proportions)
        ys.append(loss)

    xs = np.vstack(xs)
    ys = np.vstack(ys)

    # get min index of ys
    min_index = np.argmin(ys.mean(axis=1))
    lowest_played_sample = xs[min_index]
    print(
        f"Lowest played sample: {lowest_played_sample}. Average loss: {ys.mean(axis=1)[min_index]}"
    )

    n_params = xs.shape[1]
    # log_linear
    popt, pcov, mse, y_pred, y = fit_model(
        xs, ys, model, ([1] * n_params + [1] * n_params + [-1] * n_params**2)
    )

    # uncomment below to try other models
    # diagonal
    # popt, pcov, mse, y_pred, y = fit_model(xs, ys, diagonal_model, [1] * n_params + [1] * n_params + [-1] * n_params)
    # linear
    # popt, pcov, mse, y_pred, y = fit_model(xs, ys, linear_model, [1] * n_params ** 2)

    print("MSE: ", mse)
    # r^2
    ss_res = np.sum((y.flatten() - y_pred) ** 2)
    ss_tot = np.sum((y.flatten() - np.mean(y.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print("R^2: ", r2)

    # After fitting the model
    print(f"Actual minimum loss from samples: {np.min(ys.mean(axis=1))}")
    print(
        f"Predicted loss for sample with minimum actual loss: {np.mean(model(lowest_played_sample, *popt))}"
    )

    def objective(x):
        return np.average(model(x, *popt))

    def transform_params(y):
        # Transform unconstrained parameters to sum to 1
        return softmax(y)

    def objective_transformed(y):
        # Apply the transformation before calculating the objective
        x = transform_params(y)
        return objective(x)

    # Initial guess (unconstrained)
    y0 = np.array([0.0] * xs.shape[1])

    # Use Nelder-Mead method for optimization
    res = minimize(
        objective_transformed, y0, method="Nelder-Mead", options={"maxiter": 10000}
    )
    if res.success:
        logging.info("Optimization successful.")
        params = transform_params(res.x)
        print(params)
        # print predicted loss
        print(np.mean(model(params, *popt)))
    else:
        logging.info("Optimization failed.")
        logging.info("Message: %s", res.message)


ITOMETHOD = {
    0: "1000000",
    1: "0100000",
    2: "0010000",
    3: "0001000",
    4: "0000100",
    5: "0000010",
    6: "0000001",
}
SLICE_LIST = ["arxiv", "book", "cc", "c4", "github", "stackexchange", "wikipedia"]


def skills_graph(run_dir, seed, start=0, end=2857):
    dirstring = "slimpj_pythia-160m_from_scratch_{}_mixture_arxiv_book_c4_cc_github_stackexchange_wikipedia_weights_{}_static_lr_0.0005_linear_warmup_cosine"
    A = np.zeros((len(SLICE_LIST), len(SLICE_LIST)))

    for i, skill_i in enumerate(SLICE_LIST):
        thedir = run_dir + dirstring.format(end, ITOMETHOD[i])

        start_loss = load_weights_from_folder(thedir, seed, start)
        end_loss = load_weights_from_folder(thedir, seed, end)

        A[i] = (start_loss - end_loss) / start_loss

    print(A)
    np.save(run_dir + f"skills_graph_{seed}.npy", A)


if __name__ == "__main__":
    """
    linreg fits a linear regression to the data mixes, according to Ye et al. (2024) "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance."
    skills_graph computes the skill-it skills graph after performing all the independent runs.
    """
    fire.Fire({"linreg": linreg, "skills_graph": skills_graph})
