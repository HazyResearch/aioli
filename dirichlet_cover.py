import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import fire

def cover_simplex(k, N, alphas, oversample_factor=1):
    # Calculate the number of oversampled points
    n_samples = N * (1 << oversample_factor)
    
    # Step 1: Oversample from Dirichlet distribution
    samples = np.random.dirichlet(alphas, n_samples)
    
    # Step 2: Hierarchically merge points
    while len(samples) > N:
        distances = squareform(pdist(samples))
        np.fill_diagonal(distances, np.inf)
        closest_pair = np.unravel_index(np.argmin(distances), distances.shape)
        midpoint = (samples[closest_pair[0]] + samples[closest_pair[1]]) / 2
        samples = np.delete(samples, closest_pair, axis=0)
        samples = np.vstack([samples, midpoint])
    
    return samples

def save_points_to_file(points, filename):
    np.savetxt(filename, points, delimiter=',', fmt='%.6f')

def main(k, N, alphas, output_file, oversample_factor=4, seed=0):
    '''
    if oversample == 1, then this is equivalent to sampling from the dirichlet distribution.
    alphas expects a list of k values with no spaces in between.
    '''
    assert len(alphas) == k, "Number of alphas must be equal to the dimension of the simplex"

    # Set seed
    np.random.seed(seed)

    # Convert alphas to array
    alphas = np.array(alphas)

    points = cover_simplex(k, N, alphas, oversample_factor=oversample_factor)
    save_points_to_file(points, output_file)
    print(f"Generated {N} points on a {k}-dimensional simplex with an oversampling factor of {oversample_factor}.")
    print(f"Points saved to {output_file}")


if __name__ == "__main__":
    seed = 4 # 0, 1, 2, 3
    k = 9
    n = 50
    main(k, n, [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], f"dirichlet_weights/k_{k}_n_{n}_seed_{seed}.txt", seed=seed)