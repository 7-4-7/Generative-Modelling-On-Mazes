from .data import dataset_creator

# Create the dataset
dataset_creator.create_dataset(
    algorithms= [ 'wilson','r_prims','r_kruskals','dfs','hak'],
    n_explore = 10_000 # Each algorithm 10_000 examples
)
