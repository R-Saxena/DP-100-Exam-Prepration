from argparse import ArgumentParser as AP
args = AP()
args.add_argument("--n_estimator", type = int)
args.add_argument("--min_sample_leafs", type = int)
args.add_argument("--input_dataset", type = str)

argument = args.parse_args()

print(argument.n_estimator)
print(argument.min_sample_leafs)
print(argument.input_data)