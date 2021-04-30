from azureml.core import Run
import argparse
import os

run = Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument('--raw-ds', type = str, dest = 'raw_dataset_id')
parser.add_argument('--out_folder', type = str, dest = 'folder')

args = parser.parse_args()

output_folder = args.folder

raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()


prepped_df = raw_df.iloc[:,:3]

#savinf the df
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'preppad_data.csv')
prepped_df.to_csv(output_path)
