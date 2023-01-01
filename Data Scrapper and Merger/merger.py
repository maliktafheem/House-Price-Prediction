#append csv files 
import pandas as pd
import glob
import os

path = r'Data Scrapper and Final Data' # use path for data

all_files = glob.glob(os.path.join(path, "*.csv"))

df_from_each_file = (pd.read_csv(f) for f in all_files)
merger_df   = pd.concat(df_from_each_file, ignore_index=True)
merger_df.drop_duplicates(subset ="Property_ID", keep = 'first', inplace = True)
merger_df.to_csv('./final_data.csv', index=False)