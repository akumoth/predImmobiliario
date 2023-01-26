import dask.dataframe as dd
train = dd.read_parquet('../data/raw/train.parquet')
test = dd.read_parquet('../data/raw/test.parquet')
train.repartition(4).to_parquet('../data/raw/train',name_function=lambda x: f"train-{x}.parquet")
test.repartition(2).to_parquet('../data/raw/test',name_function=lambda x: f"test-{x}.parquet")