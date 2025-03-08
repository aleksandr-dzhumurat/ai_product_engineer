import dlt
from dlt.sources.filesystem import filesystem, read_parquet

scenario = 'userdata'

if scenario == 'userdata':
    filesystem_resource = filesystem(
    bucket_url="./data",
    file_glob="**/*.parquet"
    )
    filesystem_pipe = filesystem_resource | read_parquet()

    # We load the data into the table_name table
    pipeline = dlt.pipeline(pipeline_name="my_new_pipeline", destination="duckdb")
    load_info = pipeline.run(filesystem_pipe.with_name("userdata"))
    print(load_info)

    print('Cols ', pipeline.dataset(dataset_type="default").userdata.df().shape[1])
else:
    import os

    os.environ["BUCKET_URL"] = "./data"
    import dlt
    from dlt.sources.sql_database import sql_database

    source = sql_database(
        "mysql+pymysql://rfamro:@mysql-rfam-public.ebi.ac.uk:4497/Rfam",
        table_names=["family",]
    )
    pipeline = dlt.pipeline(
        pipeline_name='fs_pipeline',
        destination='filesystem', # <--- change destination to 'filesystem'
        dataset_name='fs_data',
    )

    # load_info = pipeline.run(source, loader_file_format="parquet") # <--- choose a file format: parquet, csv or jsonl
    # print(load_info)
    print(pipeline.dataset(dataset_type="default").family.df())
