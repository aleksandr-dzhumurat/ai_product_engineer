import dlt
from dlt.sources.sql_database import sql_database

source = sql_database(
    "mysql+pymysql://rfamro:@mysql-rfam-public.ebi.ac.uk:4497/Rfam",
    table_names=["family",]
)

pipeline = dlt.pipeline(
    pipeline_name="sql_database_example",
    destination="duckdb",
    dataset_name="sql_data",
    dev_mode=True,
)

# load_info = pipeline.run(source)
# print(load_info)

with pipeline.sql_client() as client:
    with client.execute_query("SELECT table_name, * FROM information_schema.tables") as table:
        print(table.df())
with pipeline.sql_client() as client:
    with client.execute_query("SELECT * FROM sql_data_20250111072201.family") as table:
        print('Family ', table.df().shape[1])