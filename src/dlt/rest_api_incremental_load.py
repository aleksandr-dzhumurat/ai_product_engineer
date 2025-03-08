import os

access_token = os.environ['GITHUB_API_TOKEN']


import dlt
from dlt.sources.helpers import requests
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.auth import BearerTokenAuth
from dlt.sources.helpers.rest_client.paginators import HeaderLinkPaginator

@dlt.source
def github_source(access_token=access_token): # <--- set the secret variable "access_token" here
    client = RESTClient(
            base_url="https://api.github.com",
            auth=BearerTokenAuth(token=access_token),
            paginator=HeaderLinkPaginator(),
    )
    
    @dlt.resource(
        name="pulls_comments",
        write_disposition="merge",
        primary_key="id"
    )
    def pulls_comments(cursor_date=dlt.sources.incremental("updated_at", initial_value="2024-12-01")):
        params = {
            "since": cursor_date.last_value,  # <--- use last_value to request only new data from API
            "status": "open"
        }
        for page in client.paginate("repos/dlt-hub/dlt/pulls/comments", params=params):
            yield page

    return pulls_comments

if __name__ == '__main__':
    pipeline = dlt.pipeline(destination="duckdb")
    load_info = pipeline.run(github_source())
    print(load_info)
    with pipeline.sql_client() as client:
        with client.execute_query("SELECT table_name, * FROM information_schema.tables") as table:
            print(table.df())
    with pipeline.sql_client() as client:
        with client.execute_query("SELECT * FROM pulls_comments") as table:
            df = table.df()
            print('Shape: ', df.shape)
            print(df.head())