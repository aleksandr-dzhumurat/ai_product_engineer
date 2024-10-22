import dlt
from dlt.sources.helpers import requests
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.auth import BearerTokenAuth
from dlt.sources.helpers.rest_client.paginators import HeaderLinkPaginator

import os

os.environ["SOURCES__SECRET_KEY"] = os.environ['GITHUB_API_TOKEN']


@dlt.source
def github_source(secret_key=dlt.secrets.value):
    client = RESTClient(
            base_url="https://api.github.com",
            auth=BearerTokenAuth(token=secret_key),
            paginator=HeaderLinkPaginator(),
    )

    @dlt.resource
    def github_pulls(cursor_date=dlt.sources.incremental("updated_at", initial_value="2024-12-01")):
        params = {
            "since": cursor_date.last_value,
            "status": "open"
        }
        for page in client.paginate("repos/dlt-hub/dlt/pulls", params=params):
            yield page


    return github_pulls


# define new dlt pipeline
pipeline = dlt.pipeline(
    pipeline_name="github_pipeline3",
    destination="duckdb",
    dataset_name="github_data",
    export_schema_path="schemas/export",
    import_schema_path="schemas/import",
)


# run the pipeline with the new resource
load_info = pipeline.run(github_source())
print(load_info)
