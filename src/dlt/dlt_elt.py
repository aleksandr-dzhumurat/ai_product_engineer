import os

access_token = os.environ['GITHUB_API_TOKEN']


import requests
import dlt
from dlt.sources.helpers import requests
import dlt
from dlt.sources.helpers import requests
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.auth import BearerTokenAuth


@dlt.source
def github_source(access_token=access_token): # <--- set the secret variable "access_token" here
    client = RESTClient(
            base_url="https://api.github.com",
            auth=BearerTokenAuth(token=access_token)
    )
    
    @dlt.resource
    def github_events():
        for page in client.paginate("orgs/dlt-hub/events"):
            yield page

    @dlt.resource(table_name="stargazers", write_disposition='replace')
    def github_stargazers():
        for page in client.paginate("repos/dlt-hub/dlt/stargazers"):
            yield page

    return github_events, github_stargazers

# Sample data containing pokemon details
data = [
    {"id": "1", "name": "bulbasaur", "size": {"weight": 6.9, "height": 0.7}},
    {"id": "4", "name": "charmander", "size": {"weight": 8.5, "height": 0.6}},
    {"id": "25", "name": "pikachu", "size": {"weight": 6, "height": 0.4}},
]

# Create a dlt resource from the data
@dlt.resource(table_name='pokemon_new') # <--- we set new table name
def my_dict_list():
    yield data

# Set pipeline name, destination, and dataset name
pipeline = dlt.pipeline(
    pipeline_name="quick_start",
    destination="duckdb",
    dataset_name="mydata",
)

def pagination(url):
    while True:
        response = requests.get(url)
        response.raise_for_status()
        yield response.json()
        # Get next page
        if "next" not in response.links:
            break
        url = response.links["next"]["url"]

@dlt.resource(table_name="issues", write_disposition='replace')
def get_issues():
    url = "https://api.github.com/repos/dlt-hub/dlt/issues?per_page=100"
    yield pagination(url)

@dlt.resource
def github_events():
    url = f"https://api.github.com/orgs/dlt-hub/events"
    response = requests.get(url)
    yield response.json()

@dlt.resource(table_name="repos", write_disposition='replace')
def github_repos():
    url = "https://api.github.com/orgs/dlt-hub/repos"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    yield response.json()
    # client = RESTClient(
    #     base_url="https://api.github.com",
    #     auth=BearerTokenAuth(token=access_token)
    # )
    # for page in client.paginate("orgs/dlt-hub/repos"):
    #     yield page

@dlt.resource(table_name="stargazers", write_disposition='replace')
def get_stargazers():
    url = "https://api.github.com/repos/dlt-hub/dlt/stargazers?per_page=50"
    yield pagination(url)

@dlt.source
def github_stargazer_data():
    return get_stargazers()

@dlt.transformer(data_from=github_repos, table_name='github_stargazer')
def github_stargazer(item):
    details = item
    print(details[0]['owner'], details['repo'])
    print()
    # for item in items:
    #   id = item["id"]
    #   url = f"https://pokeapi.co/api/v2/pokemon/{id}"
    #   response = requests.get(url)
    #   details = response.json()
    #   print(f"Details: {details}\n")
    yield details

if __name__=='__main__':
    print('hello')
    num_lesson = 4
    if num_lesson == 0:
        load_info = pipeline.run(data, table_name="pokemon")
        dataset = pipeline.dataset(dataset_type="default")
        result_df = dataset.pokemon.df()
        print('Cols ', result_df.shape[1])
        print(result_df.head())
    if num_lesson == -1:
        pipeline = dlt.pipeline(
            pipeline_name="github_pipeline",
            destination="duckdb",
            dataset_name="github_repos"
        )
        load_info = pipeline.run(github_repos)
        print(load_info)
        with pipeline.sql_client() as client:
            with client.execute_query("SELECT * FROM repos") as table:
                print('Cols: ', table.df().shape[1])
        df = pipeline.dataset(dataset_type="default").repos.df()
        cols = df.columns.tolist()
        print(len(cols), cols)
        print(df[df.columns.tolist()[:5]].head())
    elif num_lesson == 1:

        # Question 2
        pipeline = dlt.pipeline(
            pipeline_name="github_stargazer",
            destination="duckdb",
            dataset_name="all_data"
        )
        # load_info = pipeline.run(github_source())
        
        # with pipeline.sql_client() as client:
        #     with client.execute_query("SHOW ALL TABLES") as table:
        #         all_tables = table.df()["name"]
        #         print(all_tables)
            
        # print(load_info)
        with pipeline.sql_client() as client:
            with client.execute_query("SELECT * FROM stargazers") as table:
                df = table.df()
                print('Shape ', df.shape[1])
                print(df.head())


        # url = "https://api.github.com/orgs/dlt-hub/repos"
        # headers = {
        #     "Authorization": f"Bearer {access_token}"
        # }
        # response = requests.get(url, headers=headers)
        # import json
        # with open('/Users/username/PycharmProjects/ml_for_products/data/resp.json', 'w') as f:
        #     json.dump(response.json(), f, indent=4)
        # with open('/Users/username/PycharmProjects/ml_for_products/data/headers.json', 'w') as f:
        #     json.dump(dict(response.headers), f, indent=4)
        # print(response.next)
        # print(help(RESTClient))
        pass
    elif num_lesson == 2:
        duckdb_pipeline = dlt.pipeline(
            pipeline_name="github_data_duckdb",
            destination="duckdb"
        )
        load_info = duckdb_pipeline.run(github_data())
        print(load_info)

        with duckdb_pipeline.sql_client() as client:
            with client.execute_query("SHOW ALL TABLES") as table:
                all_tables = table.df()["name"]
                print(all_tables)
    elif num_lesson == 3:
        # Set pipeline name, destination, and dataset name
        # pipeline = dlt.pipeline(
        #     pipeline_name="github_pipeline",
        #     destination="duckdb",
        #     dataset_name="github_data"
        # )
        # load_info = pipeline.run(github_events())  # here is your code
        # print(load_info)
        # print(pipeline.dataset(dataset_type="default").github_events.df())

        # define new dlt pipeline
        pipeline = dlt.pipeline(destination="duckdb")
        load_info = pipeline.run(github_source())
        print(load_info)
        with pipeline.sql_client() as client:
            with client.execute_query("SELECT table_name FROM information_schema.tables") as table:
                print(table.df())
        with pipeline.sql_client() as client:
            with client.execute_query("SELECT * FROM github_stargazers WHERE id=17202864") as table:
                print(table.df())
    elif num_lesson == 4:
        import duckdb

        # 1 load package(s) were loaded to destination duckdb and into dataset github_api_data
        # The duckdb destination used duckdb:////Users/username/PycharmProjects/ml_for_products/github_api_pipeline.duckdb location to store data
        # ----------
        # conn = duckdb.connect(f"github_api_pipeline.duckdb")
        # conn.sql(f"SET search_path = 'github_api_data'")
        # df = conn.sql("DESCRIBE").df()
        # cols = df.columns.tolist()
        # print(cols)
        # print(df[cols[:5]].head())
        # print()
        # data_table = conn.sql("SELECT * FROM github_api_resource").df()
        # print(data_table[data_table.columns[:5]].head())

        conn = duckdb.connect(f"rest_api_github.duckdb")
        conn.sql(f"SET search_path = 'rest_api_data'")
        # df = conn.sql("DESCRIBE").df()
        # print(df[df.columns[:3]].head())

        for t in ('issues', 'contributors'):
            data_table = conn.sql(f"SELECT * FROM {t}").df()
            print('Num rows, cols: ', t, ' ', data_table.shape)
            print(data_table[data_table.columns[:5]].head())

        
    elif num_lesson == 5:
        pass
    elif num_lesson == 6:
        pass
    elif num_lesson == 7:
        pass
    elif num_lesson == 8:
        pass
    else:
        print('Testing code')