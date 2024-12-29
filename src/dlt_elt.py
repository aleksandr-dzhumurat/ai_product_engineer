import dlt

# Sample data containing pokemon details
data = [
    {"id": "1", "name": "bulbasaur", "size": {"weight": 6.9, "height": 0.7}},
    {"id": "4", "name": "charmander", "size": {"weight": 8.5, "height": 0.6}},
    {"id": "25", "name": "pikachu", "size": {"weight": 6, "height": 0.4}},
]
import dlt

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

# Lesson 1
import requests

# Example resource
@dlt.resource
def github_events():
    url = f"https://api.github.com/orgs/dlt-hub/events"
    response = requests.get(url)
    yield response.json()


# here is your code
import dlt
from dlt.sources.helpers import requests

def pagination(url):
    while True:
        response = requests.get(url)
        response.raise_for_status()
        yield response.json()
        
        # Get next page
        if "next" not in response.links:
            break
        url = response.links["next"]["url"]

@dlt.resource(table_name="issues")
def get_issues():
    url = "https://api.github.com/repos/dlt-hub/dlt/issues?per_page=100"
    yield pagination(url)


if __name__=='__main__':
    print('hello')
    num_lesson = 3
    if num_lesson == 1:
        load_info = pipeline.run(data, table_name="pokemon")
        dataset = pipeline.dataset(dataset_type="default")
        result_df = dataset.pokemon.df()
        print(result_df.shape)
        print(result_df.head())
        print(load_info)
    elif num_lesson == 2:
        load_info = pipeline.run(my_dict_list)
        print(load_info)
        print(pipeline.dataset(dataset_type="default").pokemon_new.df())
    else:
        # Set pipeline name, destination, and dataset name
        pipeline = dlt.pipeline(
            pipeline_name="github_pipeline",
            destination="duckdb",
            dataset_name="github_data"
        )

        load_info = pipeline.run(github_events())  # here is your code
        print(load_info)
        print(pipeline.dataset(dataset_type="default").github_events.df())
