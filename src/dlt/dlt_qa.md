
What is dlt? * An open-source Python platform for data management

Which tasks does dlt NOT automate? Data modeling

How does dlt handle database schemas? Produces the same schemas on different DB systems with slightly different datatypes based on database support.


# Lesson 1

Q: Fetch the data from the pokemon table into a dataframe and count the number of columns in the table:

Answer: 6

Q: What is one of the main benefits of using a resource in the dlt library?
A: It allows you to extract data from a specific endpoint or data source, and can also perform transformations on the data before yielding it.

Q: What is the primary role of a source in the dlt library and why is it useful?
A: The primary role of a source is to group related resources into a single source. It is useful for promoting modularity and organization in your codebase.

Q: What is the main difference between a resource and a transformer in the dlt library?
A: A resource yields data from a specific endpoint or data source, while a transformer transforms data from a resource.

Q: How many columns has the github_repos table?
A: 106

Q:How many columns has the github_stargazer table?
A: 21

What features does support RestAPIClient? All Above


What are the three ways to set up configurations and credentials in dlt? All Above


Question What type of pagination we use for GitHub API? Answer: the Link header, as used by the GitHub API so answer is HeaderLinkPaginator https://dlthub.com/docs/general-usage/http/rest-client#headerlinkpaginator


What is a verified source in dlt? A: A pre-built, fully customizable source for data pipelines


Which command initializes a dlt project with a verified source and a destination?  dlt init <verified-source> <destination>

Q: What is the main advantage of using the rest_api built-in source in dlt? A: It provides a declarative way to configure REST API sources, handling pagination and authentication automatically.

Q: What is the difference between RestAPI Client and rest_api_source? A: The RestAPI Client is used by rest_api_source under the hood; rest_api_source has a declarative interface for creating Rest API dlt sources.

Q: Which names below are dlt destinations? A: Postgres, Duckdb, BigQuery


Q: What is the purpose of incremental loading in data pipelines? To load only new or changed data instead of reloading the entire dataset. This approach provides several benefits, including low-latency data transfer and cost savings.

Q: What is the best way to deal with loading stateless (unchangeable) data? A: Via replace write disposition

Q: What are the three options available for the write_disposition parameter in dlt for incremental loading?A: replace, append, merge

Q: What is the default write_disposition in dlt? A: Append

Q: What must you specify when using the merge write disposition in dlt? A:Primary key or merge key

Q: Which dlt function is used to define an incremental field for tracking new or updated data? A: dlt.sources.incremental

Q:What are the three main steps in the dlt process? A: Extract, Transform, and Load

Q: What is the primary purpose of the Extract step in a data pipeline? A:To retrieve raw data from various sources

Q: What does the Normalize step in the dlt process do? It unpacks the nested structure into relational tables, infers data types, links tables to create parent-child relationships, and creates a schema

Q: What is the primary function of the Load step in a data pipeline? A: To move data into a target system or storage destination


Q: What is one benefit of using progress="log" in a pipeline configuration? It provides real-time metrics on CPU and memory usage


How can you specify the file format for intermediary files directly in the pipeline.run() method? A: loader_file_format

What does the file_max_items parameter control in a pipeline? A: The maximum number of items stored in a single intermediary file


--------

Q: What does the schema describe in the context of dlt? The structure of normalized data, such as tables and columns

Q: What does the version_hash in a schema represent? A hash generated from the schema’s content

Q How does the dlt library infer the initial schema for a pipeline run? Q: It automatically infers the schema from the first pipeline run

Q: What is the default naming convention used by dlt? A: snake_case

Q: Where can you view schemas in dlt? A: All of the above

Q: What is a load package in dlt? A: A collection of jobs with data for specific tables generated during a pipeline execution

Q: What is a state in the context of the dlt library? A: A Python dictionary that lives alongside your data.

??? Q: What is one of the main uses of the state in the dlt library? To store and retrieve values across pipeline runs, useful for implementing incremental loading and avoiding duplicate requests.

Q What does the pipeline's last run trace contain in dlt?  A: It contains information about the last run of the pipeline, including things like row counts.

Q: How can you view the most recent load package using CLI? A: dlt pipeline github_pipeline load-package
Q What is the _dlt_loads table used for? Tracking completed loads with their statuses








# Lesson 3

## Exercise 1: Pagination with RESTClient
Explore the cells above and answer the question below.

## Exercise 2: Run pipeline with dlt.secrets.value
Explore the cells above and answer the question below using sql_client.

Question
Who has id=17202864 in the stargazers table? Use sql_client.

Answer: rudolfix

# Lesson 4

## Exercise 1: Run rest_api source

Question: How many columns has the issues table? Answer: 164

## Exercise 2: Create dlt source with rest_api
Add contributors endpoint for dlt repository to the rest_api configuration:

Question% How many columns has the contributors table? Answer: 22

## Exercise 3: Run sql_database source
Question: How many columns does the family table have? Answer: 37


## Exercise 4: Run filesystem source
Question: How many columns does the userdata table have? Answer: 15



# Lesson 5

Question
How many columns does the comments table have?

Answer: 55

# Lesson 6

What data type does the column version in the _dlt_version table have?

Answer: 
  _dlt_version:
    columns:
      version:
        data_type: bigint