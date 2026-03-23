You don’t need jinja - in standard Library

```python
import os
from string import Template

def read_sql(sql_name, **params) -> str:
    ROOT_SQL_DIR = 'sql'
    sql_path = os.path.join(ROOT_SQL_DIR, f'{sql_name}.sql')
    with open(sql_path, 'r') as f:
        res = f.read()
    res = Template(res).substitute(**params)
    return res

sql = read_sql(
    'batch_features_pd',
    table_id=table_id,
    timestamp_col_name=timestamp_col_name,
		id_col_name=id_col_name
)
```

Example of template

```python
select 
        $id_col_name,
        $timestamp_col_name as ${id_col_name}_$timestamp_col_name,
        $feature_names
    from 
        $table_id d
```

Convert f-string to template

```python
def convert_fstring_to_template(input_string: str):
    import re
    
    pattern = r'\{([^{}]+)\}'
    replacement = r'${\1}'
    result = re.sub(pattern, replacement, input_string)
    
    return result
```