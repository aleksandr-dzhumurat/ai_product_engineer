import pandas as pd
import polars as pl

def prepare_evaluation_df(input_df, negatives_per_one_positive=2):
    input_df = pl.from_pandas(input_df) if isinstance(input_df, pd.DataFrame) else input_df
    key_fields = ['category']
    # Select distinct combinations of 'dt', 'StoreID', 'CustomerID', and 'IsOnline'
    prepared_input_df = (
        input_df
        .select(key_fields + ['CustomerID', 'ProductID']).unique()
        .with_columns([pl.lit(1).alias('target').cast(pl.Int64)])
    )
    print('Transformation started...')
    cadidates_full_df = (
        input_df.select(key_fields + ['CustomerID']).unique()
        .join(
            input_df.select(key_fields + ['ProductID']).unique(),
            on=key_fields,
            how='inner'
        )
        .join(
            prepared_input_df,
            on=key_fields+['ProductID', 'CustomerID'],
            how='left'
        )
    )
    print(f"Negative candidates: {cadidates_full_df.filter(pl.col('target').is_null()).height}, Positive samples: {input_df.height}")
    negative_candidates_df = (
        cadidates_full_df.filter(pl.col('target').is_null())
        .sample(n=int(input_df.height * negatives_per_one_positive), seed=42)
        .with_columns([pl.lit(0).alias('target').cast(pl.Int64)])
    )
    user_item_df = (
        pl.concat([
            prepared_input_df.select(key_fields+['CustomerID', 'ProductID', 'target']),
            negative_candidates_df.select(key_fields+['CustomerID', 'ProductID', 'target'])
        ])
        .sort(by=['CustomerID'])
    )
    print(f"Num negatives {user_item_df.to_pandas()['target'].value_counts(normalize=True).to_dict().get(0)}")
    return user_item_df