import pandas as pd
from pprint import pprint

def merge_DS1():
    train = pd.read_csv("./DS1/train.csv")
    test = pd.read_csv("./DS1/test.csv")
    test_labels = pd.read_csv("./DS1/test_labels.csv")

    test_combined = test.merge(test_labels, on="id")
    test_cleaned = test_combined[(test_combined[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] != -1).all(axis=1)]
    print(f"Train samples = {len(train)}")
    print(f"Test cleaned samples = {len(test_cleaned)}")

    combined_DS1 = pd.concat([train, test_cleaned], ignore_index=True)

    return combined_DS1

def map_class(row):
    """
    Maps the format from DS2 to DS1 using class label.
    0 = hate speech
    1 = offensive language
    2 = neither
    """
    if row['class'] == 0:  # hate speech
        return pd.Series([1, 0, 1, 0, 1, 1])
    elif row['class'] == 1:  # offensive language
        return pd.Series([1, 0, 1, 0, 1, 0])
    else:  # neither
        return pd.Series([0, 0, 0, 0, 0, 0])


def change_DS2_format():
    df = pd.read_csv("./DS2/twitter_data.csv")
    df = df.rename(columns={"tweet": "comment_text"})

    df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = df.apply(map_class, axis=1)

    df_cleaned = df[['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    # Add id
    df_cleaned.insert(0, 'id', [f"ds2_{i:06d}" for i in range(len(df_cleaned))])

    return df_cleaned

def load_data(with_DS3=False):
    ds1 = merge_DS1()
    ds2 = change_DS2_format()
    
    if with_DS3:
        ds3 = pd.read_csv('./DS3/processed_lyrics_dataset.csv')
        ds4 = pd.read_csv('./DS4/identity_hate.csv')
        return pd.concat([ds1, ds2, ds3, ds4], ignore_index=True)
    
    return pd.concat([ds1, ds2], ignore_index=True)




# import random
# def one_time_use():
    df = pd.read_csv('./DS3/updated_lyrics.csv')
    num_rows = len(df)
    num_rows_to_drop = int(num_rows * 0.85)
    rows_to_drop = random.sample(range(num_rows), num_rows_to_drop)


    df_dropped = df.drop(rows_to_drop)
    df_dropped.to_csv('./DS3/lyrics_half.csv', index=False)
    print("DOne")






# pprint(type(load_data()))
