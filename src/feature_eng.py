def favorite_genre(df, user_col, genre_col):
    """returns most frequent user's genre: DataFrame (user_idx, favorite_<genre_col>)"""
    counts = (df.groupby([user_col, genre_col]).size().reset_index(name="n"))
    fav = (counts.sort_values([user_col, "n"], ascending=[True, False]).drop_duplicates(user_col)
        .rename(columns={genre_col: f"fav_{genre_col}"})[[user_col, f"fav_{genre_col}"]])
    return fav
