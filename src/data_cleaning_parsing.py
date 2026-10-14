import ast
import pandas as pd
import re
import numpy as np

def clean_price(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    match = re.search(r"\d+\.?\d*", str(x))
    return float(match.group()) if match else np.nan

def parse_categories(x):
    if isinstance(x, list):
        if x and isinstance(x[0], list):
            x = x[0]
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                return parsed[0]
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []

def extract_author_name(x):
    """clean author name from dict"""
    if pd.isna(x):
        return "unknown"
    if isinstance(x, dict):
        if 'name' in x:
            name = x['name']
            return name if name else 'unknown'
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return extract_author_name(parsed)
        except (ValueError, SyntaxError):
            return x.strip() or "unknown"
    return "unknown"