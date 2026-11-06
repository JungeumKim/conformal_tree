"""Preprocessing for the eight regression datasets in the tree and forest comparison."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

DATASETS = {"s1-large": ("Data 1", 4), "s2-large": ("Data 2", 8),
            "bike": ("bike", 64), "bio": ("bio", 64),
            "community": ("community", 32), "concrete": ("concrete", 8),
            "homes": ("homes", 32), "star": ("star", 4)}


def load_dataset(name, data_dir, seed):
    if name in ("s1-large", "s2-large"):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0, 10, 1000)
        mu, sd = (3*np.sin(4/x+0.2)+1.5, x) if name == "s1-large" else (np.sin(x**(-3)), 0.1)
        return x[:, None], rng.normal(mu, sd)
    return _load_real_dataset(name, str(Path(data_dir).resolve()))


@lru_cache(maxsize=6)
def _load_real_dataset(name, data_dir):
    data_dir = Path(data_dir)
    if name == "homes":
        df = pd.read_csv(data_dir / "kc_house_data.csv")
        X, y = df.drop(columns=["id", "date", "price"]).values, df["price"].values
    elif name == "bio":
        df = pd.read_csv(data_dir / "CASP.csv")
        X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    elif name == "concrete":
        data = np.loadtxt(data_dir / "Concrete_Data.csv", delimiter=",", skiprows=1)
        X, y = data[:, :-1], data[:, -1]
    elif name == "bike":
        df = pd.read_csv(data_dir / "bike_train.csv")
        df = pd.concat([df, pd.get_dummies(df["season"], prefix="season")], axis=1)
        df = pd.concat([df, pd.get_dummies(df["weather"], prefix="weather")], axis=1)
        df = df.drop(columns=["season", "weather"])
        dates = pd.DatetimeIndex(df["datetime"])
        df["hour"], df["day"], df["month"] = dates.hour, dates.dayofweek, dates.month
        df["year"] = pd.Series(dates.year, index=df.index).map({2011: 0, 2012: 1})
        df = df.drop(columns=["datetime", "casual", "registered"])
        X, y = df.drop(columns=["count"]).values, df["count"].values
    elif name == "community":
        attributes = pd.read_csv(data_dir / "communities_attributes.csv", sep=r"\s+")
        df = pd.read_csv(data_dir / "communities.data", names=attributes["attributes"])
        df = df.drop(columns=["state", "county", "community", "communityname", "fold"])
        df = df.replace("?", np.nan)
        df[["OtherPerCap"]] = SimpleImputer(strategy="mean").fit_transform(df[["OtherPerCap"]])
        df = df.dropna(axis=1)
        X, y = df.iloc[:, :100].values, df.iloc[:, 100].values
    elif name == "star":
        df = pd.read_csv(data_dir / "STAR.csv")
        maps = {"gender": {"female": 0, "male": 1},
                "ethnicity": {"cauc": 0, "afam": 1, "asian": 2, "hispanic": 3, "amindian": 4, "other": 5}}
        for suffix in ("k", "1", "2", "3"):
            maps["star"+suffix] = {"regular": 0, "small": 1, "regular+aide": 2}
            maps["lunch"+suffix] = {"free": 0, "non-free": 1}
            maps["school"+suffix] = {"inner-city": 0, "suburban": 1, "rural": 2, "urban": 3}
            maps["degree"+suffix] = {"bachelor": 0, "master": 1, "specialist": 2,
                                    "master+" if suffix == "k" else "phd": 3}
            maps["ladder"+suffix] = {"level1": 0, "level2": 1, "level3": 2, "apprentice": 3,
                                    "probation": 4, "pending" if suffix == "k" else "noladder": 5,
                                    "notladder": 6}
            maps["tethnicity"+suffix] = {"cauc": 0, "afam": 1}
        maps["tethnicity3"]["asian"] = 2
        for column, mapping in maps.items():
            # Preserve the original treatment of categories not listed above.
            df[column] = df[column].map(lambda value: mapping.get(value, value))
        df = df.dropna()
        y = sum(df["read"+s] for s in ("k", "1", "2", "3"))
        y += sum(df["math"+s] for s in ("k", "1", "2", "3"))
        X = df.loc[:, list(df.columns[:8])+list(df.columns[17:])].values
    else:
        raise ValueError(f"Unknown dataset: {name}")
    # These conversions match the preprocessing used for the reported results.
    return np.asarray(X, dtype=np.float32).astype(float), np.asarray(y, dtype=np.float32).astype(float).reshape(-1)
