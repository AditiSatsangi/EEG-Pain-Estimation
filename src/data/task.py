# src/data/tasks.py
from sklearn.preprocessing import LabelEncoder

def load_task_data(df, task):
    le = LabelEncoder()
    window_le = LabelEncoder()

    if "window" not in df.columns:
        df["window"] = "unknown"

    y_window = window_le.fit_transform(df["window"])

    if task == "none_vs_pain":
        df["binary"] = df["rating_bin"].apply(lambda x: "none" if x == "none" else "pain")
        y = le.fit_transform(df["binary"])

    elif task == "pain_only":
        df = df[df["rating_bin"] != "none"]
        y = le.fit_transform(df["rating_bin"])
        y_window = window_le.transform(df["window"])

    elif task == "pain_threshold":
        df["binary"] = df["rating_bin"].apply(
            lambda x: "no_significant_pain" if x in ["none", "low"] else "significant_pain"
        )
        y = le.fit_transform(df["binary"])

    else:  # 5-class
        y = le.fit_transform(df["rating_bin"])

    return y, le, df, y_window, window_le
