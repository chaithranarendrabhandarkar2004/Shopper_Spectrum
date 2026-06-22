import pandas as pd


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    df = df.dropna(subset=["CustomerID"])

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df
