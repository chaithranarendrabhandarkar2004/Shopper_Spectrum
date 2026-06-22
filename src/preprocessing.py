import pandas as pd


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Date conversion
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Total Amount
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df