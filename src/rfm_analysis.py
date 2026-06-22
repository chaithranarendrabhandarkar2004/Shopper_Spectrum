import pandas as pd


def create_rfm(df):

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    })

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    return rfm