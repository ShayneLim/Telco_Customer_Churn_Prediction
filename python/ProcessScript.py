import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Load the Data
def load_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

#Clean and Process the Data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    """1. Clean the Data"""
    #TotalCharges has " " cells -> replace blanks with 0 (these customers have tenure = 0)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    #Remove customerID as it is not needed for modelling
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])


    """2. Format the Data"""
    #cols OnlineSecurity...StreamingMovies have "No Internet Service" on top of "No" -> change to "No" if InternetService is "No"
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

    #MultipleLines has "No phone service" -> change to "No" if PhoneService is "No"
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    # Binary encode Yes/No columns to 1/0
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({"Yes": 1, "No": 0}).astype("int64")


    """3. Performance Tweaks"""
    #Create categorical bins for tenure: 0–12, 12–24, 24–36, 36–48, 48–60, 60+
    tenure_bins = [0, 12, 24, 36, 48, 60, df["tenure"].max() + 1]
    tenure_labels = ["0-12", "12-24", "24-36", "36-48", "48-60", "60+"]
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        right=False,
        include_lowest=True
    )

    #One-Hot Encode Categories
    # - gender -> gender_Male, gender_Female
    # - InternetService -> DSL, Fiber optic, No
    # - Contract -> Month-to-month, One year, Two year
    # - PaymentMethod -> 4 types
    # - tenure_group -> binned categories
    ohe_cols = ["gender", "InternetService", "Contract", "PaymentMethod", "tenure_group"]
    ohe_cols = [col for col in ohe_cols if col in df.columns]  # safety check
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    #Scale the Numeric Columns
    scaler = MinMaxScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

if __name__ == "__main__":
    init_Data = load_data("data/RawTelcoData.csv")
    processed_Data = process_data(init_Data) #This is the df that you want to use for training @shayne

    #This is just for my checking.
    #I outputted the first 5 lines of the final df to a test_output.csv so I can check to make sure processing worked correctly.
    test_output_path = "data/test_output.csv"
    processed_Data.head().to_csv(test_output_path, index=False)

    print(f"\nPreview saved to: {test_output_path}")
