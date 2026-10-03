import pandas as pd
from scipy.stats import zscore

# Create a 10-day retail sales dataset
data = {
    "Day": [
        "Day 1", "Day 2", "Day 3", "Day 4", "Day 5",
        "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"
    ],
    "Revenue": [
        1200, 1350, None, 1280, 1420,
        1300, None, 1380, 15000, 1250
    ]
}

df = pd.DataFrame(data)

print("Original Retail Sales Data:")
print(df)

# (a) Report missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Calculate median revenue
median_revenue = df["Revenue"].median()

print("\nMedian Revenue:", median_revenue)

# Fill missing Revenue values with median
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nData After Filling Missing Values:")
print(df)

# (b) Min-Max Normalization
min_revenue = df["Revenue"].min()
max_revenue = df["Revenue"].max()

df["Revenue_MinMax"] = (
    (df["Revenue"] - min_revenue)
    / (max_revenue - min_revenue)
)

# Z-score Standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])

# (c) IQR-based Outlier Detection
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Outlier"] = (
    (df["Revenue"] < lower_bound)
    | (df["Revenue"] > upper_bound)
)

print("\n--- IQR Outlier Detection ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

print("\nFlagged Outlier Day(s):")
print(df.loc[df["Outlier"], ["Day", "Revenue"]])

# (d) Final Table
print("\n--- Final Retail Sales Table ---")

print(
    df[
        [
            "Day",
            "Revenue",
            "Revenue_MinMax",
            "Revenue_Zscore",
            "Outlier"
        ]
    ]
)
