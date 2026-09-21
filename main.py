import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

pd.set_option('display.max_rows', 200)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('boat_dataset.csv', encoding='utf-8', encoding_errors='replace')

# ============================================================
# 2. CLEAN PRICE (currency + numeric value)
# ============================================================
df['Currency'] = df['Price'].str.extract(r'([A-Z]{3}|£)')

df['Price_numeric'] = (
    df['Price']
    .str.replace(r'[A-Z]{3}|£', '', regex=True)   # remove currency code or symbol
    .str.replace('.', '', regex=False)             # remove thousands-separator dots
    .str.replace(',-', '', regex=False)            # remove trailing ",-"
    .str.strip()
)

# "Price on request" and similar non-numeric entries -> missing
df.loc[~df['Price_numeric'].str.match(r'^\d+$'), 'Price_numeric'] = None
df['Price_numeric'] = df['Price_numeric'].astype(float)

# Convert all currencies into a single comparable currency (EUR)
exchange_rates_to_eur = {
    'EUR': 1.0,
    'CHF': 0.93,
    '£': 1.15,
    'USD': 0.85,
    'DKK': 0.134,
    'SEK': 0.087,
}
df['Price_EUR'] = df['Price_numeric'] * df['Currency'].map(exchange_rates_to_eur)

print("--- Price cleaning check ---")
print(df[['Price', 'Currency', 'Price_numeric', 'Price_EUR']].head(10))
print(df['Currency'].value_counts())
print(df['Price_numeric'].isna().sum(), "rows have no usable price")
print(df['Price_EUR'].describe())

# ============================================================
# 3. CLEAN LENGTH & WIDTH
# ============================================================
df['Length_m'] = df['Length'].str.replace(' m', '', regex=False).astype(float)
df['Width_m'] = df['Width'].str.replace(' m', '', regex=False).astype(float)

print("\n--- Length/Width cleaning check ---")
print(df[['Length_m', 'Width_m']].describe())

# ============================================================
# 4. SIMPLIFY BOAT TYPE (group rare/combined types)
# ============================================================
common_types = df['Boat Type'].value_counts()
common_types = common_types[common_types >= 50].index  # keep types with 50+ listings

df['Boat_Type_Clean'] = df['Boat Type'].where(df['Boat Type'].isin(common_types), 'Other/Mixed')

print("\n--- Boat type groups ---")
print(df['Boat_Type_Clean'].value_counts())

# ============================================================
# 5. EXPLORATORY ANALYSIS
# ============================================================
# Log-scale price distribution (raw price is heavily right-skewed by superyachts)
plt.figure(figsize=(10, 5))
df['Price_EUR'].dropna().apply(np.log10).hist(bins=50)
plt.title('Distribution of Boat Prices (log scale)')
plt.xlabel('log10(Price in EUR)')
plt.ylabel('Number of listings')
plt.show()

# Correlation of numeric features with price
numeric_cols = ['Length_m', 'Width_m', 'Year Built', 'Price_EUR']
print("\n--- Correlation with Price_EUR ---")
print(df[numeric_cols].corr()['Price_EUR'])

# Length vs Price scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Length_m'], df['Price_EUR'], alpha=0.3, s=10)
plt.yscale('log')
plt.title('Boat Length vs Price')
plt.xlabel('Length (m)')
plt.ylabel('Price (EUR, log scale)')
plt.show()

# Median price by boat type
print("\n--- Median price by boat type ---")
price_by_type = df.groupby('Boat_Type_Clean')['Price_EUR'].median().sort_values(ascending=False)
print(price_by_type)

# ============================================================
# 6. PREPARE DATA FOR MODELING
# ============================================================
model_cols = ['Length_m', 'Width_m', 'Year Built', 'Boat_Type_Clean', 'Price_EUR']
model_df = df[model_cols].dropna()

model_df_encoded = pd.get_dummies(model_df, columns=['Boat_Type_Clean'], drop_first=True)

X = model_df_encoded.drop('Price_EUR', axis=1)
y = model_df_encoded['Price_EUR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 7. MODEL 1 — predict raw price
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model 1: raw price ---")
print("R² score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# ============================================================
# 8. MODEL 2 — predict log(price) (handles skew from superyachts)
# ============================================================
model_df_encoded['Log_Price'] = np.log10(model_df_encoded['Price_EUR'])

X_log = model_df_encoded.drop(['Price_EUR', 'Log_Price'], axis=1)
y_log = model_df_encoded['Log_Price']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)

y_pred_log = model_log.predict(X_test_log)

y_pred_eur = 10 ** y_pred_log
y_test_eur = 10 ** y_test_log

print("\n--- Model 2: log(price) ---")
print("R² score (log scale):", r2_score(y_test_log, y_pred_log))
print("Mean Absolute Error (converted back to EUR):", mean_absolute_error(y_test_eur, y_pred_eur))