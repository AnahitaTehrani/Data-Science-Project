# This code was made by me with the help from Chat gpt


import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# 1. Load and clean raw data from CSV files
# ======================

with open("Price change Spotify.csv", encoding="utf-8") as f:
    raw_price_lines = f.readlines()

with open("Spotify premium user.csv", encoding="utf-8") as f:
    raw_premium_lines = f.readlines()

# Extract price data from raw lines
price_clean = []
for line in raw_price_lines[1:]:
    parts = line.strip().split(";")
    if len(parts) < 3:
        continue
    time_str = parts[1].strip()
    price_str = parts[2].strip().replace("\xa0€", "").replace(",", ".")
    try:
        price = float(price_str)
        price_clean.append((time_str, price))
    except:
        continue

# Extract premium user data from raw lines
premium_clean = []
for line in raw_premium_lines[2:]:
    parts = line.strip().split(";")
    if len(parts) < 3:
        continue
    quarter = parts[1].strip()
    try:
        users = int(parts[2].strip())
        q, y = quarter.split()
        month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[q]
        date = datetime(int(y), month, 1)
        premium_clean.append((date, users))
    except:
        continue

# Define price periods with start and end dates
price_periods = []
for period_str, price in price_clean:
    if "–" in period_str or "-" in period_str:
        parts = period_str.replace("–", "-").split("-")
        try:
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            try:
                start_date = datetime.strptime(start_str, "%Y")
            except:
                start_date = datetime.strptime(start_str, "%B %Y")
            try:
                end_date = datetime.strptime(end_str, "%B %Y")
            except:
                end_date = datetime.strptime(end_str, "%Y")
            end_date += relativedelta(months=1)
        except:
            continue
    elif "Since" in period_str:
        try:
            start_date = datetime.strptime(period_str.replace("Since", "").strip(), "%B %Y")
            end_date = datetime(2100, 1, 1)
        except:
            continue
    else:
        continue
    price_periods.append((start_date, end_date, price))

# Function to match date with corresponding price
def match_price(date):
    for start, end, price in price_periods:
        if start <= date <= end:
            return price
    return None

# Create final DataFrame with matched price and premium users
premium_df = pd.DataFrame(premium_clean, columns=["Date", "PremiumUsers_Mio"])
premium_df["Price_EUR"] = premium_df["Date"].apply(match_price)
final_df = premium_df.dropna()

# 2. Calculate correlation between price and premium users
# ======================

print("Correlation:")
print(final_df[["Price_EUR", "PremiumUsers_Mio"]].corr())

# 3. Create scatter plot of price vs premium users
# ======================

plt.figure(figsize=(8, 5))
plt.scatter(final_df["Price_EUR"], final_df["PremiumUsers_Mio"], s=60)
plt.title("Correlation Between Spotify Price and Number of Premium Users")
plt.xlabel("Price in EUR")
plt.ylabel("Premium Users (in millions)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Create time series chart of price and premium users
# ======================

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel("Year")
ax1.set_ylabel("Premium Users (Mio)", color='tab:blue')
ax1.plot(final_df["Date"], final_df["PremiumUsers_Mio"], label="Premium Users", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Price (EUR)", color='tab:red')
ax2.plot(final_df["Date"], final_df["Price_EUR"], label="Price", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle("Spotify: Development of Price and Number of Premium Users")
fig.tight_layout()
plt.show()