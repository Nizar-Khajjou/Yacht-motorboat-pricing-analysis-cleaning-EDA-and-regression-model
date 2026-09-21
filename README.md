# Yacht/Motorboat Price Analysis

## Overview
An exploratory data analysis and predictive modeling project examining what
drives pricing in the recreational yacht/motorboat market, using real listing
data scraped from boat24.com in 2020.

## Data Source
- **Dataset:** [Yacht/Motorboat Pricing Data (10,000+ listings)](https://www.kaggle.com/datasets/artemkorottchenko/yachtmotorboat-pricing-data) — Kaggle, by Artem Korottchenko
- 10,344 listings, 38 original features (price, dimensions, manufacturer, boat type, equipment, etc.)

## Question
**What drives the price of a recreational yacht/motorboat, and how well can
it be predicted from a boat's basic characteristics?**

## Data Cleaning
- Fixed file encoding issues (UTF-8 with error replacement) affecting
  special characters like the £ symbol
- Parsed the raw `Price` text field into a `Currency` code and a numeric
  `Price_numeric` value, handling European number formatting and non-numeric
  entries ("Price on request")
- Converted all listings into a single comparable currency (`Price_EUR`)
  using approximate 2020 exchange rates
- Cleaned `Length` and `Width` from text with units ("4.00 m") into numeric
  meters
- Simplified `Boat Type` (135 raw values, many rare multi-label combinations)
  into ~20 usable categories, grouping rare/combined types into "Other/Mixed"

## Key Findings
- **Size matters, moderately:** Length and Width each show a moderate
  positive correlation with price (r = 0.57 and 0.52 respectively)
- **Age barely matters on its own:** Year Built shows almost no correlation
  with price (r = 0.11)
- **Boat type is a strong price driver:** median prices range from
  €23,575 (Fishing Boat) to €4,401,500 (Mega Yacht) — a >180x spread
- **Price is heavily right-skewed:** a small number of superyacht listings
  pull the mean (€299,821) far above the median (€89,646)

## Model
A linear regression model was trained on Length, Width, Year Built, and
Boat Type (one-hot encoded) to predict price.

| Approach | R² |
|---|---|
| Predicting raw price (EUR) | 0.55 |
| Predicting log10(price) | **0.82** |

Modeling price on a **log scale** substantially improved performance,
reflecting that boat prices scale multiplicatively (a larger/fancier boat
costs a percentage more, not a fixed amount more) rather than additively.
This is the key methodological insight of the project — a raw-price model
understates how predictable pricing actually is.

## Tools
Python, Pandas, Matplotlib, scikit-learn (LinearRegression)

## Possible Next Steps
- Incorporate `Manufacturer` as a predictor (brand effect)
- Try non-linear models (e.g. Random Forest) to capture interactions between
  size and type
- Build a Power BI dashboard on top of the cleaned dataset
