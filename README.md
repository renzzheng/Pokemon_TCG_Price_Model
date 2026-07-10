# Pokémon TCG Price Model

Machine learning project that predicts Pokémon trading card market prices from structured card metadata using the PokémonTCG API.

## Introduction & Problem

In recent years, the Pokémon Trading Card Game (TCG) has experienced a resurgence in popularity and has developed a large secondary market where collectors and investors buy and sell individual cards. Card prices vary widely and may be influenced by factors such as rarity, set, card type, artist, and the popularity of specific Pokémon.

This project explores whether machine learning models can predict Pokémon card market prices using only structured card metadata. The goal is not only to estimate market price, but also to identify which features appear most influential in a card’s predicted value.

*Since this dataset does not include historical time-series price data, this project does **not** attempt to forecast future prices. Instead, it focuses on modeling the current price structure of Pokémon TCG cards based on available metadata.*

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, PokémonTCG API, Google Colab

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- Random Forest with log-transformed target

## Model Results

| Model | Test MAE | Test R² |
|---|---:|---:|
| Linear Regression | 15.62 | 0.282 |
| Ridge Regression | 15.16 | 0.321 |
| Lasso Regression | 11.70 | 0.276 |
| Elastic Net | 15.50 | 0.289 |
| Random Forest | 11.68 | 0.365 |
| Random Forest with Log Target | 5.36 | 0.661 |

## Dataset and Source

**Source:** Pokémon TCG API: https://pokemontcg.io/  
**Credit:** Andrew Backes and contributors

This dataset contains Pokémon Trading Card Game card metadata and market price information collected through the PokémonTCG API. Each record includes structured attributes such as rarity, set, card type, artist, supertypes, subtypes, release information, and TCGPlayer market price data.

The PokémonTCG API contains approximately 18,000 unique Pokémon card records. Because API access for version 1 was unavailable during the final stage of this project, I used a saved dataset of U.S.-released cards collected up to 2026-01-02. After flattening the available market price fields, the final dataset contained 33,005 card-price observations.

*Market prices are highly volatile and may be affected by collector demand, speculation, and market manipulation. As a result, the conclusions from this project should be interpreted as correlations rather than causal claims. The dataset also lacks important contextual features such as card condition, grading status, print population, and historical market trends.*

## Preliminary Approach

I began with a baseline Linear Regression model to establish a simple comparison point. From there, I tested regularized linear models, including Ridge, Lasso, and Elastic Net, to evaluate whether regularization could improve performance with many one-hot encoded categorical features.

To incorporate a nonlinear model, I also tested Random Forest models, which can capture more complex feature relationships than linear models. Because Pokémon card prices are highly skewed, I also tested a Random Forest model using a log-transformed target variable.

## Features Used

The target variable was:

- `market_price`

The model used metadata features such as:

- rarity
- artist
- release date
- set name
- supertype
- subtype
- card subject

## Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- R² score
- Predicted-vs-actual price visualizations

For Random Forest models, I also analyzed feature importances to identify which metadata features were most influential in the model’s predictions. These feature importances were compared with descriptive analyses such as average market price by rarity, set, and card subject.

## Results Summary

The Random Forest with a log-transformed target was the strongest model overall, achieving the lowest test MAE at about \$5.36 and the highest test R² at about 0.661. This suggests that applying a log transformation helped the model handle the highly skewed distribution of Pokémon card prices.

Among the non-log models, Random Forest and Lasso achieved the lowest average prediction errors, with test MAE values of about \$11.68 and \$11.70 respectively. Ridge Regression had the strongest R² among the regularized linear models, explaining about 32.1% of the variation in test prices.

Overall, the results suggest that structured metadata and individual card subject features contain meaningful pricing signal. However, even the best model still struggled with some high-value outliers, showing that collectible card prices are also influenced by factors not fully captured in the dataset.
