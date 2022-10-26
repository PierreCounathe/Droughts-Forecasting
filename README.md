# Droughts-Forecasting
---
**Content of this README:**
- Motivation
- Results
- Data despcription
- Problem approach
- Next steps
- Repository organization
---

## Motivation

Since 1980, the US experienced 26 major droughts, each event averaging a cost of $9.6 billion. It is also well known that droughts frequency and intensities will increase in the US and globally. It is thus interesting to build droughts prediction models that could be able to generalize well and to predict droughts with the greatest possible lead times

## Results

A summary of the performance of the different models tested can be found in [`summary/Droughts Forecasting Summary.pdf`](https://github.com/PierreCounathe/Droughts-Forecasting/blob/lstmbranch/summary/Droughts%20Forecasting%20Summary.pdf).

Mainly, we notice that a Naive model has comparable performance with previous work done on droughts forecasting, that non-temporal Machine Learning models have strong limitations for drought prediction, and that it is possible to build strong Deep Learning models that can predict droughts with lead times greater than 6 weeks.

## Data description

The dataset used for this project is composed of daily weather observations in the US, that come from the NASA POWER (Prediction of Worldwide Energy Resources) Project, of soil data that come from the Harmonized World Soil Database, and of drought score that come from the U.S. Drought Monitor.

**Weather data:** The weather data contains daily observations of 18 weather variables (Temperature, Wind, Humidity, Precipitations, etc. information) in the US, at a county level (each of them represented by a FIPS code). It is already split in three sets: a train set, a validation and test set that all cover 3058 counties, respectively from 2000 to 2016, 2016 to 2018 and from 2018 to 2020.

**Soil data:** The soil data is static data that contains around 30 variables concerning the soil agricultural characteristics (distribution of slopes levels, aspects, land use, etc. in each county). This data is available for every county that either appears in the train, or in all timeseries.

**Target variable:** The U.S. Drought Monitor (USDM) is produced through a partnership between the National Drought Mitigation Center at the University of Nebraska-Lincoln, the United States Department of Agriculture, and the National Oceanic and Atmospheric Administration. It provides for each of the 3058 studied counties a continuous drought score ranging from 0 to 5, once per week and for each county, but is often presented in 5 classes (No Drought - Exceptional Drought). The particularity of this drought score is that it combines multiple drought indices or indicators to create a unique metric, that results in being a more objective drought value. Is is a combination of the Palmer Drought Severity Index (PDSI), of the CPC Soil Moisture Model,and of the Standardized Precipitation Index (SPI). As the score is provided on a weekly basis, we interpolate it for convenience in the model development.

## Problem approach

**Naive models:** We first develop two Naive models that represent our performance baseline. 

**Classic ML models:** Then we address the problem using classic ML techniques (Ridge regression, Gradient Boosted Trees). We flatten the weather observations to take into account long-term dependencies between weather and droughts. We try different combination of regularization, time window (how far we look in the past), features to best solve the problem. This exploration lies in `src/ml_models.ipynb`.

**LSTM:** Using the best set of features found while devising ml models, we develop an LSTM model, that is run in `src/lstm_model.ipynb`.

## Discussion and next steps

**Discussion:** The fact that the Naive model performs that well compared to the other models opens a discussion on the prediction’s scope. 6 weeks might be too short compared to the timeframe of droughts phenomena. Also, the fact that the best LSTM has such a high score advocates for predictions with higher lead times.

**Next steps:** Thus, immediate future work would consist in determining what prediction lead time can be achieved with the best model. The greater the lead time is, the easier it is for farmers and corporations to adapt to droughts levels changes. Also, other predictors such as weather forecasts could be included, and Attention Networks and Transformers could be used in place of an LSTM.

## Repository organisation

    .
    ├── README.md
    ├── data
    │   ├── train_timeseries_lower_dim_2.pickle
    │   └── validation_timeseries_lower_dim_2.pickle
    ├── src
    │   ├── constants.py
    │   ├── data_exploration.ipynb
    │   ├── lstm.py
    │   ├── lstm_model.ipynb
    │   ├── ml_models.ipynb
    │   ├── preprocessing.py
    │   └── process_outputs.py
    └── summary
        └── Droughts Forecasting Summary.pdf
