# Selected-leaf coverage on the real datasets

For each regression dataset we held the first saved benchmark predictor fixed and resampled its calibration and test records. For classification we used the saved model probabilities and scored the true class by one minus its predicted probability. We used the original benchmark leaf caps and minimum child sizes. All methods used the same draws. An independent pilot selected four nonroot cells per dataset using only their selection frequencies, pooled across the two fits.

Except for skin disease, we drew 10,000 independent samples of size n+1 without replacement from the available records. Within each sample we took 40 observations in turn as the test point, giving 400,000 evaluations. We used pointwise 95% Monte Carlo intervals for coverage, with standard errors computed over the 10,000 independent samples. For skin disease, coverage in some pilot-selected cells was close to 100%; we used 400,000 independent calibration and test draws (one query per sample) and exact binomial intervals. The estimates describe repeated splits of these records with the predictor fixed.

Selection frequency, observed coverage and lower bounds are percentages. Penalties are percentage points. Each lower bound is estimated as max(0, 90% minus the penalty).

## Hallucination tree

Dataset | Cell | Selected (%) | Penalty (pp) | Lower bound (%) | Coverage (%) | 95% interval (%)
--- | --- | --- | --- | --- | --- | ---
bike | A | 7.77 | 0.34 | 89.66 | 90.18 | 89.84–90.51
bike | B | 2.88 | 0.15 | 89.85 | 90.47 | 89.95–90.99
bike | C | 1.67 | 0.40 | 89.60 | 90.83 | 90.14–91.51
bike | D | 3.40 | 0.55 | 89.45 | 90.53 | 90.03–91.02
bio | A | 15.32 | 0.01 | 89.99 | 90.10 | 89.87–90.34
bio | B | 9.51 | 0.01 | 89.99 | 90.15 | 89.86–90.45
bio | C | 1.54 | 0.02 | 89.98 | 89.96 | 89.24–90.69
bio | D | 36.05 | 0.01 | 89.99 | 89.94 | 89.79–90.09
community | A | 1.32 | 1.33 | 88.67 | 91.14 | 90.39–91.89
community | B | 0.91 | 1.37 | 88.63 | 91.48 | 90.63–92.34
community | C | 0.69 | 1.12 | 88.88 | 91.26 | 90.26–92.27
community | D | 1.17 | 1.17 | 88.83 | 91.41 | 90.67–92.16
concrete | A | 23.03 | 0.90 | 89.10 | 90.82 | 90.65–91.00
concrete | B | 5.90 | 0.86 | 89.14 | 90.92 | 90.56–91.28
concrete | C | 1.18 | 2.23 | 87.77 | 89.55 | 88.75–90.36
concrete | D | 5.36 | 0.00 | 90.00 | 92.78 | 92.45–93.12
homes | A | 21.64 | 0.03 | 89.97 | 89.84 | 89.64–90.04
homes | B | 11.61 | 0.01 | 89.99 | 90.05 | 89.77–90.32
homes | C | 2.32 | 0.11 | 89.89 | 90.03 | 89.42–90.63
homes | D | 2.79 | 0.04 | 89.96 | 90.52 | 89.97–91.06
star | A | 12.35 | 0.64 | 89.36 | 90.57 | 90.32–90.82
star | B | 5.61 | 0.39 | 89.61 | 90.24 | 89.86–90.62
star | C | 1.51 | 0.73 | 89.27 | 90.67 | 89.98–91.37
star | D | 3.46 | 1.39 | 88.61 | 90.37 | 89.88–90.85
legislator | A | 10.35 | 0.31 | 89.69 | 90.08 | 89.80–90.36
legislator | B | 3.23 | 1.36 | 88.64 | 90.19 | 89.67–90.70
legislator | C | 1.47 | 1.00 | 89.00 | 90.50 | 89.75–91.25
legislator | D | 3.10 | 0.17 | 89.83 | 90.58 | 90.08–91.09
skin | A | 9.16 | 2.27 | 87.73 | 99.95 | 99.92–99.97
skin | B | 5.82 | 0.00 | 90.00 | 95.72 | 95.45–95.98
skin | C | 1.43 | 2.76 | 87.24 | 100.00 | 99.94–100.00
skin | D | 3.78 | 5.54 | 84.46 | 99.62 | 99.50–99.71

## Ordinary tree

Dataset | Cell | Selected (%) | Penalty (pp) | Lower bound (%) | Coverage (%) | 95% interval (%)
--- | --- | --- | --- | --- | --- | ---
bike | A | 7.77 | 0.34 | 89.66 | 90.18 | 89.84–90.51
bike | B | 2.88 | 0.15 | 89.85 | 90.47 | 89.95–90.99
bike | C | 1.67 | 0.40 | 89.60 | 90.83 | 90.14–91.51
bike | D | 3.40 | 0.55 | 89.45 | 90.53 | 90.03–91.02
bio | A | 15.32 | 0.01 | 89.99 | 90.10 | 89.87–90.34
bio | B | 9.51 | 0.01 | 89.99 | 90.15 | 89.86–90.45
bio | C | 1.54 | 0.02 | 89.98 | 89.96 | 89.24–90.69
bio | D | 36.05 | 0.01 | 89.99 | 89.94 | 89.79–90.09
community | A | 1.30 | 2.69 | 87.31 | 91.18 | 90.42–91.94
community | B | 0.91 | 1.56 | 88.44 | 91.47 | 90.62–92.33
community | C | 0.69 | 1.12 | 88.88 | 91.26 | 90.26–92.27
community | D | 1.18 | 1.59 | 88.41 | 91.40 | 90.65–92.14
concrete | A | 23.13 | 1.31 | 88.69 | 90.83 | 90.66–91.01
concrete | B | 5.94 | 1.51 | 88.49 | 90.94 | 90.58–91.29
concrete | C | 1.18 | 2.75 | 87.25 | 89.61 | 88.81–90.41
concrete | D | 5.34 | 0.25 | 89.75 | 92.80 | 92.47–93.14
homes | A | 21.64 | 0.03 | 89.97 | 89.84 | 89.64–90.04
homes | B | 11.61 | 0.01 | 89.99 | 90.05 | 89.77–90.32
homes | C | 2.32 | 0.12 | 89.88 | 90.03 | 89.42–90.63
homes | D | 2.79 | 0.04 | 89.96 | 90.52 | 89.97–91.06
star | A | 12.36 | 0.70 | 89.30 | 90.56 | 90.31–90.81
star | B | 5.62 | 0.48 | 89.52 | 90.22 | 89.84–90.61
star | C | 1.51 | 0.73 | 89.27 | 90.67 | 89.98–91.37
star | D | 3.47 | 1.58 | 88.42 | 90.39 | 89.90–90.87
legislator | A | 10.35 | 0.31 | 89.69 | 90.08 | 89.80–90.36
legislator | B | 3.23 | 1.36 | 88.64 | 90.19 | 89.67–90.70
legislator | C | 1.47 | 1.00 | 89.00 | 90.50 | 89.75–91.25
legislator | D | 3.10 | 0.17 | 89.83 | 90.58 | 90.08–91.09
skin | A | 9.46 | 3.15 | 86.85 | 99.93 | 99.90–99.96
skin | B | 5.54 | 5.11 | 84.89 | 95.78 | 95.51–96.04
skin | C | 1.46 | 2.90 | 87.10 | 99.97 | 99.88–100.00
skin | D | 4.11 | 10.98 | 79.02 | 99.58 | 99.47–99.67

## Leaf regions

Conditions within a row all apply. Cutpoints are rounded to six significant digits for display. Exact cutpoints and counts are in results.json.

Dataset | Cell | Conditions
--- | --- | ---
bike | A | holiday ≤ 0.5, workingday > 0.5, windspeed ≤ 25.9993, season_1 ≤ 0.5, season_2 ≤ 0.5, day ≤ 3, year > 0.5
bike | B | holiday ≤ 0.5, workingday > 0.5, windspeed ≤ 25.9993, season_1 ≤ 0.5, season_2 ≤ 0.5, weather_3 ≤ 0.5, day ≤ 3, year > 0.5
bike | C | holiday ≤ 0.5, workingday > 0.5, humidity > 50, windspeed ≤ 25.9993, season_1 ≤ 0.5, season_2 ≤ 0.5, weather_3 ≤ 0.5, year ≤ 0.5
bike | D | holiday ≤ 0.5, workingday > 0.5, windspeed ≤ 25.9993, season_1 > 0.5, weather_3 ≤ 0.5, year > 0.5
bio | A | F1 ≤ 17316.1, F9 > 35.3066
bio | B | F4 ≤ 107.999, F9 > 35.3066
bio | C | F1 ≤ 17316.1, F2 ≤ 6447.3, 0.213798 < F3 ≤ 0.335095, F4 ≤ 195.105, F5 ≤ 2.39341e+06, F6 ≤ 255.834, F9 ≤ 35.3066
bio | D | F9 ≤ 35.3066
community | A | medIncome > 0.525, pctWSocSec > 0.53
community | B | racePctHisp ≤ 0.5, medIncome ≤ 0.525, pctWFarmSelf ≤ 0.5, medFamInc ≤ 0.535, blackPerCap ≤ 0.5, MedRent ≤ 0.535, PctSameCity85 > 0.495, PctUsePubTrans ≤ 0.5
community | C | medIncome ≤ 0.525, medFamInc ≤ 0.535, blackPerCap ≤ 0.5, indianPerCap ≤ 0.5, PctNotSpeakEnglWell ≤ 0.5, MedRent ≤ 0.535, PctSameCity85 > 0.495, PctUsePubTrans ≤ 0.5
community | D | racePctAsian ≤ 0.5, medIncome ≤ 0.525, medFamInc ≤ 0.535, blackPerCap ≤ 0.5, PctNotSpeakEnglWell ≤ 0.5, MedRent ≤ 0.535, PctSameCity85 > 0.495, PctUsePubTrans ≤ 0.5
concrete | A | superplasticizer ≤ 16.1, fine aggregate ≤ 793.3
concrete | B | fly ash ≤ 48.75, superplasticizer ≤ 16.1, fine aggregate ≤ 793.3
concrete | C | superplasticizer ≤ 16.1, 793.3 < fine aggregate ≤ 892.95
concrete | D | superplasticizer > 16.1
homes | A | 6.75 < grade ≤ 8, long ≤ -121.915, sqft_lot15 ≤ 28007.1
homes | B | floors ≤ 2.25, 5.5 < grade ≤ 8, long ≤ -121.915, sqft_lot15 ≤ 28007.1
homes | C | floors ≤ 2.25, 6.75 < grade ≤ 8, lat > 47.4699, long ≤ -121.915
homes | D | 5.5 < grade ≤ 6.75, long ≤ -121.915, sqft_lot15 ≤ 28007.1
star | A | degreek ≤ 1.5, experiencek ≤ 13.5
star | B | degreek ≤ 1.5, ladderk ≤ 2.5
star | C | degreek ≤ 1.5, experiencek ≤ 13.5, tethnicityk ≤ 0.5
star | D | degreek ≤ 1.5, experiencek > 13.5
legislator | A | NOMINATE 1 ≤ 0
legislator | B | 0 < NOMINATE 1 ≤ 0.5, NOMINATE 2 ≤ 0
legislator | C | NOMINATE 1 > 0, NOMINATE 2 ≤ 0
legislator | D | NOMINATE 1 > 0.5
skin | A | follicular papules ≤ 1.5, 1.5 < knee and elbow involvement ≤ 2.75
skin | B | follicular papules > 1.5
skin | C | polygonal papules ≤ 1.5, follicular papules ≤ 1.5, 1.5 < knee and elbow involvement ≤ 2.75
skin | D | 1.5 < polygonal papules ≤ 2.75, follicular papules ≤ 1.5, knee and elbow involvement ≤ 1.5
