# Battery Dataset Remediation Report

*Cells processed: B0005, B0006, B0007, B0018 | Original rows: 636 | Final rows: 730*


## 1. Summary of All Calculated Parameters (Original)

| Column | Count | Mean | Median | Std | Min | Max | Skew | Kurt |
|--------|-------|------|--------|-----|-----|-----|------|------|
| capacity_ah | 636 | 1.5927 | 1.5690 | 0.1956 | 1.1733 | 2.0303 | 0.097 | -1.029 |
| cap_endpoint_ah | 636 | 1.5817 | 1.5597 | 0.1986 | 1.1538 | 2.0353 | 0.126 | -1.008 |
| mean_voltage_v | 636 | 3.4986 | 3.5025 | 0.0475 | 3.4039 | 3.5736 | -0.313 | -0.920 |
| min_voltage_v | 636 | 2.3877 | 2.4210 | 0.2197 | 1.7370 | 2.7000 | -0.480 | -0.565 |
| max_voltage_v | 636 | 4.1919 | 4.1906 | 0.0086 | 4.1576 | 4.2333 | -0.100 | 0.639 |
| mean_abs_current_a | 636 | 1.8341 | 1.8511 | 0.1026 | 1.5170 | 1.9998 | -0.602 | -0.119 |
| mean_temperature_c | 636 | 32.4371 | 32.3749 | 1.0662 | 30.2136 | 35.0195 | 0.165 | -0.839 |
| duration_s | 636 | 3116.9777 | 3084.2810 | 242.0067 | 2742.8430 | 3690.2340 | 0.418 | -0.881 |
| n_samples | 636 | 292.0142 | 309.0000 | 55.8737 | 179.0000 | 371.0000 | -0.868 | -0.605 |
| dt_s | 636 | 11.2482 | 9.3750 | 3.2683 | 9.3600 | 18.7180 | 1.596 | 0.938 |
| soh | 636 | 0.7963 | 0.7845 | 0.0978 | 0.5867 | 1.0151 | 0.097 | -1.029 |


## 2. Anomalies Detected

**Total flags: 164**

| Row | Column | Method | Value | Reason |
|-----|--------|--------|-------|--------|
| 89 | max_voltage_v | Z-score | 4.22292 | Z=3.62 > 3.0 |
| 257 | max_voltage_v | Z-score | 4.22252 | Z=3.58 > 3.0 |
| 287 | max_voltage_v | Z-score | 4.16357 | Z=3.30 > 3.0 |
| 317 | max_voltage_v | Z-score | 4.16424 | Z=3.22 > 3.0 |
| 334 | max_voltage_v | Z-score | 4.15755 | Z=4.00 > 3.0 |
| 425 | max_voltage_v | Z-score | 4.23333 | Z=4.84 > 3.0 |
| 331 | mean_abs_current_a | Z-score | 1.51704 | Z=3.09 > 3.0 |
| 332 | mean_abs_current_a | Z-score | 1.52387 | Z=3.02 > 3.0 |
| 333 | mean_abs_current_a | Z-score | 1.52380 | Z=3.02 > 3.0 |
| 327 | mean_abs_current_a | IQR | 1.55384 | Outside [1.5637, 2.1220] |
| 328 | mean_abs_current_a | IQR | 1.55221 | Outside [1.5637, 2.1220] |
| 329 | mean_abs_current_a | IQR | 1.54529 | Outside [1.5637, 2.1220] |
| 330 | mean_abs_current_a | IQR | 1.53872 | Outside [1.5637, 2.1220] |
| 334 | mean_abs_current_a | IQR | 1.53913 | Outside [1.5637, 2.1220] |
| 335 | mean_abs_current_a | IQR | 1.54271 | Outside [1.5637, 2.1220] |
| 0 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 1 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 2 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 3 | dt_s | IQR | 18.71800 | Outside [5.6604, 15.5659] |
| 4 | dt_s | IQR | 18.70300 | Outside [5.6604, 15.5659] |
| 5 | dt_s | IQR | 18.64900 | Outside [5.6604, 15.5659] |
| 6 | dt_s | IQR | 18.66450 | Outside [5.6604, 15.5659] |
| 7 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 8 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 9 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 10 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 11 | dt_s | IQR | 18.65650 | Outside [5.6604, 15.5659] |
| 12 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 13 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 14 | dt_s | IQR | 18.64050 | Outside [5.6604, 15.5659] |
| 15 | dt_s | IQR | 18.59300 | Outside [5.6604, 15.5659] |
| 16 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 17 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 18 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 19 | dt_s | IQR | 18.68800 | Outside [5.6604, 15.5659] |
| 20 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 21 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 22 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 23 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 24 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 25 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 26 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 27 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 28 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 29 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 42 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 168 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 169 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 170 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 171 | dt_s | IQR | 18.71800 | Outside [5.6604, 15.5659] |
| 172 | dt_s | IQR | 18.70300 | Outside [5.6604, 15.5659] |
| 173 | dt_s | IQR | 18.64900 | Outside [5.6604, 15.5659] |
| 174 | dt_s | IQR | 18.66450 | Outside [5.6604, 15.5659] |
| 175 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 176 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 177 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 178 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 179 | dt_s | IQR | 18.65650 | Outside [5.6604, 15.5659] |
| 180 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 181 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 182 | dt_s | IQR | 18.64050 | Outside [5.6604, 15.5659] |
| 183 | dt_s | IQR | 18.59300 | Outside [5.6604, 15.5659] |
| 184 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 185 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 186 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 187 | dt_s | IQR | 18.68800 | Outside [5.6604, 15.5659] |
| 188 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 189 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 190 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 191 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 192 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 193 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 194 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 195 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 196 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 197 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 210 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 336 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 337 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 338 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 339 | dt_s | IQR | 18.71800 | Outside [5.6604, 15.5659] |
| 340 | dt_s | IQR | 18.70300 | Outside [5.6604, 15.5659] |
| 341 | dt_s | IQR | 18.64900 | Outside [5.6604, 15.5659] |
| 342 | dt_s | IQR | 18.66450 | Outside [5.6604, 15.5659] |
| 343 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 344 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 345 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 346 | dt_s | IQR | 18.67950 | Outside [5.6604, 15.5659] |
| 347 | dt_s | IQR | 18.65650 | Outside [5.6604, 15.5659] |
| 348 | dt_s | IQR | 18.68700 | Outside [5.6604, 15.5659] |
| 349 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 350 | dt_s | IQR | 18.64050 | Outside [5.6604, 15.5659] |
| 351 | dt_s | IQR | 18.59300 | Outside [5.6604, 15.5659] |
| 352 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 353 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 354 | dt_s | IQR | 18.60900 | Outside [5.6604, 15.5659] |
| 355 | dt_s | IQR | 18.68800 | Outside [5.6604, 15.5659] |
| 356 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 357 | dt_s | IQR | 18.66400 | Outside [5.6604, 15.5659] |
| 358 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 359 | dt_s | IQR | 18.65700 | Outside [5.6604, 15.5659] |
| 360 | dt_s | IQR | 18.64100 | Outside [5.6604, 15.5659] |
| 361 | dt_s | IQR | 18.67200 | Outside [5.6604, 15.5659] |
| 362 | dt_s | IQR | 18.65600 | Outside [5.6604, 15.5659] |
| 363 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 364 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 365 | dt_s | IQR | 18.62500 | Outside [5.6604, 15.5659] |
| 378 | dt_s | IQR | 18.58650 | Outside [5.6604, 15.5659] |
| 0 | multivariate | IsolationForest | 1.85977 | Flagged as multivariate outlier (contamination=5%) |
| 1 | multivariate | IsolationForest | 1.84779 | Flagged as multivariate outlier (contamination=5%) |
| 47 | multivariate | IsolationForest | 1.79464 | Flagged as multivariate outlier (contamination=5%) |
| 168 | multivariate | IsolationForest | 2.03029 | Flagged as multivariate outlier (contamination=5%) |
| 169 | multivariate | IsolationForest | 2.01807 | Flagged as multivariate outlier (contamination=5%) |
| 170 | multivariate | IsolationForest | 2.00823 | Flagged as multivariate outlier (contamination=5%) |
| 171 | multivariate | IsolationForest | 2.00167 | Flagged as multivariate outlier (contamination=5%) |
| 172 | multivariate | IsolationForest | 2.00000 | Flagged as multivariate outlier (contamination=5%) |
| 173 | multivariate | IsolationForest | 2.00509 | Flagged as multivariate outlier (contamination=5%) |
| 174 | multivariate | IsolationForest | 2.00674 | Flagged as multivariate outlier (contamination=5%) |
| 175 | multivariate | IsolationForest | 1.96610 | Flagged as multivariate outlier (contamination=5%) |
| 187 | multivariate | IsolationForest | 1.97744 | Flagged as multivariate outlier (contamination=5%) |
| ... | ... | ... | ... | *44 more rows omitted* |


## 3. Fixes Applied

- Domain violations (24 rows) exceed 2% threshold — capping at physics limits instead of deletion.
- Capped 6 values in 'max_voltage_v' to IQR fence [4.1655, 4.2195].
- Capped 9 values in 'mean_abs_current_a' to IQR fence [1.5637, 2.1220].
- Capped 93 values in 'dt_s' to IQR fence [5.6604, 15.5659].


## 4. New Dataset Specifications

| Column | Mean (new) | Std (new) | Skew (new) | Kurt (new) |
|--------|-----------|-----------|-----------|-----------|
| capacity_ah | 1.5606 | 0.1969 | 0.275 | -0.993 |
| cap_endpoint_ah | 1.5856 | 0.1939 | 0.129 | -0.931 |
| mean_voltage_v | 3.4988 | 0.0471 | -0.314 | -0.834 |
| min_voltage_v | 2.3905 | 0.2124 | -0.329 | -0.957 |
| max_voltage_v | 4.1917 | 0.0085 | -0.237 | -0.347 |
| mean_abs_current_a | 1.8343 | 0.1003 | -0.527 | -0.282 |
| mean_temperature_c | 32.4371 | 1.0573 | 0.143 | -0.813 |
| duration_s | 3120.6046 | 239.2297 | 0.393 | -0.815 |
| n_samples | 659.3411 | 960.0851 | 2.233 | 3.075 |
| dt_s | 9.5352 | 3.9095 | -0.803 | 0.709 |
| soh | 0.7803 | 0.0984 | 0.275 | -0.993 |