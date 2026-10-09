# Python port vs NetLogo: ensemble comparison

NetLogo table: `netlogo_table.csv`. Model years 2016-2049.
z = (mean_python - mean_netlogo) / standard error. Expected by chance: about 5 % of cells with |z| > 1.96. Cells where both models are always zero count as z = 0.

| case | scenario | seeds | total renovations NetLogo | Python | z | cells |z|>1.96 | cells |z|>3 |
|---|---|---|---|---|---|---|---|
| ES | Informative | 100 | 105.8 | 102.5 | -1.26 | 6.9 % | 1 of 408 |
| ES | No learning | 100 | 25.6 | 25.6 | +0.00 | 5.6 % | 1 of 408 |
| ES | Slow dynamics | 100 | 25.6 | 25.6 | +0.03 | 3.4 % | 0 of 408 |
| NL | Informative | 100 | 476.7 | 471.1 | -0.85 | 5.6 % | 2 of 408 |
| NL | No learning | 100 | 199.1 | 199.1 | +0.00 | 4.9 % | 2 of 408 |
| NL | Slow dynamics | 100 | 199.7 | 199.2 | -0.13 | 5.6 % | 3 of 408 |

All cells: 5.4 % with |z| > 1.96, 9 of 2448 with |z| > 3.

## ES / Informative (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 105.8 | 102.5 | 1.94 | 2039 |
| group1.a1 | 7.0 | 6.5 | 2.64 | 2038 |
| group2.a1 | 38.3 | 37.8 | 2.19 | 2019 |
| group3.a1 | 33.3 | 32.4 | 3.24 | 2035 |
| group4.a1 | 13.0 | 13.1 | 1.71 | 2031 |
| group5.a1 | 14.3 | 12.7 | 1.95 | 2038 |
| dwage1.a1 | 41.8 | 40.9 | 2.03 | 2018 |
| dwage2.a1 | 39.3 | 37.8 | 2.55 | 2018 |
| dwage3.a1 | 24.7 | 23.8 | 2.23 | 2039 |
| number.dwage1 | 12228.0 | 12253.3 | 2.66 | 2037 |
| number.dwage2 | 9665.0 | 9658.7 | 2.28 | 2037 |
| number.dwage3 | 5069.1 | 5050.0 | 2.42 | 2028 |

## ES / No learning (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 25.6 | 25.6 | 2.10 | 2031 |
| group1.a1 | 2.7 | 2.7 | 2.28 | 2034 |
| group2.a1 | 6.7 | 6.4 | 2.52 | 2031 |
| group3.a1 | 4.6 | 4.7 | 2.03 | 2045 |
| group4.a1 | 4.3 | 4.7 | 2.03 | 2042 |
| group5.a1 | 7.3 | 7.1 | 1.66 | 2049 |
| dwage1.a1 | 7.4 | 7.2 | 2.73 | 2047 |
| dwage2.a1 | 11.0 | 10.9 | 2.49 | 2029 |
| dwage3.a1 | 7.1 | 7.5 | 2.63 | 2035 |
| number.dwage1 | 12224.8 | 12258.3 | 1.98 | 2016 |
| number.dwage2 | 9670.1 | 9667.7 | 2.30 | 2036 |
| number.dwage3 | 5067.1 | 5036.0 | 3.48 | 2036 |

## ES / Slow dynamics (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 25.6 | 25.6 | 1.67 | 2030 |
| group1.a1 | 2.7 | 2.7 | 1.75 | 2042 |
| group2.a1 | 6.7 | 6.4 | 1.87 | 2030 |
| group3.a1 | 4.6 | 4.8 | 1.79 | 2034 |
| group4.a1 | 4.3 | 4.7 | 1.75 | 2043 |
| group5.a1 | 7.2 | 7.1 | 1.18 | 2049 |
| dwage1.a1 | 7.2 | 7.1 | 1.71 | 2049 |
| dwage2.a1 | 11.2 | 11.1 | 1.57 | 2035 |
| dwage3.a1 | 7.1 | 7.4 | 2.18 | 2040 |
| number.dwage1 | 12225.8 | 12251.4 | 1.98 | 2016 |
| number.dwage2 | 9668.4 | 9660.3 | 2.26 | 2047 |
| number.dwage3 | 5067.8 | 5050.3 | 2.07 | 2028 |

## NL / Informative (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 476.7 | 471.1 | 2.25 | 2039 |
| group1.a1 | 13.0 | 12.3 | 1.73 | 2028 |
| group2.a1 | 138.5 | 139.1 | 1.97 | 2032 |
| group3.a1 | 205.2 | 198.6 | 2.43 | 2030 |
| group4.a1 | 79.6 | 81.1 | 2.09 | 2037 |
| group5.a1 | 40.4 | 39.9 | 2.95 | 2038 |
| dwage1.a1 | 147.3 | 144.5 | 2.61 | 2031 |
| dwage2.a1 | 150.3 | 148.3 | 2.13 | 2039 |
| dwage3.a1 | 179.1 | 178.3 | 2.56 | 2041 |
| number.dwage1 | 11090.0 | 11101.2 | 4.11 | 2042 |
| number.dwage2 | 8504.6 | 8501.8 | 3.73 | 2042 |
| number.dwage3 | 6211.4 | 6203.0 | 2.71 | 2029 |

## NL / No learning (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 199.1 | 199.1 | 2.59 | 2038 |
| group1.a1 | 3.7 | 3.5 | 2.23 | 2027 |
| group2.a1 | 38.8 | 41.9 | 2.16 | 2024 |
| group3.a1 | 97.0 | 94.5 | 2.28 | 2028 |
| group4.a1 | 43.4 | 41.9 | 3.11 | 2038 |
| group5.a1 | 16.2 | 17.2 | 2.15 | 2020 |
| dwage1.a1 | 40.6 | 39.0 | 2.13 | 2042 |
| dwage2.a1 | 64.8 | 64.7 | 2.61 | 2039 |
| dwage3.a1 | 93.7 | 95.4 | 1.91 | 2031 |
| number.dwage1 | 11107.6 | 11116.9 | 2.54 | 2030 |
| number.dwage2 | 8489.2 | 8500.6 | 3.69 | 2030 |
| number.dwage3 | 6209.2 | 6188.6 | 2.20 | 2038 |

## NL / Slow dynamics (100 seeds)

| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |
|---|---|---|---|---|
| a1 | 199.7 | 199.2 | 2.84 | 2037 |
| group1.a1 | 3.8 | 3.5 | 1.75 | 2046 |
| group2.a1 | 39.1 | 42.0 | 2.37 | 2046 |
| group3.a1 | 97.1 | 94.5 | 2.64 | 2037 |
| group4.a1 | 43.5 | 41.9 | 2.01 | 2018 |
| group5.a1 | 16.3 | 17.2 | 2.59 | 2032 |
| dwage1.a1 | 41.4 | 38.9 | 2.97 | 2037 |
| dwage2.a1 | 65.0 | 64.3 | 3.12 | 2043 |
| dwage3.a1 | 93.3 | 96.0 | 1.70 | 2039 |
| number.dwage1 | 11109.6 | 11102.7 | 3.02 | 2043 |
| number.dwage2 | 8492.2 | 8500.3 | 2.53 | 2043 |
| number.dwage3 | 6204.1 | 6202.9 | 1.91 | 2038 |

