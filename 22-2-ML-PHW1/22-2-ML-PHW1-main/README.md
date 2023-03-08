# [22-2] Machine Learning PHW#1 
### _TEAM 5_

# Structure & Flow
<img width="874" alt="Screen Shot 2022-10-10 at 8 05 21 PM" src="https://user-images.githubusercontent.com/65584699/194852404-3fdc1a33-6b5f-48ed-a817-e17db06692b3.png">

# Dataset
* link : <https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>

# Prepare Dataset
1. Load original dataset
2. Set column names
3. Cleaning dataset (descibed in LAB#1)
4. Split into predictor / target

# Define Search Space
1. Various scalers
2. Various models
3. Various model parameters

- We have five scaler states.
````python
scalers = [None, StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
````

- Detailed setting of models and parameters:
`searchspace = [...]`  

# Find best model / parameter
 Using 'Pipeline' and 'RandomizedSearchCV'
1. Build pipeline that sequentially apply a list of transforms and a final estimator (Scaler -> Classifier)
2. Using RandomizedSearchCV, build & fit for parameter combinations randomly
3. Above steps will repeated by cross-validation generator with different folds

# Visualize result
1. From the result, find best model for each fold
2. Visualize

# output (visualize)
`k`is define for `KFold`

**k = 2**|**k = 3**
-----|-----
<img width="738" alt="Screen Shot 2022-10-10 at 7 54 17 PM" src="https://user-images.githubusercontent.com/65584699/194851019-2577644a-90c1-4c3c-a4fa-f50b653ee5a8.png">|<img width="738" alt="Screen Shot 2022-10-10 at 7 54 28 PM" src="https://user-images.githubusercontent.com/65584699/194851007-fd032d87-4ad7-4bd7-a3f7-df5128ac98a7.png">

**k = 4**|**k = 5**
-----|-----
<img width="738" alt="Screen Shot 2022-10-10 at 7 54 43 PM" src="https://user-images.githubusercontent.com/65584699/194850999-9d1b0605-822c-4917-bbec-c271bade81b7.png">|<img width="738" alt="Screen Shot 2022-10-10 at 7 54 52 PM" src="https://user-images.githubusercontent.com/65584699/194850985-d6475b1a-8602-40d1-9ebe-83ce66124c54.png">




## Requirements

This project is currently created using the following libraries.

| library | version |
| ------ | ------ |
| python | 3.10.5 |
| pandas | 1.4.4 |
| numpy | 1.23.3 |
| scipy | 1.9.1 |
| scikit-learn | 1.1.2 |
| matplotlib | 3.5.3 |

### Members - _TEAM 5_
- 202299178 Jo Byeong Wook
- 201835542 Han sunggoo
- 202035509 kim yeeun
- 202035535 Lee jiyun
