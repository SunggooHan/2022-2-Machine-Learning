# 22-2-ML-PHW2
### *Team 5*

## Structure and Flow
![그림1](https://user-images.githubusercontent.com/82069882/194892811-6247b946-ae6e-4034-8f29-5708bc2b49e7.png)

## Dataset
* link: https://www.kaggle.com/camnugent/california-housing-prices

## Prepare Dataset
1. Load original dataset<br>
2. Drop missing values<br>
3. Delete unnecessary attribute<br>
4. Label encoding for categorial attribute

## Define Search Space
1. Various scalers
2. Various models
3. Various combination of features

* We have 4 scalers
```python
scalers = [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]
```

* We use 4 models - [K-Means, GMM(EM), CLARANS, DBSCAN]
To get silhouette score and making plot with these models, we made these functions
```python
kmeans_silhouette_eblow(cluster_lists, X_features, scaler)
elbowPlot(scaler)
K_means_plot(dataset, cluster_lists, scaler)
gmm_cluster(dataset, scaler)
gmm_silhouette(cluster_lists, X_features, scaler)
plot_gmm(gmm, X, label=True, ax=None)
clustering_clarans(select_df)
findOptimalNClustersDB(dataset)
```
* Detailed setting of each functions
> **K_means_plot(dataset, cluster_lists, scaler)**<br>
  Parameter:<br>
          - cluster_lists:: List<br>
            The number of clusters in list<br><br>
          - X_features:: pd.Dataframe<br>
            The dataset to perform clustering<br><br>
          - scaler:: scalers in sklearn.preprocessing<br>
            The scaler to transform dataset<br>
<br>Describe: Visualize silhouette score after clustering and visualize elbow plot and KMeans clustering result<br><br>
          1. Clear the result array<br>
          2. KMeans clustering with defined hyperparameter<br>
          3. Calculate silhouette score<br>
          4. Visualize the result with bar graph<br>
          5. Repeat with given number of cluster in list<br>
          6. Call functions to visualize elbow plot and clustering result


> **elbowPlot(scaler)**<br>
Parameter:<br>
          - scaler:: scalers in sklearn.preprocessing<br>
            The scaler to transform dataset<br>
<br>Describe: Visualize elbow plot after clustering <br><br>
             1. Initialize with stored result<br>
             2. Visualize elbow plot


> **K_means_plot(dataset, cluster_lists, scaler)**<br>
Parameter:<br>
            - dataset:: pd.Dataframe<br>
              The dataset to perform clustering<br><br>
            - cluster_lists:: List<br>
              The number of clusters in list<br><br>
            - scaler:: scalers in sklearn.preprocessing<br>
              The scaler to transform dataset<br>
 <br>Describe: Visualize KMeans clustering result<br><br>
           1. Do PCA to plot graph<br>
           2. KMeans clustering with defined hyperparameter<br>
           3. Labeling the clusters, draw cirle at centers


> **gmm_cluster(dataset, scaler)**<br>
Parameter:<br>
            - dataset:: pd.Dataframe<br>
             The dataset to perform clustering<br><br>
            - scaler:: scalers in sklearn.preprocessing<br>
              The scaler to transform dataset<br>
<br>Describe: Visualize GMM clustering result and visualize silhouette score after clustering<br><br>
           1. Do PCA to plot graph<br>
           2. GMM clustering with defined hyperparameter<br>
           3. Plot the AIC score<br>
           4. Call function that calculate and visualize silhoutte score<br>
           5. Call function that visualize GMM result<br>


> **gmm_silhouette(cluster_lists, X_features, scaler)**<br>
 Parameter: <br>
            - cluster_lists:: List<br>
             The number of clusters in list<br><br>
            - X_features:: pd.Dataframe<br>
            The dataset to perform clustering<br><br>
            - scaler:: scalers in sklearn.preprocessing<br>
            The scaler to transform dataset<br>
<br>Describe: Visualize silhouette score after clustering<br><br>
           1. Clear the result array<br>
           2. GMM with defined hyperparameter<br>
           3. Visualize the result with bar graph


> **plot_gmm(gmm, X, label, ax)**<br>
 Parameter:<br>
            - GMM:: sklearn.mixture.GaussianMixture<br>
              GMM model to visualize<br><br>
            - X:: pd.Dataframe<br>
              Dataset used in GMM<br>
<br>Describe: Visualize GMM clustering result<br><br>
           1. Visualize the GMM clustering result


> **clustering_clarans(select_df)** <br>
 Parameter:<br>
            - select_df:: pd.Dataframe<br>
              The dataset to perform clustering<br>
<br>Describe: Visuzlize silhoutte score, distance using wce after CLARANS clustering<br><br>
           1. Clear the result array<br>
           2. CLARANS clustering with defined hyperparameter<br>
           3. Calculate silhouette score<br>
           4. Calculate elbow wce<br>
           5. Visualize the results<br>
           6. Repeat with given number of cluster in list


> **findOptimalNClustersDB(dataset)**<br>
 Parameter:<br>
           - dataset:: pd.Dataframe<br>
             The dataset to perform clustering<br>
<br>Describe: Visuzlize silhoutte score, result after DBSCAN<br>
           1. Clear the result array<br>
           2. DBSCAN with defined hyperparameter<br>
           3. Calculate silhouette score<br>
           4. Visualize the results<br>
           5. Repeat with given number of cluster in list


* Using Pearson Correlation of features, we select these 4 combinations
```python
predictor=[['total_rooms', 'total_bedrooms'],
           ['population', 'households'],
           ['total_rooms', 'total_bedrooms', 'households'],
           ['total_rooms', 'total_bedrooms', 'households','population']]
```

## Get Result
1. Initialize lists to save result
2. Set feature sets
3. Set scaler lists
4. Iterate over feature sets and scalers

## Optimal Clustering
* Silhouette score in case of ['total_rooms', 'total_bedrooms']

|Model|Finding Option|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|-----|
|K-Means|Best k|4|4|4|4|
|GMM|Best component|60|60|60|60|
|CLARANS|Best Silhouette cluster|3|3|3|3|
|DBSCAN|Best eps|0.05|0.05|0.05|0.05|

|Model|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|
|K-Means|0.516|0.515|0.515|0.515|
|GMM|0.379|0.379|0.379|0.379|
|CLARANS|0.875|0.877|0.877|0.873|
|DBSCAN|-0.3271|0.8973|0.8974|-0.1640|

=> **MinMax-DBSCAN** is well clustered

* Silhouette score in case of ['population', 'households']

|Model|Finding Option|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|-----|
|K-Means|Best k|4|4|4|4|
|GMM|Best component|60|60|60|60|
|CLARANS|Best Silhouette cluster|3|3|3|3|
|DBSCAN|Best eps|0.05|0.05|0.05|0.05|

|Model|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|
|K-Means|0.516|0.515|0.515|0.515|
|GMM|0.379|0.379|0.379|0.379|
|CLARANS|0.875|0.877|0.883|0.873|
|DBSCAN|-0.3271|0.8974|-0.1640|0.2035|

=> **MinMax-CLARANS** is well clustered

* Silhouette score in case of ['total_rooms', 'total_bedrooms', 'households']

|Model|Finding Option|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|-----|
|K-Means|Best k|4|4|4|4|
|GMM|Best component|60|70|80|70|
|CLARANS|Best Silhouette cluster|3|3|3|3|
|DBSCAN|Best eps|0.05|0.05|0.05|0.05|

|Model|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|
|K-Means|0.512|0.515|0.515|0.514|
|GMM|0.379|0.423|0.378|0.419|
|CLARANS|0.876|0.8907|0.8909|0.886|
|DBSCAN|-0.6317|0.8496|0.8495|-0.6596|

=> **MinMax-CLARANS** is well clustered

* Silhouette score in case of ['total_rooms', 'total_bedrooms', 'households','population']

|Model|Finding Option|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|-----|
|K-Means|Best k|4|4|4|4|
|GMM|Best component|80|80|80|80|
|CLARANS|Best Silhouette cluster|3|3|3|3|
|DBSCAN|Best eps|0.05|0.05|0.05|0.05|

|Model|Standard|MinAbs|MinMax|Robust|
|-----|-----|-----|-----|-----|
|K-Means|0.481|0.5|0.499|0.479|
|GMM|0.364|0.376|0.372|0.367|
|CLARANS|0.883|0.889|0.889|0.874|
|DBSCAN|-0.6478|0.8517|0.8540|-0.6707|

=> **MinMax-CLARANS** is well clustered

## Requirements
This project currently created using the following libraries.

|library|version|
|:-----:|-------|
|python|3.10.5|
|pandas|1.4.4|
|numpy|1.23.3|
|scipy|1.9.1
|scikit-learn|1.1.2|
|seaborn|0.11.2|
|pyclustering|0.10.1.2|

### Members - *TEAM 5*
* 202299178 Jo Byeong-wook
* 201835542 Han Sung-goo
* 202035509 Kim Ye-eun
* 202035535 Lee Ji-yun
