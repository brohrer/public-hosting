# Quantile Regression: Car price prediction

Linear regression models generally predict the mean of the target column, given a set of input features. However, in some applications, such as price prediction, we are interested in predicting the range or entire distribution of the target column instead of a single estimate. To do this, you can use techniques such as Bayesian regression and quantile regression. Further, tree-based quantile regression models can predict non-parametric distributions.

In this experiment, we attempt to predict the 25th, 50th and 75th percentiles for the price for an automobile given some attributes.

![][image7]

## Data
The data consists of 205 examples, each with 26 attributes. We are interested in predicting the price of the automobile, given the other attributes in the dataset. More information about the dataset can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Automobile)

![][image1]

## Model
### Data Pre-processing
The first step is data preprocessing, using the following modules:

 1. The **Clean Missing Data** module is used to replace missing values with zeros.
 2. The data is split into 80% training and 20% testing sets using the **Split** module.

![][image2]

### Parameters
There are two ways that you can set parameters for the **Fast Forest Quantile Regression** module, using the **Create trainer mode**  option in the module properties:

 * Single Parameter: If you use this option, you specify parameters of the learner manually.
 * Parameter Range: If you use this option, you should obtain the parameters by using the **Sweep Parameters** module. Multiple values or a range of values can be specified for each tunable parameter. In this experiment, we used a _grid sweep_ to optimize the values for each of the tunable parameters.

__Required Quantile Values__: You can predict multiple quantile values using a single module. In this experiment, we will output the 25th, 50th and 75th percentile predictions from the same model.

![][image3]

### Sweep Parameters
The **Sweep Parameters** module is used to select the best combination of parameters from among the selected parameters and possible parameter values. Since we do not have a separate set of validation data, we used the training data to perform cross validation to optimize the parameters.

When predicting multiple quantiles, a separate loss function is associated with each quantile. The **Sweep Parameters** module optimizes for the average quantile loss across all the specified quantiles.

### Training
One of the outputs of the **Sweep Parameters** module is a learner with settings that have been optimized for the average quantile loss. This learner can be directly connected to the **Score Model** module to score the test set. Alternatively, we can take the learner with the optimal settings and retrain the model using all the training data. (In contrast, **Sweep Parameters** uses only a subset of the training data.) For this experiment, we choose the former approach.

### Scores
Predictions from the quantile regression model can be obtained using the generic **Score Model** module. The output provides an additional column with predictions for each of the quantiles specified.


![][image4]

We can extract the quantile predictions from the dataset of scores by using the **Project Columns** module.

![][image6]

## Evaluation Results
Quantile regression models optimize for _check loss_ or _quantile loss_. This loss is a generalization of the absolute error metric used in linear regression. More information about these loss functions can be found [here](http://en.wikipedia.org/wiki/Quantile_regression).

![][image5]

<!-- images -->
[image1]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image1.png
[image2]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image2.PNG
[image3]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image3.png
[image4]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image4.PNG
[image5]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image5.PNG
[image6]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image6.PNG
[image7]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Quantile_Regression_Car_price_prediction/image7.PNG
