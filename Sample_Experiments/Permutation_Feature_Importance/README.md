# Permutation Feature Importance #

In this experiment, we demonstrate how the **Permutation Feature Importance** module can be used to compute feature importance scores given a trained model and some test data.

The idea behind the algorithm is borrowed from the feature randomization technique used in Random Forests and described by Brieman in his seminal work [Random Forests](http://www.academia.edu/download/31115690/breiman_randomforests.pdf). Permutation Feature Importance computes importance scores for feature variables by determining the sensitivity of a model to random permutations of the values of those features. The main assumption is that permuting the values of important features results in a more significant reduction in a model's performance, compared to the effect of less important ones.

Given a trained model (regression or classification), a test dataset, and an evaluation metric, the **Permutation Feature Importance** module returns an ordered list of the feature variables and their corresponding importance scores.

## Running the Experiment ##
In the following section, we use the [Adult](http://archive.ics.uci.edu/ml/datasets/Adult) dataset, which is available in Azure Machine Learning Studio, to train a Support Vector Machine (SVM) binary classifier and compute its features' permutation importance scores. The results will give us insight on which feature columns contribute the most to the predictive accuracy of the model (how well we can predict the income level of an individual, given demographic and socioeconomic features).

### Experiment Details ###

1. Add the **Adult Census Income Binary Classification dataset** to your experiment.
2. Add a **Split** module to create a training and test datasets.
3. Add a **Two-Class Support Vector Machine** module to initialize the SVM classifier.
4. Add a **Train Model** module to train the classifier, and connect the SVM module to the left input port and the training dataset to the right input port. Using the column selector set the Label column to *income*.
5. Add a **Permutation Feature Importance** module and connect the trained model and the test dataset to the left and right input ports respectively. Set the **Metric for measuring performance** property to *Classification - Accuracy*.

The following figure shows the overall workflow of the experiment:

![][image0]

## Results ##

Run the experiment, click on the output port of **Permutation Feature Importance**, and select visualize to inspect the output results of the module. The following figure shows the list of features sorted in descending order of their permutation importance scores.

![][image1]

### Using other models ###
In addition to binary classification models, the **Permutation Feature Importance** module can also operate on multi-class classifiers or regression models. The process is exactly the same as described in the example above. For regression models, however, make sure you select a regression evaluation metric under the **Metric for measuring performance** property.

![][image2]

<!-- Images -->
[image0]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Permutation_Feature_Importance/image0.PNG
[image1]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Permutation_Feature_Importance/image1.PNG
[image2]:https://raw.githubusercontent.com/mezmicrosoft/Sample_Experiments/master/Permutation_Feature_Importance/image2.PNG
