Real world data is usually missing values, which trip up a lot of machine learning algorithms. There are lots of tricks for dealing with these, but you have to be careful. The way in which you fill them can change the result dramatically. Being explicit and thoughtful about how you handle missing values will get you the very best results.

I've illustrated a large handful of approaches to missing values here in a fake data set. The data shows a group of employees, some of their personal data, and some data regarding an upcoming office party. In every case, knowing what the data means is the most important part of handling it well.

![Missing values experiment graph from Azure ML, top half][1]

1. Replace missing values with the mean. For this age data, it's reasonable to assume that missing values are distributed similarly to the values that are there. The formal name for this assumption is Missing at Random (MAR). In this case, substituting values 

2. Replace missing values with the median.

3. Replace missing values with an interpolated estimate.

4. Replace missing values with a constant.

5. Replace missing values using imputation. 

6. Replace missing values with a missing rank.

7. Replace missing values with a dummy value and create an indicator variable for "missing."

8. Replace missing values with 0.

9. Replace missing values with 0 and create an indicator variable for "missing."

    ![Missing values experiment graph from Azure ML, bottom half][2]

10. Replace missing values with a string.

11. Add an indicator variable showing which strings are considered "missing."

12. Delete columns that are missing too many values to be useful.

13. Delete rows that are missing critical values.

Here's an abbreviated link to this page: http://bit.ly/1jreneN


  [1]: https://raw.githubusercontent.com/brohrer-ms/public-hosting/master/missing_values_graph_top.png
  [2]: https://raw.githubusercontent.com/brohrer-ms/public-hosting/master/missing_values_graph_bottom.png