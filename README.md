# Linear-Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It serves as a fundamental tool in predictive modeling, providing insights into the quantitative associations among variables. 

Simple Linear regression
Simple linear regression is a statistical method used to model the relationship between a single independent variable (predictor) and a dependent variable (response). The goal is to establish a linear equation that best represents the relationship between the variables. The equation takes the form:
y=mx+b

Where:
y is the dependent variable (response),

x is the independent variable (predictor),

m is the slope of the line, representing the change in y for a one unit change in x

b is the y-intercept, representing the value of y when x is 0

Multiple Linear regression : 
Multiple linear regression is a statistical method used to model the relationship between a dependent variable and two or more independent variables. In multiple linear regression, the goal is to find the best-fitting linear equation that explains the variation in the dependent variable based on the multiple independent variables. The general form of the multiple linear regression equation is:

y= b0+b1x1+b2x2+ … + bnxn+ε

Where:
y is the dependent variable,
x1, x2, ….xn are the independent variables
b0 is the y-intercept,
b1, b2, …, bn are the coefficients associated with each independent variable,ε represents the error term

Assumptions of Linear Regression:
1. Linearity:
Assumption: The relationship between the dependent variable and independent variable(s) is linear.
Rationale: This assumption posits that the effect of a unit change in the independent variable(s) is constant across all levels.

Solutions to achieve linearity: Explore transformations, such as polynomial regression, or employ spline functions to capture non-linear patterns

2. No Autocorrelation:
Assumption: Residuals (the differences between observed and predicted values) are independent of each other.
Rationale: Independence ensures that the occurrence of an error in predicting the dependent variable at one point does not provide information about errors at other points. Violations of independence can lead to biased estimates and incorrect inferences.
Durbin Watson test : Interpretation: Values around 2 suggest no autocorrelation. Deviations may indicate positive or negative autocorrelation.

3. Homoscedasticity:
Assumption: Residuals exhibit constant variance across all levels of the independent variable(s).
Rationale: Homoscedasticity is crucial to maintain the efficiency and reliability of parameter estimates. Heteroscedasticity, or varying levels of variance, can lead to biased standard errors and impact the precision of statistical tests.
Solution: Consider data transformations, such as log transformations, or use weighted least squares regression to address heteroscedasticity.
4. Normality of Residuals:
Assumption: Residuals are normally distributed.
Rationale: Normality is essential for valid hypothesis testing and constructing reliable confidence intervals. While the central limit theorem suggests that the distribution of residuals becomes normal as the sample size increases, assessing normality directly is still important, especially for smaller samples.
Solution: Apply transformations to the dependent variable or use robust regression techniques that are less sensitive to non-normality.
5. No Perfect Multicollinearity:
Assumption: Independent variables are not perfectly correlated.
Rationale: Perfect multicollinearity, where one independent variable can be exactly predicted by another, makes it challenging to estimate the individual effects of variables. It can lead to inflated standard errors and imprecise parameter estimates.
Solution: Identify highly correlated variables using VIFs and consider removing or combining variables to mitigate multicollinearity.
Model Validation Techniques:
1. Residual Analysis:
Method: Examine residual plots, such as scatterplots, against predicted values or fitted values.
Purpose: Identify patterns, outliers, or non-linearity in residuals.
2. Hypothesis Testing:
Method: Conduct hypothesis tests for individual coefficients and the overall model (F-test).
Purpose: Assess the statistical significance of predictors and the model as a whole.
3. Coefficient of Determination (R2)
Definition: Measures the proportion of the variance in the dependent variable that is explained by the independent variables in the model
Interpretation: R2 values range from 0 to 1, where a higher value indicates a better fit. It's essential to consider R2 along with other metrics
4. Adjusted R2: 
 Definition: A modified version of R2 that adjusts for the number of predictors in the model. 
Interpretation: Helps prevent overfitting by penalizing the inclusion of unnecessary variables

5. Mean Squared Error (MSE) or Residual Sum of Squares (RSS):
Definition: Measures the average squared difference between observed and predicted values.
Interpretation: Lower MSE or RSS values indicate better model performance. MSE is sensitive to outliers.


6.Root Mean Squared Error (RMSE):
Definition: The square root of the MSE, providing an interpretable scale in the same units as the dependent variable.
Interpretation: Similar to MSE, lower RMSE values indicate better predictive performance.

