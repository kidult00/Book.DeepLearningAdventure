Regression returns a numeric value.

Classification returns a state.

## Linear regression 线性回归
线性回归是最基础的回归类型，常用于做预测分析。

models relationship between independent & dependent variables via line of best fit. regression analysis assumes a dependence or causal relationship between one or more independent variables and one dependent variable.

Three major uses for regression analysis are
- causal analysis：identify the strength of the effect that the independent variable(s) have on a dependent variable.  what is the strength of relationship between dose and effect, sales and marketing spend, age and income.
- forecasting an effect：forecast effects or impact of changes.  how much the dependent variable change with a change in one or more independent variables.  Typical questions are, “how much additional Y do I get for one additional unit X?”
- trend forecasting：predicts trends and future values.  to get point estimates.  “what will the price for gold be in 6 month from now?”  “What is the total effort for a task X?”

类型：

- Simple linear regression 简单线性回归
1 dependent variable, 1 independent variable
- Multiple linear regression 多元线性回归
1 dependent variable, 2+ independent variables
- Logistic regression 逻辑回归
1 dependent variable, 2+ independent variable(s)
- Ordinal regression 序数回归
1 dependent variable, 1+ independent variable(s)
- Multinominal regression 多项式回归
1 dependent variable, 1+ independent variable(s)
- Discriminant analysis 判别分析
1 dependent variable, 1+ independent variable(s)

缺点：

- 只适用于本身是线性关系的数据
- 对 outliner 敏感

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588681bb_lin-reg-w-outliers/lin-reg-w-outliers.png)

Multiple Linear Regression 多元线性回归

如果需要预测的结果依赖于多个变量，可以用多元线性回归，比如：

$$y = m_1x_1 + m_2x_2 + b$$

Ref
- [What is Linear Regression? - Statistics Solutions](http://www.statisticssolutions.com/what-is-linear-regression/)
