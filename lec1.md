# notation:

**Training sample**: $S=\{(x_i, y_i)\}_{i=1}^m$, the training examples $(x, y) \in X \times \mathcal{Y}$ independently drawn from a identical distribution (i.i.d) $D$ defined on $X \times \mathcal{Y}, X$ is a space of inputs, $\mathcal{Y}$ is the space of outputs

**Model or hypothesis** $h: X \mapsto \mathcal{Y}$ that we use to predict outputs given the inputs $x$

**Loss function**: $L: \mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}, L(\ldots) \geq 0, L(y, y^{\prime})$ is the loss incurred when predicting $y^{\prime}$ when $y$ is true

Optimization procedure to find the hypothesis $h$ that minimize the loss on the training sample

**Empirical risk**: computing the average of the loss on individual instances:

$$
\hat{R}(h)=\frac{1}{m} \sum_{i=1}^m L(h(x_i), y_i)
$$

Different kinds of loss function:

squared loss is used in regression: $L_{s q}(y, y^{\prime})=(y^{\prime}-y)^2, y, y^{\prime} \in \mathbb{R}$

$0 / 1$ loss is used in classification: $L_{0 / 1}(y, y^{\prime})=\mathbf{1}_{y \neq y^{\prime}}$

Hamming loss is used in multilabel learning:

$$
L(y, y^{\prime})=\sum_{j=1}^d L_{0 / 1}(y_j, y_j^{\prime}), y, y^{\prime} \in\{-1,+1\}^d
$$

We would like to minimize the generalization error. or the true risk:

$$
R(h)=\mathbb{E}_{(\mathbf{x}, y) \sim D}[L(h(\mathbf{x}), y)],
$$

# Linear Regression

Training Data: $\{(x_i, y_i)\}_{i=1}^m,(x, y) \in \mathbb{R}^d \times \mathbb{R}$

Loss function: squared loss $L_{s q}(y, y^{\prime})=(y-y^{\prime})^2$

Hypothesis class: hyperplanes in $h(\mathbf{x})=\sum_{j=1}^d w_j x_j+w_0$

Model: $y=h(\mathbf{x})+\epsilon$, where $\epsilon$ is random noise corrupting the output.

We assume zero-mean normal distributed noise: $\epsilon \sim \mathcal{N}(0, \sigma^2)$, with unknown $\sigma$

Optimization: essentially, inverting a matrix (low polynomial time complexity)

Optimization problem:

$$
\begin{gathered}
\operatorname{minimize} \sum_{i=1}^m(y_i-\sum_{j=1}^d w_j x_{i j}+w_0)^2 \\
\text { w.r.t. } w_j, j=0, \ldots, d
\end{gathered}
$$

Write this in matrix form:

$$
\begin{aligned}
& \operatorname{minimize}(\mathbf{y}-\mathbf{X} \mathbf{w})^T(\mathbf{y}-\mathbf{X} \mathbf{w}) \\
& \text { w.r.t. } \mathbf{w} \in \mathbb{R}^{d+1}
\end{aligned}
$$

where:

$$
\text { where } \mathbf{X}=[\begin{array}{cc}
1 & \mathbf{x}_1 \\
\vdots & \vdots \\
1 & \mathbf{x}_i \\
\vdots & \vdots \\
1 & \mathbf{x}_m
\end{array}], \mathbf{w}=[\begin{array}{c}
w_0 \\
w_1 \\
\vdots \\
w_d
\end{array}] \mathbf{y}=[\begin{array}{c}
y_1 \\
\vdots \\
y_i \\
\vdots \\
y_m
\end{array}]
$$

Minimum this formula is attained when the derivatives w.r.t $\mathbf{w}$  go to 0:

$$
\begin{aligned}
& \frac{\partial}{\partial w}(\mathbf{y}-\mathbf{X} \mathbf{w})^T(\mathbf{y}-\mathbf{X} \mathbf{w}) \\
& \quad=\frac{\partial}{\partial w} \mathbf{y}^T \mathbf{y}-\frac{\partial}{\partial w} 2(\mathbf{X} \mathbf{w})^T \mathbf{y}+\frac{\partial}{\partial w}(\mathbf{X} \mathbf{w})^T \mathbf{X} \mathbf{w} \\
& \quad=-2 \mathbf{X}^T \mathbf{y}+2(\mathbf{X}^T \mathbf{X}) \mathbf{w}=0
\end{aligned}
$$

If $\mathbf{X}^T \mathbf{X}$ invertible, this formula can be solved by computing a pseudo-inverse matrix


# Binary Classification

The label is a binary variable

$$
y= \begin{cases}1 & \text { if } \mathbf{x} \ \\ 0 & \text { otherwise}\end{cases}
$$

For example we choose as the Hypothesis class $\mathcal{H}=\{h: X \mapsto\{0,1\}\}$ the set of axis prarllel rectangles in $\mathbb{R}^2$, that is 

$$
h(\mathbf{x})=(p_1 \leq x_1 \leq p_2) A N D(e_1 \leq x_2 \leq e_2)
$$

But we don't know the real classification $C$  so we cannot measure exactly how close h is to C

If a hypothesis correctly classifies all training examples we call it a **consistent hypothesis**

**Version space**:  the set of all consistenthypotheses of the hypothesis class

**Most general hypothesis** G:  cannot be expanded without including negative training examples

**Most specific hypothesis** S:  cannot be made smaller without excluding positive training points


# Model evaluation

zero-one loss: $L_{0 / 1}(y, y^{\prime})=\mathbf{1}_{y \neq y^{\prime}}$,   where

$$
\mathbf{1}_A= \begin{cases}1 & \text { if } A \text { is true } \\ 0 & \text { otherwise }\end{cases}
$$

not a good metric when class distributions are imbalanced or False Negative is costly

## Confusion matrix

True Positives: $m_{T P}=\mid\{\mathbf{x}_i.$ : $h(\mathbf{x}_i)=1$ and $.y_i=1\} \mid$

True Negatives: $m_{T N}=\mid\{\mathbf{x}_i.$ : $h(\mathbf{x}_i)=0$ and $.y_i=0\} \mid$

False Positives: $m_{F P}=\mid\{\mathbf{x}_i.$ : $h(\mathbf{x}_i)=1$ and $.y_i=0\} \mid$

False Negatives: $m_{F N}=\mid\{\mathbf{x}_i.$ : $h(\mathbf{x}_i)=0$ and $.y_i=1\} \mid$

and some evaluation metrics

Empirical risk (zero-one loss as the loss function):

$$
\hat{R}(h)=\frac{1}{m}(m_{F P}+m_{F N})
$$

Precision or Positive Predictive Value:

$$
\operatorname{Prec}(h)=\frac{m_{T P}}{m_{T P}+m_{F P}}
$$

Recall or Sensitivity:

$$
\operatorname{Rec}(h)=\frac{m_{T P}}{m_{T P}+m_{F N}}
$$

F1 score: $F_1(h)=2 \frac{\operatorname{Prec}(h) \cdot \operatorname{Rec}(h)}{\operatorname{Prec}(h)+\operatorname{Rec}(h)}=$

$$
\frac{2 m_{T P}}{2 m_{T P}+m_{F P}+m_{F N}}
$$

![1700140558815](image/lec1/1700140558815.png)

## Receiver Operating Characteristics (ROC) Curve

$x$-coordinate: False positive rate $F P R=m_{F P} / m$

$y$-coordinate: True positive rate $T P R=m_{T P} / m$

The ROC curve is created by plotting TPR and FPR at different thresholds. each threshold corresponds to a single point on the ROC curve, show the trade-off between TPR and FPR at various threshold levels..

The diagonal line from the bottom left to the top right represents a no-skill classifier (akin to random guessing).

TPR:

* **Intuitive Understanding** : Imagine a medical test for a specific disease. TPR answers the question: "Of all the people who actually have the disease, how many did we correctly diagnose as sick?"
* **Example** : If 100 people have a disease, and the test correctly identifies 80 of them as having the disease, the TPR is 80/100 = 0.8 or 80%. This means the test is quite good at catching cases of the disease.

FPR:

* **Intuitive Understanding** : Using the same medical test, FPR answers the question: "Of all the people who are actually healthy, how many did we incorrectly diagnose as sick?"
* **Example** : If there are 200 people who don't have the disease, but the test incorrectly identifies 40 of them as having the disease, the FPR is 40/200 = 0.2 or 20%. This means the test has a tendency to give a fair number of false alarms.

The higher the ROC curve goes, the better the algorithm or model

If two ROC curves cross it means neither model/algorithm is globally better

The area under the curve is called AUC

## Testing

We can compute an approximation of the true risk by computing theempirical risk on a independent test sample

$$
R_{\text {test }}(h)=\sum_{(\mathbf{x}_i, y_i) \in S_{\text {test }}}^m L(h(\mathbf{x}_i), y_i)
$$

The expectation of $R_{\text {test }}(h)$ is the true risk $R(h)$



\end
