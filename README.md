# From Basics to Applications: <br> A Machine Learning Curriculum for High School

A collection of machine learning programs designed as part of my Matura thesis to introduce high school students to fundamental ML concepts. This resource hub is designed to help educators introduce machine learning concepts to students in an interactive and accessible way!

## Introduction
This repository contains four machine learning projects covering supervised and unsupervised learning. The curriculum aims to make machine learning accessible for high school students, focusing on simplicity, clarity, and educational value.

---

## 📈 Linear Regression: Caffeine Intake vs. Participation

### Overview
This program uses **linear regression** to predict class participation based on **caffeine intake**. It introduces predictive modeling by fitting a line to data points to find the trend that minimizes error.

### Key Mathematics
The model finds the **line of best fit** defined by the linear equation:

$$y = mx + b$$

It optimizes the slope ($m$) and intercept ($b$) by minimizing the **Sum of Squared Errors (SSE)**:

$$\text{SSE}  = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

---

## 🌳 Decision Tree Classification: Study Hours vs. Exam Passing

### Overview
This program implements a **Decision Tree Classifier** to predict if a student passes an exam based on **scores** and **study hours**. It visualizes how the algorithm splits data recursively to create decision boundaries.

### Key Mathematics
**Entropy ($H$):**
Measures impurity in a dataset $S$:

$$H(S) = - \sum_{i} p_i \cdot \log_2(p_i)$$

**Information Gain ($IG$):**
Determines the best split by comparing parent entropy to the weighted entropy of children nodes ($S_c$):

$$IG(S) = H(S) - \sum_{c} \frac{|S_c|}{|S|} H(S_c)$$

---

## 🔍 k-Means Clustering: Homework Effort Analysis

### Overview
This program uses **k-Means Clustering** to categorise homework assignments into **low, medium, and high effort** based on difficulty and workload. It demonstrates how machines find hidden patterns in unlabelled data.

### Key Mathematics
The algorithm groups points by calculating the **Euclidean distance** between a data point ($x$) and a cluster center (centroid $\mu$):

$$d(x, \mu) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}$$

The centroid is updated iteratively by calculating the **mean** of all points in the cluster:

$$\mu_j = \frac{1}{n} \sum x_i$$

---

## 📝 Naive Bayes Text Prediction: Next Word Generator

### Overview
This program uses a **Naive Bayes** model to generate text in the style of Shakespeare. It calculates the probability of the next word occurring based on the current word, using temperature scaling to adjust creativity.

### Key Mathematics
The model calculates transition probabilities using **Bayes' Theorem**:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

To introduce variety, predictions are adjusted using **Temperature ($T$)** inside a Softmax function:

$$P'(w) = \frac{e^{\log{(P(w))}/T}}{\sum e^{\log{(P(w_i))}/T}}$$

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mattia-3rne/machine-learning-curriculum-for-high-school.git
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## 📂 Project Structure

* `decision_tree_classification.ipynb`: Decision Tree Classification.
* `naive_bayes_text_prediction.ipynb`: Naive Bayes text prediction.
* `k_means_clustering.ipynb`: k-Means Clustering.
* `linear_regression.ipynb`: Linear Regression.
* `requirements.txt`: Python dependencies.
* `README.md`: Project documentation.
* `Slides`: Teaching slides.
