# From Basics to Applications: <br> A Machine Learning Curriculum for High School
 


📘 A collection of machine learning programs designed as part of my Matura thesis to introduce high school students to fundamental ML concepts. Explore the **High School Machine Learning Curriculum** I developed, hosted on Notion. This resource hub is designed to help educators introduce machine learning concepts to students in an interactive and accessible way:
[![Notion Site](https://img.shields.io/badge/Notion-View%20Site-black?logo=notion)](https://mattia-erne.notion.site/High-School-Machine-Learning-Curriculum-A-Resource-Hub-For-Educators-13bd0dc06fea806fa924d097bf0de1a6)



## Introduction
This repository contains four machine learning projects developed to provide hands-on examples of core machine learning algorithms. The programs cover supervised and unsupervised learning, offering practical insights into how algorithms operate on data.

The curriculum aims to make machine learning accessible for high school students, focusing on simplicity, clarity, and educational value. Each program includes essential comments to help students and teachers understand the code and underlying concepts.

---

## 1. 📈 Linear Regression: Caffeine Intake vs. Participation

### Overview:  
This program uses **linear regression** to predict class participation based on **caffeine intake**, illustrating how machine learning can model relationships between variables. By fitting a line to data points, the project provides an intuitive introduction to **predictive modeling** and **trend analysis**.  

Students learn key concepts like the **sum of squared errors (SSE)** for measuring accuracy and the **least squares method** for calculating the best fit. The program features scatter plot visualisations and allows users to input caffeine levels to predict participation, offering a hands-on and engaging way to explore machine learning fundamentals.

### Concepts Covered:
  - **Sum of Squared Error (SSE):** Measures the total deviation between the actual and predicted values, representing how well the regression line fits the data.  
  - **Model Fitting:** The least squares method calculates the optimal slope and intercept to minimize prediction errors.  
  - **Visualisation:** A regression line is plotted alongside the data points, providing a clear graphical representation of how the model interprets trends in the data.  
  - **Prediction and User Interaction:** The program allows users to input their own caffeine intake to receive a predicted participation level, fostering engagement and real-time exploration.  

<details>
 <p></p>
  <summary>How to Run the Program:</summary>
    
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/mattia-3rne/MA_MachineLearningCurriculum.git
2. **Navigate to the Project Directory:**  
   ```bash
   cd MA_MachineLearningCurriculum
3. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook
4. **Open the Notebook:**  
   ```bash
    MA_LinearRegression.ipynb
5. **Install Dependencies:**   
   ```bash
   pip install numpy pandas matplotlib notebook

</details>

---

## 2. 🌳 Decision Tree Classification: Study Hours vs. Exam Passing

### Overview:
This program implements a **Decision Tree Classifier** to predict whether a student passes an exam based on their **average score** and **hours studied**. By modeling decision boundaries through recursive data splitting, the algorithm visualises how machine learning can handle **classification tasks** in an intuitive and interpretable way.

The project focuses on **entropy** and **information gain** to evaluate how well a particular feature separates the data, providing a clear and practical introduction to core decision tree concepts. This makes it an ideal hands-on experience for students learning about classification models and decision-making in machine learning.

### Concepts Covered:
  - **Entropy:** Measures the level of uncertainty or disorder in the data. The goal of the model is to reduce entropy through optimal splits.  
  - **Information Gain:** Represents the reduction in entropy after splitting the data. It helps the model identify the most effective feature and threshold for each decision point.  
  - **Visualisation:** The decision tree's structure and decision boundaries are visualised alongside the training data, making the algorithm's logic more accessible.  
  - **Threshold Exploration:** The program explores different thresholds for features to identify the best point to split the data, reinforcing the concept of recursive partitioning.
  

<details>
 <p></p>
  <summary>How to Run the Program:</summary>
    
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/mattia-3rne/MA_MachineLearningCurriculum.git
2. **Navigate to the Project Directory:**  
   ```bash
   cd MA_MachineLearningCurriculum
3. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook
4. **Open the Notebook:**  
   ```bash
    MA_DecisionTreeClassifier.ipynb
5. **Install Dependencies:**   
   ```bash
   pip install numpy pandas matplotlib seaborn notebook

</details>

---

## 3. 🔍 k-Means Clustering: Homework Effort Analysis

### Overview:
This program implements **k-means clustering** to classify homework assignments into three categories based on **difficulty** and **amount of work**. By clustering similar data points, the algorithm provides an intuitive way to segment tasks into **low, medium, and high effort** levels.

The project offers a practical application of unsupervised learning, demonstrating how clustering can identify hidden patterns in data without labeled examples. This hands-on experience introduces students to the core principles of **data segmentation and clustering algorithms**.

### Concepts Covered:
  - **Clustering:** Demonstrates how data points are grouped into clusters based on similarity. The program divides homework tasks into effort-based categories. 
  - **k-Means Algorithm:** Uses the k-means method to iteratively find cluster centers that minimize the variance within each cluster.
  - **Cluster Visualisation:** The program plots the clusters and visualises their boundaries, providing a clear representation of how the model organizes data. 
  - **Sorting and Labeling:** After clustering, the program reorders the clusters to reflect increasing effort, ensuring the output is interpretable and meaningful.
  

<details>
 <p></p>
  <summary>How to Run the Program:</summary>
    
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/mattia-3rne/MA_MachineLearningCurriculum.git
2. **Navigate to the Project Directory:**  
   ```bash
   cd MA_MachineLearningCurriculum
3. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook
4. **Open the Notebook:**  
   ```bash
    MA_K-MeansClustering.ipynb
5. **Install Dependencies:**   
   ```bash
   pip install numpy matplotlib scikit-learn notebook

</details>

---

## 4. 📝 Naive Bayes Text Prediction: Next Word Generator

### Overview:
This program implements a **Naive Bayes-based text prediction model** to generate the next words in a sequence based on a given starting sentence. The model uses **Bayes' Theorem** to calculate transition probabilities between words, predicting future words based on prior context. The project applies **temperature scaling** to adjust the randomness of predictions, offering a simple yet effective introduction to **natural language processing (NLP)** concepts.

By analysing a sample text (such as the script of Romeo and Juliet), the program builds a probabilistic model capable of generating coherent sentences, mimicking the behavior of language models. This project helps students understand the fundamentals of **probabilistic text generation** and **word prediction**, bridging the gap between basic machine learning algorithms and more advanced NLP models.
### Concepts Covered:
  - **Bayes' Theorem:** The model predicts the likelihood of a word appearing after another by applying conditional probability to word transitions.
  - **Bigram Model:** The program counts pairs of consecutive words (bigrams) to build transition probabilities, reflecting how often one word follows another.
  - **Temperature Scaling:** Adjusts the certainty of predictions, allowing control over the model's randomness and creativity
  - **Markov Chain Text Generation:** Words are predicted sequentially, forming a chain of probabilities that drive the text generation process.
  

<details>
 <p></p>
  <summary>How to Run the Program:</summary>
    
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/mattia-3rne/MA_MachineLearningCurriculum.git
2. **Navigate to the Project Directory:**  
   ```bash
   cd MA_MachineLearningCurriculum
3. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook
4. **Open the Notebook:**  
   ```bash
    MA_NaiveBayesTextPrediction.ipynb
5. **Install Dependencies:**   
   ```bash
   pip install numpy matplotlib notebook  

</details>

---


