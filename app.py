import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns
import networkx as nx
import squarify
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import docx2txt
import textract
import plotly.express as px

st.title("Resume Classification")
# Create the tabs
tabs = st.tabs(["Resume Classify", "EDA", "Visualization","Models"])
# Resume classification tab
with tabs[0]:
    # Load the data
    data = pd.read_csv(r"C:\Users\sr000\Desktop\Resumes\data.csv")

    # Convert the "Resume" column to string
    data['Resume'] = data['Resume'].astype(str)

    # Encode the "Profile" column
    label_encoder = LabelEncoder()
    data['Profile'] = label_encoder.fit_transform(data['Profile'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Resume'], data['Profile'], test_size=0.2, random_state=44)

    # Create the select box
    model_list = ['Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'Bagging', 'Decision Tree', 'KNN Classifier']
    model = st.selectbox('Select the model:', model_list)

    # Create the file upload widget
    file = st.file_uploader('Upload a resume file (.doc, .docx, or .pdf)')

    # Function to extract text from a file
    def extract_text_from_file(file):
        file_type = file.name.split('.')[-1]
        if file_type == 'docx':
            text = docx2txt.process(file)
        elif file_type == 'pdf':
            text = textract.process(file, method='pdfminer').decode('utf-8')
        else:
            text = ''
        return text

    # If a file is uploaded, extract the text and classify the resume
    if file:
        resume_text = extract_text_from_file(file)

        # Apply the selected model
        if model == 'Naive Bayes':
            classifier = MultinomialNB()
        elif model == 'SVM':
            classifier = SVC()
        elif model == 'Logistic Regression':
            classifier = LogisticRegression()
        elif model == 'Random Forest':
            classifier = RandomForestClassifier()
        elif model == 'Gradient Boosting':
            classifier = GradientBoostingClassifier()
        elif model == 'AdaBoost':
            classifier = AdaBoostClassifier()
        elif model == 'Bagging':
            classifier = BaggingClassifier()
        elif model == 'Decision Tree':
            classifier = DecisionTreeClassifier()
        elif model == 'KNN Classifier':
            classifier = KNeighborsClassifier()

        # Vectorize the training data
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Vectorize the input text
        resume_text_vec = vectorizer.transform([resume_text])

        # Train the classifier
        classifier.fit(X_train_vec, y_train)

        # Predict the profile for the input resume text
        profile = classifier.predict(resume_text_vec)

        # Decode the profile label
        profile_label = label_encoder.inverse_transform(profile)

        st.write('The predicted profile is:', profile_label[0])
# EDA tab
with tabs[1]:
    # Perform some exploratory data analysis on the dataset
    st.write("EDA tab")
    # Load the dataset
    data = pd.read_csv(r"C:\Users\Acer\Desktop\Resumes\data.csv")
    # Data inspection
    st.title("Data Inspection")
    st.write("### Data Structure")
    st.write(len(data))
    st.write(data.shape)
    st.write("### Data Dimension")
    st.write(data.ndim)
    st.write("### Data Types")
    st.write(data.dtypes)
    st.write("### Checking Missing Values")
    st.write(data.isnull().sum())
    st.write(data.info())
    st.write(data.Profile.value_counts())    

# Visualization tab
with tabs[2]:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Load the dataset
    data = pd.read_csv(r"C:\Users\Acer\Desktop\Resumes\data.csv")

    # Count plot
    st.title("Count Plot")
    fig = plt.figure(figsize=(10, 6))  # Adjust the size of the plot as needed
    sns.countplot(y="Profile", palette="Set3", data=data, order=data['Profile'].value_counts().index)
    st.pyplot(fig)

    # Histogram
    st.title("Histogram")
    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(111)
    plt.hist(data['Profile'], bins=5, edgecolor='black', color='#4287f5')
    plt.xlabel('Profile')
    plt.ylabel('Numbers')
    plt.title('Distribution of Profiles')
    plt.grid(axis='y', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_facecolor('#f0f0f0')
    plt.tick_params(axis='both', which='both', bottom=False, left=False)
    st.pyplot(fig)

    # Scatter plot
    st.title("Scatter Plot")
    fig = plt.figure()
    plt.scatter(data.index, data.Profile)
    plt.xlabel('Index')
    plt.ylabel('Column Data')
    plt.title('Scatter Plot')
    st.pyplot(fig)
   
    # Network graph
    st.title("Network Graph")
    column_data = data["Profile"]
    G = nx.Graph()
    nodes = column_data.unique()
    G.add_nodes_from(nodes)
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    G.add_edges_from(edges)
    fig = plt.figure(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray')
    st.pyplot(fig)

    # Tree map
    st.title("Tree map")
    column1 = data["Profile"]
    value_counts = column1.value_counts(normalize=True)
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    fig = plt.figure(figsize=(10, 6))
    squarify.plot(sizes=value_counts, label=value_counts.index, color=colors, alpha=0.7)
    st.pyplot(fig)

    # Histogram
    st.title("Histogram")
    column1 = data["Profile"]
    fig = plt.figure(figsize=(10, 6))
    plt.hist(column1, bins=10)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    st.pyplot(fig)

    # Heat map
    st.title("Heat map")
    column1 = 'Profile'
    column2 = 'Resume'
    pivot_table = pd.pivot_table(data, values=column2, index=column1, aggfunc='count')
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='g')
    st.pyplot(fig)

# Models tab
with tabs[3]:
    # Train and evaluate some models on the dataset
    st.write("Models tab")

    # Classifier names
    classifier_names = [
         "Naive Bayes",
         "Support Vector Machine",
         "Logistic Regression",
         "Random Forest",
         "Gradient Boosting",
         "AdaBoost",
         "Bagging Classifier",
         "Decision Tree",
         "KNN Classifier",
     ]

    # Display the classifier names
    st.write("Classifiers:")
    for classifier_name in classifier_names:
        st.write(classifier_name)