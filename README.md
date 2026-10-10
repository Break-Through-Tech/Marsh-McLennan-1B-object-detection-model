# Marsh McLennan Object Detection Model

### 👥 **Team Members**
---

| Name                  | GitHub Handle | Contribution                                                             |
|-----------------------|---------------|---------------------------------------------------------------------|
| Alina Dang            | @             |                 |
| Fadhili Mboya         | @             |                 |
| Gnana Varshita Chakka | @gc832        |                 |
| Kritin Ganesh         | @CactusKritin7220             |                 |
| Lizeth Fernandez      | @lfern101     |                 |
| Shiu Wong             | @             |                 |

---

## 🎯 **Project Highlights**

- Developing an NLP-based system to identify and classify objects mentioned in injury descriptions.
- Using injury narratives from the U.S. Consumer Product Safety Commission's National Electronic Injury Surveillance System (CPSC NEISS).
- Exploring a hybrid NLP approach that combines object extraction with product classification.
- The final system will be evaluated on anonymized workers' compensation accident descriptions provided by Marsh McLennan.
- Model performance will be evaluated using metrics such as precision, recall, F1 score, Micro F1, Macro F1, and Exact Match Accuracy.

---

## 👩🏽‍💻 **Setup and Installation**
  
### Clone the repository

```bash
git clone <repository-url>
cd Marsh-McLennan-1B-object-detection-model
```
---

### Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset
Our team is currently using the 2025 CPSC NEISS dataset across all incident locations.
The dataset can be downloaded from:
https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data 

Place the downloaded Excel file in:
data/neiss2025.xlsx 

Run the data exploration/cleaning script
```bash
python data/data_cleaning.py
```
This script currently creates a column inventory that helps us understand the structure of the NEISS dataset before preprocessing.

## 🏗️ **Project Overview**

**Describe:**

This project is part of the Break Through Tech AI Studio Fall 2026 program in partnership with Marsh McLennan.
The goal of the project is to develop an NLP-based system that can read accident or injury descriptions and identify the objects involved in the incident.
For example, given an injury narrative mentioning a chair, table, ladder, or other product, the system should be able to extract the object and classify it into the appropriate product category.
Marsh McLennan plans to evaluate the final system using anonymized workers' compensation accident descriptions. The goal is to explore whether NLP and large language model techniques can improve object detection in injury reports and support workplace safety and loss-prevention analysis.

---

## 📊 **Data Exploration**

We are using the 2025 CPSC NEISS dataset for our initial training data.
The dataset contains emergency room injury records, including:
- Injury narratives
- Product codes associated with each incident
- Incident location
- Patient and injury information
- Other coded fields describing the incident
For our project, the most important fields are the injury narratives and their associated product codes.
We decided to use the full 2025 dataset rather than narrowing the data to a specific incident location.

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

Current Data Exploration Work
So far, we have:
- Reviewed the overall dataset structure
- Identified fields relevant to object extraction and classification
- Created a column inventory showing data types, missing values, unique values, and sample values
- Begun planning the data-cleaning process
- Identified the need to connect product codes with their corresponding product names
Further preprocessing and exploratory analysis are currently in progress.

---

## 🧠 **Model Development**

Our current plan is to use a hybrid NLP approach.
The proposed pipeline has two main parts:
1. Object Extraction
   - Identify possible objects mentioned in an injury narrative
   - Explore tools such as spaCy and rule-based NLP techniques
2. Object Classification
   - Classify the extracted objects into the appropriate product categories
   - Explore machine learning and/or Large Language Model approaches
Using a hybrid approach allows us to separate the task of finding objects in the text from the task of deciding which product category they belong to.
We will experiment with different methods and refine the pipeline based on our validation results.

---

## 📈 **Results & Key Findings**

Model development and evaluation are currently in progress.
We plan to evaluate our system using:
- Precision
- Recall
- F1 Score
- Micro F1
- Macro F1
- Exact Match Accuracy
Results will be added as model development progresses.

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

Our next steps include:
- Clean and preprocess the 2025 NEISS dataset
- Match product codes with their corresponding product names
- Prepare injury narratives for NLP processing
- Develop an initial object extraction method
- Develop an initial object classification method
- Combine the components into a hybrid NLP pipeline
- Evaluate the first version of the system
- Refine the approach based on validation results

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

- U.S. Consumer Product Safety Commission. **National Electronic Injury Surveillance System (NEISS) Injury Data.**  
  https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

- spaCy. **spaCy 101: Everything You Need to Know.**  
  https://spacy.io/usage/spacy-101

- Natural Language Toolkit (NLTK). **NLTK Documentation.**  
  https://www.nltk.org/

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
