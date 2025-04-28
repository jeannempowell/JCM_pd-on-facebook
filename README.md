# Analysis Pipeline for AIME2025 Submission

## Overview

This repository contains the analysis pipeline for the study:
“Evaluation of Facebook as a Longitudinal Data Source for Parkinson’s Disease Insights.”

The study explores the feasibility of using Facebook as a longitudinal data source for Parkinson’s disease (PD) research, leveraging natural language processing (NLP) techniques and machine learning classifiers to detect PD-related discourse in social media posts.

The pipeline covers:
- Text extraction from Facebook archives
- PD-related content identification using a Naïve Bayes classifier
- Pre- and post-diagnosis analysis of PD-related posts
- Evaluation of classifier performance and sensitivity analysis excluding exercise-related content

Note: Files constants.py and constants.R are omitted as they contain confidential API keys.

## Repository Structure

JCM_pd-on-facebook/

│── figures/

│── python/

│── r/

│── term_dictionary/

│── word_clusters/

│── README.md

## Figures

figures/ contains visualizations used in the manuscript:
	•	PD_Timeline.svg – Timeline of PD-related posts
	•	ROC_Best_NB_Model.pdf – ROC curve for the best Naïve Bayes model (Fig. 2 in the manuscript)
	•	facebook-timeline-2025-2-2-AIME2025.ai/pdf – Facebook posting history (Fig. 1 in the manuscript)

## Python Analysis Pipeline

python/ contains the core analysis pipeline, implemented in Jupyter Notebooks and Python scripts:

File	Description
- 1_extract_text_from_facebook.ipynb	Extracts text from Facebook JSON archives
- 1a_facebook_account_history.ipynb	Analyzes account history (duration, activity)
- 2_flag_posts_with_term_dictionary.ipynb	Flags posts using the PD-specific term dictionary
- 2a_irr.ipynb	Inter-rater reliability analysis for manual annotations
- 2b_manuallabelcount_termcount.ipynb	Compares manual labels with term-based counts
- 3_train-test-split.ipynb	Splits data for classifier training and testing
- 4_classifier-experiment.ipynb	Trains multiple classifiers and optimizes hyperparameters
- 5_apply-classifier.ipynb	Applies the best model to classify PD-relevant posts
- 6_pd-relevant-posts.ipynb	Analyzes PD-related posts before and after diagnosis
- _init_.py	Initializes the Python module
- functions.py	Helper functions for data processing, text cleaning, and classification

## R Analysis

r/ contains additional analyses related to dataset characteristics:

File	Description
- dataset_characteristics.Rmd	Summarizes demographic and clinical features
- facebook_users-vs-nonusers.Rmd	Compares Facebook users vs. non-users in the dataset
- functions.R	Helper functions for statistical analysis

## Term Dictionary

term_dictionary/ contains:
	•	term_dictionary.json – A custom dictionary of PD-related terms for keyword-based classification

## Word Clusters

word_clusters/ contains:
	•	50mpaths2.txt – Pretrained word clusters from Owoputi et al. (2013) used in NLP feature extraction

## Data Access

The Facebook data used in this study is not publicly available due to privacy concerns. If you wish to replicate parts of the analysis, you will need to provide your own text data in a similar format.

#3 Reproducing the Analysis

To follow the complete pipeline, execute the notebooks in the order listed under the Python Analysis Pipeline section.

1. Extract and Process Data

Run:

jupyter notebook python/1_extract_text_from_facebook.ipynb

This extracts and processes Facebook post text.

2. Classify PD-Relevant Posts

Train and evaluate classifiers:

jupyter notebook python/4_classifier-experiment.ipynb

Apply the best-performing classifier:

jupyter notebook python/5_apply-classifier.ipynb

3. Analyze PD-Related Discourse

Analyze pre- and post-diagnosis content:

jupyter notebook python/6_pd-relevant-posts.ipynb

## Key Findings

	•	PD-related discourse appears both before and after diagnosis.
	•	69% of individuals with PD explicitly mentioned PD, while 93% had at least one PD-related post.
	•	After excluding exercise-related posts, PD-related discourse before diagnosis decreased but was still present.
	•	The Naïve Bayes classifier achieved 86% recall and 94% AUC in identifying PD-related content.

For full details, see the manuscript in progress.

## Citing This Work

If you use this repository, please cite:
Powell, J.M., Cao, C., Means, K., Lakamana, S., Sarker, A., & McKay, J.L. Evaluation of Facebook as a Longitudinal Data Source for Parkinson’s Disease Insights. AIME 2025 (In Review).

## Acknowledgments

We thank our study participants for their time and data contributions. This research was supported by a Professional Development Support Funds Competitive Research Grant from Emory University’s Laney Graduate School.

## Contact & Contributions

For inquiries, please contact:
jeanne.marie.powell@emory.edu
