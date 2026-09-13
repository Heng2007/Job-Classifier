# Job Market Intelligence Tool

A personal tool that classifies job postings, scores them against my current skills, and turns the gap into a "learn X → unlock N postings" plan.

## Setup

## Data Sources
|    source    | rows |                           description                                               |
|--------------|------|-------------------------------------------------------------------------------------|
| Greenhouse   | 4664 |           public Greenhouse Job Board API, picked 20 companies                      |
| Kaggle       | 1167 | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings, pick 1200 out of 120k|
| CLNx         | 805  |                   UofT 2025 summer work study program                               |       
## Methodology
1. How the labels for training were made: 220 lables are hand labeled by me, the rest of 6416 labels are labeled by LLM. `label_source` distinguishing them.
## Results
1. Changing from 8 categories to 5 categories effectivelly dropped the disgreement rate between me and LLM. 55% under 8 classes → 96% under 5 classes


### Logistic Regression

| class | precision | recall | f1-score | support |
|------------------------------|-----------|--------|----------|---------|
| Data engineering / analytics | 0.36 | 0.38 | 0.37 | 21 |
| Machine learning / AI | 0.93 | 0.83 | 0.88 | 52 |
| Not relevant | 0.97 | 0.96 | 0.97 | 1038 |
| Research assistant | 0.51 | 0.70 | 0.59 | 27 |
| Software engineering | 0.87 | 0.89 | 0.88 | 190 |
| **accuracy** | | | 0.93 | 1328 |
| **macro avg** | 0.73 | 0.75 | 0.74 | 1328 |
| **weighted avg** | 0.94 | 0.93 | 0.94 | 1328 |

### Multilayer Perceptron

| class | precision | recall | f1-score | support |
|------------------------------|-----------|--------|----------|---------|
| Data engineering / analytics | 0.47 | 0.38 | 0.42 | 21 |
| Machine learning / AI | 1.00 | 0.75 | 0.86 | 52 |
| Not relevant | 0.96 | 0.97 | 0.97 | 1038 |
| Research assistant | 0.61 | 0.63 | 0.62 | 27 |
| Software engineering | 0.88 | 0.90 | 0.89 | 190 |
| **accuracy** | | | 0.94 | 1328 |
| **macro avg** | 0.78 | 0.73 | 0.75 | 1328 |
| **weighted avg** | 0.94 | 0.94 | 0.94 | 1328 |

## Limitations
1. Since the original category, `NLP / LLM` and `Machain Learning`, are hard to differ. They are collapsed into one category, the cost is that the model can no longer find a job specicifically for `NLP / LLM` or specifically for `Machain Learning`. 
2. Only single digit number of `quant/finance` job present in the 6 thousands postings in training data, therefore the category is removed. The cost is that the model can no longer identify `quant/finance` jobs.
3. A keyword can match a word without matching a role, then the model got tricked by the mismatch. For example, a big part of positions involve labs were SpaceX manufacturing roles ("Chemical Lab Technician", "EEE Parts Lab Specialist"). Fixed it by replacing "lab" with a more specfic keywords.
