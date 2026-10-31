# Job Market Intelligence Tool

A tool that finds the job postings that best fit your skills: it filters out non-tech postings with a trained classifier, scores each posting by the skills it asks for, and saves the top matches to Excel.

## Setup

1. Clone the repo and install the dependencies (Python 3.10+):
   ```bash
   git clone https://github.com/Heng2007/Job-Classifier.git
   cd Job-Classifier
   pip install -r requirements.txt
   ```
2. In `config.py`, set `JOBS_CSV_PATH` to your postings CSV. It needs the columns
   `external_id, company, title, description, url, fetched_at`.
3. To change how many postings you get, edit the last line of `analysis/rank_postings.py`:
   `fitted_excel(dataset, 30, True)`. `30` is the number of postings; `True` keeps only
   junior/intern roles, `False` keeps the rest.
4. Run from the project root:
   ```bash
   python -m analysis.rank_postings
   ```
5. Type the numbers of your skills, or press Enter to reuse the ones saved in `my_skills.json`.
6. Your best-fit postings are saved as an Excel file in `fitted_posts/`.

## Data Sources
|    source    | rows |                           description                                               |
|--------------|------|-------------------------------------------------------------------------------------|
| Greenhouse   | 4664 |           public Greenhouse Job Board API, picked 20 companies                      |
| Kaggle       | 1167 | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings, pick 1200 out of 120k|
| CLNx         | 805  |                   UofT 2025 summer work study program                               |       
## Methodology
1. How the labels for training were made: 220 lables are hand labeled by me, the rest of 6416 labels are labeled by LLM. `label_source` distinguishing them.
2. What the classifier was trained on: the title + description of all 6,636 labeled postings, turned into TF-IDF word features, with a stratified 80/20 train/test split into 5 categories. I trained a Logistic Regression and a PyTorch MLP (2 hidden layers: 256 → 64, dropout 0.5) and kept the MLP.
3. How I tuned it: Optuna ran 50 trials with 3-fold cross-validation on the training set, searching the TF-IDF settings (n-grams, min_df, vocabulary size) and the MLP settings (layers, units, dropout, learning rate, epochs). The best model (1 hidden layer of 314 units, dropout 0.23, 1–2 word n-grams, 10,000 features) raised test macro-F1 from 0.7507 to 0.7878.
4. How I built the skill extractor: a regex pattern for each of 67 skills in `analysis/skills_taxonomy.py`. I grew the list by reading postings and adding the skills it missed(read 16 posts total, 4 from each category).
5. How they are used together: `analysis/rank_postings.py` runs the classifier to drop non-tech postings (keeping any with a tech title like "software" or "data"), extracts each posting's skills, and scores it as the share of its skills I have. It then marks titles as Junior / Senior and saves the top matches to Excel.

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
* A third of `Research assistant` posts get misidentified as `Not relevant`

### Final model and ranking
1. Optuna tuning raised macro-F1 from 0.7507 to 0.7878, so the tuned MLP is the one `rank_postings` uses.
2. On a real internship CSV, the classifier labelled 36 of 67 tech-titled postings `Not relevant`, so postings with a tech title (software, data, ML, AI, engineer) are kept regardless.
3. Skills that would unlock the most near-miss postings for me(based on the skills I have, not applicable to everyone): AWS (121), Spark (112), Distributed Computing (103), C++ (101), LLM (79).


## Limitations
1. Since the original category, `NLP / LLM` and `Machain Learning`, are hard to differ. They are collapsed into one category, the cost is that the model can no longer find a job specicifically for `NLP / LLM` or specifically for `Machain Learning`. 
2. Only single digit number of `quant/finance` job present in the 6 thousands postings in training data, therefore the category is removed. The cost is that the model can no longer identify `quant/finance` jobs.
3. A keyword can match a word without matching a role, then the model got tricked by the mismatch. For example, a big part of positions involve labs were SpaceX manufacturing roles ("Chemical Lab Technician", "EEE Parts Lab Specialist"). Fixed it by replacing "lab" with a more specfic keywords.
4. The model can barely detect `Research Assistant` role, since the posts that in this category for training are all from a single place(CLNX)
5. Most of the training data (4,664 of 6,636 postings) comes from the 20 companies I picked on the Greenhouse API, so the model learned their wording and roles. It may be less accurate on postings from other companies or job boards than its test scores.
