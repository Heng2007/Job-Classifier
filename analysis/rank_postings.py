r"""Rank any table of job postings by how well they fit my skills.

Responsible for: taking a DataFrame of postings from any source, detecting the
skills each one asks for, scoring how much of that set I already have, and
returning the postings ranked best-fit first. Also the two filters that decide
which postings are worth ranking at all: the relevance classifier and the
junior-eligibility title filter.

Outputs: a copy of the input with columns added:
  skills        set of skills detected in title + description
  matched       skills I have
  missing       skills it wants that I don't have
  match_score   len(matched) / len(skills); 0 when no skills were detected
  category      predicted category from the tuned MLP
sorted by match_score, then by len(matched) so a posting that names one skill
does not outrank one that names five I all have.

"""
import re
import sys
from datetime import datetime
import json
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis.skills_taxonomy import extract_skills
import config

mlp = joblib.load(config.TUNED_MLP_MODEL_PATH)
vectorizer = joblib.load(config.TUNED_VECTORIZER_PATH)
encoder = joblib.load(config.LABEL_ENCODER_PATH)
october= pd.read_csv(config.JOBS_CSV_PATH)

#import my skills and convert it into list
with open(config.SKILLS_PROFILE_PATH) as f:
    load = json.load(f)
my_skill_list = load["known_skills"]


def rank_postings(df: pd.DataFrame, my_skills: list):
    """external_id: a unique ID for the posting, used to remove duplicates and to skip jobs you've already seen.
       company: the employer's name.
       title: the job title.
       description: the full job description, in plain text with simple Markdown formatting.
       url: a link to the posting on Indeed or LinkedIn.
       fetched_at: when the script scraped it."""
    df["skills"] = df["description"].map(lambda skills: extract_skills(str(skills)))
    #copied from match_scoring
    df["match_score"] = df["skills"].apply(
        lambda skills: len(set(skills) & set(my_skills)) / len(skills) if skills else 0)
    df["matched"] = df["skills"].apply(lambda skills: len(set(skills) & set(my_skills)))
    df.sort_values(["match_score", "matched"], ascending=False, inplace=True)

def categorize(df:pd.DataFrame, mlp: torch.nn.Module, vectorizer: TfidfVectorizer, encoder: LabelEncoder ):
    """add a categorized column to the original dataframe"""
    fitted_test = vectorizer.transform(df["title"]+" "+df["description"])
    tensor_test = torch.tensor(fitted_test.todense(), dtype= torch.float32)
    mlp.eval()
    with torch.no_grad():
        prediction = mlp(tensor_test).argmax(dim = 1)
    encoded_result = encoder.inverse_transform(prediction)
    df["category"] = encoded_result
    df["relevant"] = (df["category"] != config.NOT_RELEVANT_CATEGORY) | df["title"].str.contains(config.TECH_TITLE_PATTERN, case=False)


def junior_filter(df:pd.DataFrame):
    """give each post a lable fo Junior, Senior, or Not Verified """
    df["pattern"] = df["title"].map(lambda title: "Senior" if re.search(config.SENIOR_TITLE_PATTERN, str(title), re.IGNORECASE) 
                                    else ("Junior" if re.search(config.JUNIOR_TITLE_PATTERN, str(title), re.IGNORECASE) 
                                          else "Not Verified"))

def fitted_excel(df: pd.DataFrame, top_postings: int, Junior: bool):
  categorize(df, mlp, vectorizer, encoder)
  rank_postings(df, my_skill_list)
  junior_filter(df)
  df = df.sort_values(["matched", "match_score"], ascending=False)
  if Junior:
      df = df[(df["pattern"] == "Junior")&(df["relevant"])]
      df = df.head(top_postings)
  else:
      df = df[(df["pattern"] != "Junior")&(df["relevant"])]
      df = df.head(top_postings)
  df = df.drop(["external_id", "relevant", "skills","matched","pattern"], axis = 1)
  print(df)
  Path(config.FITTED_POSTS_DIR).mkdir(exist_ok=True)
  df.to_excel(Path(config.FITTED_POSTS_DIR) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx", index = False)





fitted_excel(october, 30, True)


    

