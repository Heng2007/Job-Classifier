"""Find the job postings that best fit your skills.

Run from the project root: python -m analysis.rank_postings

1. Set JOBS_CSV_PATH in config.py to your postings CSV
   (columns: external_id, company, title, description, url, fetched_at).
2. Pick your skills by number when prompted, or press Enter to reuse
   the ones saved in my_skills.json.
3. The top postings are saved as an Excel file in fitted_posts/,
   named by date and time.

How it ranks: the tuned MLP drops non-tech postings, titles mark each
posting Junior / Senior / Not Verified, and postings are sorted by how
many of your skills they name, then by match_score (share of the
posting's skills you have).
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
from analysis.skills_taxonomy import extract_skills, SKILL_PATTERNS
import config

mlp = joblib.load(config.TUNED_MLP_MODEL_PATH)
vectorizer = joblib.load(config.TUNED_VECTORIZER_PATH)
encoder = joblib.load(config.LABEL_ENCODER_PATH)
dataset = pd.read_csv(config.JOBS_CSV_PATH)

#import my skills and convert it into list
with open(config.SKILLS_PROFILE_PATH) as f:
    load = json.load(f)
my_skill_list = load["known_skills"]


def choose_skills(saved: list) -> list:
    """print every skill with a number, let the user type the numbers they have;
    pressing Enter keeps the saved list. New picks are saved to the json file."""
    all_skills = list(SKILL_PATTERNS)
    for i, skill in enumerate(all_skills, 1):
        print(f"{i:>2}. {skill}")
    print(f"\nSaved skills: {', '.join(saved)}")
    answer = input("Type the numbers of your skills (e.g. 1, 4, 17), or press Enter to keep the saved ones: ")
    if not answer.strip():
        return saved
    chosen = [all_skills[int(n) - 1] for n in answer.replace(",", " ").split()]
    with open(config.SKILLS_PROFILE_PATH, "w") as f:
        json.dump({"known_skills": chosen}, f, indent=2)
    return chosen


my_skill_list = choose_skills(my_skill_list)


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





fitted_excel(dataset, 30, True)


    

