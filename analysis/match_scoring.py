r"""Near-miss list, unlock table, and the script I run to find jobs that fit me.

Responsible for: the entry point of the job fitter. Loads a postings dataset
and my skills, hands them to analysis/rank_postings.rank_postings for scoring
and filtering, then builds the two views that turn the ranking into a plan:
the near-miss list and the "learn X -> unlock N postings" table. Run from the
project root as `python -m analysis.match_scoring`.

Inputs: a postings dataset — a CSV with title + description (+ any other
columns), or the postings table in config.DB_PATH; my skills from
config.SKILLS_PROFILE_PATH (the "known_skills" list).

Outputs: the top-N ranked postings; a near-miss list — postings missing 1 to
config.NEAR_MISS_MAX_MISSING_SKILLS (2) skills, with the missing skills named;
and an unlock table counting, for each skill I don't have, how many additional
postings learning it would turn into near-misses — sorted by that count.

Ranking, the relevance classifier and the junior filter live in
rank_postings.py, not here.

If I ever want a better junior filter than the title test, the signal is in
the description — 85% of Greenhouse postings state a years requirement,
extractable with something like (\d{1,2})\s*\+?\s*year — making eligibility
"junior title OR stated years <= 2, minus senior title, minus years >= 5".

How I'll know it works: the near-miss list contains only postings missing 1-2
skills, and the top of the unlock table matches what I'd guess by eye from the
shortlist (I already have PyTorch and SQL, so expect things like Spark, AWS,
LLM; if MATLAB is #1, inspect why before believing it).


My skill -> number
whatever has the highest number
"""
import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis.skills_taxonomy import extract_skills
import config

#import my skills and convert it into list
with open(config.SKILLS_PROFILE_PATH) as f:
    load = json.load(f)
my_skill_list = load["known_skills"]

#match each skill with its number
conn = sqlite3.connect(config.DB_PATH)

def get_skills_number(conn: sqlite3.Connection) -> list:
    """fetch all the skills corrosponding to number and tier in the form of
      (number, skill_name, tier)"""
    return conn.execute("SELECT id, name, tier FROM skills").fetchall()
#in (number, skill_name, tier)
skills_list = get_skills_number(conn)


def convert_skill_numebr(skills_list:list, my_skill_list:list)->list:
    """convert all the skills into the tuple of (skill_number, tier)"""
    return_value = []
    for my_skill in my_skill_list:
        for skill in skills_list:
            if my_skill == skill[1]:
                return_value.append(skill[0])
    return return_value


#It is in (skill_number)
converted_my_skill = convert_skill_numebr(skills_list, my_skill_list)

"created data frame with posting skills"
posting_skills = pd.read_sql_query("SELECT posting_id, skill_id FROM posting_skills", conn)
posting_skills = posting_skills.groupby("posting_id", as_index = False).agg(list)

ranking = pd.DataFrame()
ranking["posting_id"] = posting_skills["posting_id"]
ranking["match_score"] = posting_skills["skill_id"].apply(
    lambda ids: len(set(ids) & set(converted_my_skill)) / len(ids))
ranking = ranking.sort_values("match_score", ascending=False)


def tie_break(ranking: pd.DataFrame, posting_skills: pd.DataFrame, my_skills: list,
              skills_col: str = "skill_id") -> pd.DataFrame:
    """Re-sort ranking by match_score, then by how many of my skills each
    posting names"""
    mine = set(my_skills)
    out = ranking.copy()
    out["matched"] = posting_skills[skills_col].apply(lambda s: len(set(s) & mine))
    return out.sort_values(["match_score", "matched"], ascending=False)


def near_miss(posting_skills: pd.DataFrame, my_skills: list,
              max_missing: int = config.NEAR_MISS_MAX_MISSING_SKILLS,
              skills_col: str = "skill_id") -> pd.DataFrame:
    """Postings missing between 1 and max_missing of their skills, with the
    missing skills named and the fewest-missing first."""
    mine = set(my_skills)
    out = posting_skills.copy()
    out["missing"] = out[skills_col].apply(lambda s: set(s) - mine)
    out["n_missing"] = out["missing"].apply(len)
    out = out[out["n_missing"].between(1, max_missing)]
    return out.sort_values("n_missing")


def unlock_table(posting_skills: pd.DataFrame, my_skills: list,
                 max_missing: int = config.NEAR_MISS_MAX_MISSING_SKILLS,
                 skills_col: str = "skill_id") -> pd.Series:
    """For each skill I lack: how many postings would newly become near-misses
    if I learned it, sorted most to fewest."""
    mine = set(my_skills)
    missing = posting_skills[skills_col].apply(lambda s: set(s) - mine)
    just_outside = missing[missing.apply(len) == max_missing + 1]
    return just_outside.explode().value_counts()

#convert the most need skill to list, still skill number
# top_skill_need_number = unlock_table(posting_skills, converted_my_skill).sort_values(ascending=False).head().to_dict()
# skill_dict = {}
# for all_skill in skills_list:
#     for skill_number, appeareance in top_skill_need_number.items():
#         if all_skill[0] == skill_number:
#             skill_dict[all_skill[1]] = appeareance
# print(skill_dict)
#-----------------------------------------------------------------------------
# top_skill_needed = [i[1] for i in skills_list if i[0] in top_skill_need_number.values()]
# print(top_skill_needed)

id_to_name = {skill_id: name for skill_id, name, _ in skills_list}
top_skill_need_number = unlock_table(posting_skills, converted_my_skill).head(10).rename(index=id_to_name).to_dict()
print(top_skill_need_number)




