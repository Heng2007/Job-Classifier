r"""Match scoring against my skill profile, and the unlock table.

Responsible for: scoring every posting by how much of its required skill set I
already have, flagging near-misses (postings missing at most 2 of my skills),
and building the "learn X -> unlock N postings" table that turns the database
into a learning plan.

Inputs: my_skills.json; the skills detected per posting by
analysis/skills_taxonomy.py; posting relevance/category from the trained model.

Outputs: a match percentage per posting; a ranked shortlist; a near-miss list
with the specific missing skills named; and an unlock table counting, for each
skill I don't have, how many additional postings learning it would qualify me
for — sorted by that count.

The junior-eligibility filter: title matches config.JUNIOR_TITLE_PATTERN and
does not match config.SENIOR_TITLE_PATTERN. Both are whole-word patterns —
without \b, "intern" matches inside "Internal"/"International" and "lead"
inside "Leadership".

Known limit, measured on the 5,447 Greenhouse postings: a title is a name, not
a requirement, so this cannot detect a posting titled "intern" that wants 8
years. 35 postings pass the title test and 1 of those is a fake (7 years); 358
postings asking for <= 1 year fail it because their titles carry no junior
keyword. The filter therefore discards ~358 good postings to catch 1 bad one,
and tightening the word list makes that worse.

The real defence is the match score itself: an over-senior posting demands
skills I don't have, scores low, and sinks in the ranking without any seniority
rule. If I do want to fix the filter, the signal is in the description — 85% of
Greenhouse postings state a years requirement, extractable with something like
(\d{1,2})\s*\+?\s*year — making eligibility "junior title OR stated years <= 2,
minus senior title, minus years >= 5". Note whichever choice I make in the
README's limitations section.

How I'll know it works: a posting requiring only skills I have scores 100%, a
posting requiring none of them scores 0%, the near-miss list contains only
postings missing 1-2 skills, and the top of the unlock table matches what I'd
guess by eye from the shortlist.


My skill -> number
whatever has the highest number
"""
import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

#import my skills and convert it into list
with open(config.SKILLS_PROFILE_PATH) as f:
    load = json.load(f)
my_skill_list = load[list(load.keys())[0]]

#match each skill with its number
conn = sqlite3.connect(config.DB_PATH)

def get_skills_number(conn: sqlite3.Connection) -> list:
    """fetch all the skills corrosponding to number and tier in the form of
      (numebr, skill_name, tier)"""
    return conn.execute("SELECT id, name, tier FROM skills").fetchall()

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
ranking["skill_count"] = posting_skills.apply(lambda x: len([1 for i in x["skill_id"] if i in converted_my_skill]) , axis = 1)
ranking = ranking.sort_values("skill_count", ascending=False)
print(ranking.head())












