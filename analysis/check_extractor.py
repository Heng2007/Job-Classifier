"""This document is used to check the accuracy of the regex extractor"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import skills_taxonomy
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



conn = sqlite3.connect(config.DB_PATH)

# def extract_postings(conn: sqlite3.Connection) -> list:
#     """Responsible for extracting post from the sql table"""
#     cur = conn.cursor()
#     cur.execute("SELECT title, description FROM postings  " 
#     "WHERE id in (SELECT posting_id FROM ( " 
#         "SELECT posting_id," 
#         "ROW_NUMBER() OVER (PARTITION BY category ORDER BY RANDOM()) AS rn" 
#         " FROM labels WHERE category != 'Not relevant') WHERE rn<=4)")
#     return cur.fetchall()
# conn.close
df = pd.read_sql_query("SELECT p.title, p.description, l.category FROM postings p "
                       "JOIN labels l ON l.posting_id = p.id WHERE l.category != 'Not relevant' "
                     , conn)
sample = df.groupby("category").sample(n=4, random_state=config.RANDOM_SEED)

print(sample.keys())
sample["skill"] = sample["description"].map(
    lambda x: skills_taxonomy.extract_skills(str(x)))
path = Path("compare_skill.xlsx")

sample.to_excel(path, index = False)
