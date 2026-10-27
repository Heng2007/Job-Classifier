"""responsible for populate the posting_skills(a table) with the jobs id and extracted skill"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from analysis import skills_taxonomy
from db.database import insert_skills


conn = sqlite3.connect(config.DB_PATH)

#Update skill that just got added into the list
insert_skills(conn, skills_taxonomy.SKILL_TIER.items())


export = pd.read_sql_query("SELECT id, description FROM postings", conn)
print("started to adding things")
export["skill"] = export["description"].map(lambda x: skills_taxonomy.extract_skills(str(x)))
export = export.explode("skill").dropna(subset = ["skill"])
export["skill"] = export["skill"].apply(lambda x: 
                                        conn.execute("SELECT id FROM skills WHERE name = ?", (x,)).fetchone()[0])  


def store_data(conn: sqlite3.Connection):
    conn.executemany("INSERT OR IGNORE INTO posting_skills(posting_id, skill_id)" \
    "Values(?,?)", export[["id","skill"]].itertuples(False, None))
    conn.commit()
    print("successfully import data")

print("storing data")
conn.execute("DELETE FROM posting_skills")
store_data(conn)


print("an example: "+ str(conn.execute("SELECT posting_id, skill_id FROM posting_skills WHERE posting_id = 100").fetchall()))


