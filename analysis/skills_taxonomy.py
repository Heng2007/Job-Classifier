"""Skill vocabulary and regex extraction of skills from posting text.

Responsible for: defining the skill list with a learning tier for each skill,
mapping skills to the 8 categories, and extracting which skills a posting
mentions. Extraction is regex, not ML — deliberately. Short tokens (R, AI, ML,
NLP, LLM) need word-boundary patterns, because plain substring matching
false-positives on words like "training" and "enrollment". That bug bit the
UofT analysis earlier in this project and is worth avoiding here too.

SKILL_TIER encodes rough learning priority (1 = foundational, higher = more
specialized) so match scoring can rank "missing skills" sensibly instead of
just listing them alphabetically.

Inputs: raw posting description text; the skill pattern table defined in this
module.

Outputs: the set of skills detected in a posting; each skill's tier (used to
order what to learn next); the skill-to-category mapping used by the app and by
match scoring.

How I'll know it works: tests/test_taxonomy.py passes — known strings produce
exactly the expected skills, and the word-boundary cases behave: "enrollment"
does not match R, "training" does not match AI, "R and Python" does match R.
"""
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

SKILL_PATTERNS: dict[str, str] = {
    # Programming languages
    "Python": r"\bpython\b",
    "R": r"(?<![a-zA-Z])r(?![a-zA-Z])",
    "SQL": r"\bsql\b",
    "Java": r"\bjava\b(?!script)",
    "C++": r"c\+\+",
    "C": r"(?<![a-z0-9])c(?![a-z0-9+#-])",
    "Julia": r"\bjulia\b",
    "MATLAB": r"\bmatlab\b",
    "Stata": r"\bstata\b",
    # Classical ML
    "scikit-learn": r"scikit-learn|sklearn",
    "Random Forest": r"random forest",
    "XGBoost": r"xgboost|gradient boosting",
    "Logistic Regression": r"logistic regression",
    "SVM": r"support vector machine|\bsvm\b",
    "Cross-validation": r"cross-validation|cross validation",
    "Optimization": r"convex optimization|numerical optimization|mathematical optimization|linear programming|\bmilp?\b",

    # Deep learning
    "PyTorch": r"pytorch",
    "TensorFlow": r"tensorflow",
    "Keras": r"\bkeras\b",
    "Neural Networks": r"neural network",
    "CNN": r"\bcnn\b|convolutional neural",
    "RNN/LSTM": r"\brnn\b|lstm",
    "Autoencoder": r"autoencoder|variational autoencoder",
    # NLP / LLM
    "NLP": r"natural language processing|\bnlp\b",
    "NLTK": r"\bnltk\b",
    "BERT": r"\bbert\b",
    "Transformers": r"transformer",
    "HuggingFace": r"hugging ?face",
    "LLM": r"\bllm\b|large language model",
    "Generative AI": r"generative ai|genai",
    "RAG": r"retrieval.augmented generation|\brag\b",
    "Prompt Engineering": r"prompt engineering",
    "Computer Vision": r"computer vision|\bopencv\b|image recognition|object detection",
    "RLHF":    r"\brlhf\b|reinforcement learning from human feedback",
    "Agentic AI":      r"\bagentic\b|\b(?:ai|llm|coding|autonomous) agents?\b|\bagent frameworks?\b",
    "MCP":             r"\bmcp\b|model context protocol",
    "LLM Evaluation":  r"\bevals\b|evaluation (?:and quality )?frameworks?|llm evaluation",
    
    # Data / databases / viz
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "dbt": r"\bdbt\b",
    "Looker": r"\blooker\b",
    "Snowflake": r"\bsnowflake\b",
    "Elasticsearch": r"elasticsearch",
    "MongoDB": r"mongodb",
    "Tableau": r"\btableau\b",
    "Airflow": r"\bairflow\b",
    "Power BI": r"power ?bi",
    "Spark": r"\b(?:py)?spark\b",
    "ETL": r"\betl\b|\belt\b|extract,? transform,? (?:and )?load",
    "Kafka": r"\bkafka\b",
    "Flink": r"\bflink\b",
    # Infra / engineering practice
    "Git": r"\bgit\b|github",
    "Linux": r"\blinux\b",
    "AWS": r"\baws\b",
    "Docker": r"\bdocker\b",
    "SLURM/HPC": r"\bslurm\b|compute canada|high.performance computing",
    "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "Distributed Computing": r"distributed (?:computing|systems?|training)",
      "AI Coding Tools": r"\bcursor\b(?! position)|claude code|\bcodex\b|\bcopilot\b",
      "System Design":   r"systems? design",
    #Front End
    "React":   r"\breact(?:\.?js)?\b(?! (?:well|quickly|to|and adapt|appropriately|calmly|fast))",
    "Angular": r"\bangular(?:\.?js)?\b",
    "JavaScript": r"\bjavascript\b",
    "HTML": r"\bhtml5?\b",
    "CSS": r"\bcss\b",
    # Specialized
    "Reinforcement Learning": r"reinforcement learning|imitation learning",
    "Bayesian Optimization": r"bayesian optimization",
}

# 1 = build this first, 4 = specialized / only if targeting a specific niche
SKILL_TIER: dict[str, int] = {
    "Python": 1, "Pandas": 1, "NumPy": 1, "Git": 1,
    "Logistic Regression": 1, "Random Forest": 1, "XGBoost": 1, "Cross-validation": 1,
    "scikit-learn": 1,
    "SQL": 2, "R": 2, "PyTorch": 2, "TensorFlow": 2, "Neural Networks": 2, "AI Coding Tools	":2,
    "NLP": 2, "NLTK": 2, "BERT": 2, "Transformers": 2, "Computer Vision": 2, "System Design":2,
    "HuggingFace": 3, "LLM": 3, "Generative AI": 3, "RAG": 3, "Prompt Engineering": 3, "MCP": 3, "Agentic AI": 3,
    "Reinforcement Learning": 3,"RLHF":3, "Airflow":3, "ETL": 3, "Spark":3, "Bayesian Optimization": 3,
    "LLM Evaluation": 3, "CNN": 3, "RNN/LSTM": 3, "REACT":3, "Distributed Computing":3,"Optimization":3,  
    "Kafka": 3,  "Kubernetes":3, "Autoencoder": 3, "snowflake":3, "dbt":3, "Keras": 3,
    "JavaScript":3, "HTML":3, "CSS":3, "MATLAB": 4, "C":4, "Angular":4, "Julia": 4, "Stata": 4, "C++": 4, 
    "Java": 4, "Flink":4, "Elasticsearch": 4, "MongoDB": 4, "Tableau": 4, "Power BI": 4,
    "Linux": 2, "AWS": 3, "Docker": 3, "Looker":4,"SLURM/HPC": 4, "SVM": 2,
}

CATEGORIES = config.CATEGORIES


# Keys must stay in sync with config.CATEGORIES, or weak_label_category returns
# a category name that no longer exists. These hints now only decide WHICH
# postings get sampled for review -- every posting in the corpus is labeled.
CATEGORY_HINTS: dict[str, list[str]] = {
    "Machine learning / AI": ["machine learning", "neural network", "pytorch", "tensorflow",
                              "deep learning", "cnn", "lstm", "random forest", "xgboost",
                              "logistic regression", "scikit-learn", "gradient boosting",
                              "feature engineering",
                              "nlp", "llm", "bert", "transformer", "huggingface",
                              "generative ai", "prompt engineering"],
    # "sql", "reporting" and "kpi" dropped from the analytics side: all three are
    # finance-department words and pooled 1,270 postings that were overwhelmingly
    # Financial Analyst / Accountant / Controller. Only 29 postings in the whole
    # corpus carry an actual data-role title.
    "Data engineering / analytics": ["data engineer", "data engineering", "data pipeline",
                                     "etl", "elt", "airflow", "dbt", "kafka", "snowflake",
                                     "data warehouse", "data lake",
                                     "dashboard", "tableau", "power bi", "data analyst",
                                     "analytics", "business intelligence"],
    "Software engineering": ["api", "backend", "ci/cd", "microservice", "software engineer", "full stack"],
    "Research assistant": ["research assistant", "principal investigator", "literature review",
                           "research lab", "research group", "research project", "postdoctoral",
                           "research associate", "research intern"],
}


def extract_skills(text: str) -> list[str]:
    """Return the canonical skill names detected in a piece of text."""
    text = text.lower()
    return sorted(
        name for name, pattern in SKILL_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    )


def weak_label_category(text: str) -> str:
    """Cheap keyword-based guess at category -- use to pre-fill labels for
    faster manual review."""
    text = text.lower()
    scores = {cat: sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", text))
              for cat, kws in CATEGORY_HINTS.items()}
    best = max(scores, key=lambda cat: scores[cat])
    return best if scores[best] > 0 else config.NOT_RELEVANT_CATEGORY


