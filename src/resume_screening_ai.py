# resume_screening_ai.py
# Final CLI: role discovery + target screening + ML hint + CSV logging

import re
import difflib
import csv
import os
from datetime import datetime

import joblib
import numpy as np

# ==================== ROBUST PATHS (NEVER BREAK) ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

# -------------------- LOAD MODEL --------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file:\n{MODEL_PATH}\nRun train.py first.")

if not os.path.exists(VEC_PATH):
    raise FileNotFoundError(f"Missing vectorizer file:\n{VEC_PATH}\nRun train.py first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# -------------------- TEXT NORMALIZATION --------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "with", "on", "at", "as",
    "looking", "seeking", "need", "required", "requirements", "requirement", "role",
    "developer", "engineer", "analyst", "specialist", "intern", "candidate", "hiring",
    "years", "year", "experience"
}

def normalize(text: str) -> str:
    text = (text or "").lower()

    # common phrase → single token
    text = text.replace("machine learning", "machinelearning")
    text = text.replace("deep learning", "deeplearning")
    text = text.replace("power bi", "powerbi")
    text = text.replace("rest api", "restapi")
    text = text.replace("spring boot", "springboot")
    text = text.replace("data warehouse", "datawarehouse")

    # common misspellings / variants
    text = text.replace("tableu", "tableau")
    text = text.replace("scikitlearn", "scikit")

    # tech variants
    text = text.replace("node.js", "node").replace("nodejs", "node")
    text = text.replace("react.js", "react").replace("reactjs", "react")
    text = text.replace("c#", "csharp")
    text = text.replace(".net", "dotnet")

    # keep letters, numbers, + # .
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    text = normalize(text)
    tokens = []
    for w in text.split():
        if len(w) <= 2:
            continue
        if w in STOPWORDS:
            continue
        tokens.append(w)
    return set(tokens)

# -------------------- ROLE LIBRARY (CORE/OPTIONAL) --------------------
ROLE_SKILLS = {
    # WEB
    "web developer": {
        "core": ["html", "css", "javascript"],
        "optional": ["react", "node", "express", "git", "api", "restapi", "responsive", "bootstrap", "tailwind", "sql"]
    },
    "frontend developer": {
        "core": ["html", "css", "javascript"],
        "optional": ["react", "vue", "angular", "responsive", "bootstrap", "tailwind", "git"]
    },
    "backend developer": {
        "core": ["api", "restapi", "sql"],
        "optional": ["node", "express", "python", "django", "flask", "java", "spring", "mysql", "postgresql", "git"]
    },
    "full stack developer": {
        "core": ["html", "css", "javascript"],
        "optional": ["react", "node", "express", "api", "restapi", "sql", "python", "django", "git"]
    },
    "react developer": {
        "core": ["react", "javascript"],
        "optional": ["redux", "html", "css", "api", "git"]
    },
    "nodejs developer": {
        "core": ["node", "express", "javascript"],
        "optional": ["api", "restapi", "mongodb", "sql", "git"]
    },
    "django developer": {
        "core": ["python", "django"],
        "optional": ["api", "restapi", "sql", "git"]
    },
    "flask developer": {
        "core": ["python", "flask"],
        "optional": ["api", "restapi", "sql", "git"]
    },

    # DATA
    "data analyst": {
        "core": ["sql", "excel", "analysis", "visualization"],
        "optional": ["powerbi", "tableau", "python", "pandas", "numpy", "statistics"]
    },
    "business analyst": {
        "core": ["requirements", "documentation", "analysis"],
        "optional": ["excel", "sql", "reporting", "powerbi", "tableau"]
    },
    "data scientist": {
        "core": ["python", "machinelearning", "statistics"],
        "optional": ["pandas", "numpy", "scikit", "tensorflow", "pytorch", "deeplearning"]
    },
    "data engineer": {
        "core": ["sql", "etl", "python"],
        "optional": ["spark", "hadoop", "airflow", "datawarehouse", "bigdata", "aws"]
    },
    "bi developer": {
        "core": ["sql", "visualization"],
        "optional": ["powerbi", "tableau", "excel", "reporting", "dax"]
    },

    # SOFTWARE
    "python developer": {
        "core": ["python"],
        "optional": ["django", "flask", "api", "restapi", "sql", "git"]
    },
    "java developer": {
        "core": ["java"],
        "optional": ["spring", "springboot", "hibernate", "mysql", "restapi", "git", "maven"]
    },
    "dotnet developer": {
        "core": ["dotnet", "csharp"],
        "optional": ["sql", "api", "git"]
    },

    # DEVOPS / CLOUD
    "devops engineer": {
        "core": ["linux", "docker", "git"],
        "optional": ["kubernetes", "ci", "cd", "jenkins", "aws", "monitoring", "bash"]
    },
    "cloud engineer": {
        "core": ["aws", "linux", "networking"],
        "optional": ["azure", "gcp", "docker", "kubernetes", "security"]
    },

    # QA / TESTING
    "qa engineer": {
        "core": ["testing", "testcase"],
        "optional": ["selenium", "automation", "api", "git", "bug"]
    },
    "automation tester": {
        "core": ["automation", "testing"],
        "optional": ["selenium", "python", "java", "postman", "cypress"]
    },

    # SECURITY / NETWORK / IT OPS
    "cyber security analyst": {
        "core": ["security", "network", "linux"],
        "optional": ["firewall", "risk", "incident", "siem", "monitoring"]
    },
    "network engineer": {
        "core": ["network", "tcp", "ip"],
        "optional": ["dns", "router", "switch", "linux", "troubleshooting"]
    },
    "system administrator": {
        "core": ["linux", "server"],
        "optional": ["network", "monitoring", "security", "backup", "virtualization"]
    },
    "it support specialist": {
        "core": ["troubleshooting", "support"],
        "optional": ["hardware", "software", "network", "windows", "linux"]
    },

    # MOBILE
    "android developer": {
        "core": ["kotlin", "android"],
        "optional": ["java", "sdk", "api", "firebase", "mobile"]
    },
    "ios developer": {
        "core": ["swift", "ios"],
        "optional": ["xcode", "api", "mobile"]
    },
}

ROLE_GROUPS = {
    "Web & Software": {
        "web developer", "frontend developer", "backend developer", "full stack developer",
        "react developer", "nodejs developer", "django developer", "flask developer",
        "python developer", "java developer", "dotnet developer"
    },
    "Data": {"data analyst", "business analyst", "data scientist", "data engineer", "bi developer"},
    "DevOps & Cloud": {"devops engineer", "cloud engineer"},
    "QA & Testing": {"qa engineer", "automation tester"},
    "Security & IT Ops": {"cyber security analyst", "network engineer", "system administrator", "it support specialist"},
    "Mobile": {"android developer", "ios developer"},
}

# -------------------- ROLE SEARCH / SUGGEST --------------------
def suggest_roles(user_input: str, n: int = 5):
    ui = normalize(user_input)
    roles = list(ROLE_SKILLS.keys())
    return difflib.get_close_matches(ui, roles, n=n, cutoff=0.35)

def list_roles():
    print("\nAvailable job roles:")
    for r in sorted(ROLE_SKILLS.keys()):
        print(" -", r)

# -------------------- ML CATEGORY HINT --------------------
def top_predictions(resume_text: str, top_k: int = 3):
    vec = vectorizer.transform([resume_text.lower()])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    idx = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], float(probs[i]) * 100) for i in idx]

# -------------------- ATS SCORING (WEIGHTED CORE/OPTIONAL) --------------------
def weighted_role_score(resume_tokens: set, role: str):
    role_obj = ROLE_SKILLS.get(role)

    if not isinstance(role_obj, dict) or "core" not in role_obj or "optional" not in role_obj:
        return {
            "role": role, "score": 0.0,
            "core_matched": [], "core_missing": [],
            "opt_matched": [], "opt_missing": []
        }

    core = set(role_obj["core"])
    opt = set(role_obj["optional"])

    core_matched = core & resume_tokens
    core_missing = core - resume_tokens

    opt_matched = opt & resume_tokens
    opt_missing = opt - resume_tokens

    core_score = (len(core_matched) / len(core)) * 100 if core else 0.0
    opt_score = (len(opt_matched) / len(opt)) * 100 if opt else 0.0

    score = 0.7 * core_score + 0.3 * opt_score

    return {
        "role": role,
        "score": score,
        "core_matched": sorted(core_matched),
        "core_missing": sorted(core_missing),
        "opt_matched": sorted(opt_matched),
        "opt_missing": sorted(opt_missing),
    }

def recommend_roles(resume_text: str, roles_to_consider=None, top_n: int = 3):
    rtokens = tokenize(resume_text)
    roles = roles_to_consider if roles_to_consider else ROLE_SKILLS.keys()
    results = [weighted_role_score(rtokens, role) for role in roles]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

# -------------------- FINAL DECISION (DYNAMIC WEIGHTING) --------------------
def final_decision(ml_conf: float, ats_score: float, resume_text: str):
    wc = len(normalize(resume_text).split())

    if wc <= 25:
        final_score = 0.9 * ats_score + 0.1 * ml_conf
    else:
        final_score = 0.7 * ats_score + 0.3 * ml_conf

    if final_score >= 70:
        decision = "Recommended"
    elif final_score >= 45:
        decision = "Maybe Suitable"
    else:
        decision = "Not Recommended"

    return final_score, decision

# -------------------- OUTPUT --------------------
def label_for_score(score: float) -> str:
    if score >= 70:
        return "Ready"
    if score >= 45:
        return "Almost"
    return "Needs work"

def print_simple_recommendations(resume_text: str):
    print("\nTop job roles for you (by track):")
    for group_name, group_roles in ROLE_GROUPS.items():
        top = recommend_roles(resume_text, roles_to_consider=group_roles, top_n=1)[0]
        print(f"- {group_name}: {top['role'].title()} — {top['score']:.0f}% ({label_for_score(top['score'])})")

def print_target_simple(target_result, ml_top3, resume_text):
    role = target_result["role"]
    ats_score = target_result["score"]
    ml_conf = ml_top3[0][1]
    final_score, decision = final_decision(ml_conf, ats_score, resume_text)

    improvements = target_result["core_missing"][:2] if target_result["core_missing"] else target_result["opt_missing"][:2]

    print(f"\nBest fit check: {role.title()}")
    print(f"Decision: {decision}  |  Match: {final_score:.0f}%")
    if improvements:
        print("Improve next:", ", ".join(improvements))

def print_details(target_result):
    print("\n[Details]")
    print("Core matched:", target_result["core_matched"])
    print("Core missing:", target_result["core_missing"])
    print("Optional matched:", target_result["opt_matched"][:15])
    print("Optional missing:", target_result["opt_missing"][:15])

# -------------------- CSV LOGGING --------------------
def save_result_csv(resume_text, target_role, decision, final_score, ats_score, ml_top1_cat, ml_top1_conf, improvements):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "screening_results.csv")

    header = [
        "timestamp", "resume_input", "target_role",
        "decision", "final_score", "ats_score",
        "ml_top1_category", "ml_top1_confidence",
        "improvements"
    ]
    row = [
        datetime.now().isoformat(timespec="seconds"),
        resume_text,
        target_role,
        decision,
        round(final_score, 2),
        round(ats_score, 2),
        ml_top1_cat,
        round(ml_top1_conf, 2),
        ", ".join(improvements) if improvements else ""
    ]

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("AI Resume Screening System")
    print("--------------------------")
    print("Tip: type 'list roles' to view supported roles.")
    print("Tip: type 'details' after target screening to view matched/missing.\n")

    resume = input("Enter Resume text or skills (single line): ").strip()
    if not resume:
        print("Please enter resume text/skills.")
        raise SystemExit

    print_simple_recommendations(resume)

    ml_top3 = top_predictions(resume, top_k=3)
    print("\nML Category Hint (supporting):")
    for cat, conf in ml_top3:
        print(f"- {cat}: {conf:.2f}%")

    job = input("\nEnter a target job title to screen (or press Enter to skip): ").strip()

    if normalize(job) == "list roles":
        list_roles()
        raise SystemExit

    if job:
        job_n = normalize(job)

        if job_n not in ROLE_SKILLS:
            suggestions = suggest_roles(job, n=5)
            if suggestions:
                print("\nDid you mean:")
                for s in suggestions:
                    print(" -", s)
                job_n = suggestions[0]
                print("Using closest match:", job_n)
            else:
                print("No close role found. Use 'list roles' to see supported roles.")
                raise SystemExit

        target_result = weighted_role_score(tokenize(resume), job_n)
        print_target_simple(target_result, ml_top3, resume)

        ats_score = target_result["score"]
        ml_top1_cat, ml_top1_conf = ml_top3[0]

        final_score, decision = final_decision(ml_top1_conf, ats_score, resume)
        improvements = target_result["core_missing"][:2] if target_result["core_missing"] else target_result["opt_missing"][:2]

        save_result_csv(resume, job_n, decision, final_score, ats_score, ml_top1_cat, ml_top1_conf, improvements)

        show = input("\nType 'details' to see matched/missing (or press Enter to finish): ").strip().lower()
        if show == "details":
            print_details(target_result)

    print("\nDone.")