# AI Job Displacement Risk Observatory

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end data analytics project quantifying AI-driven job displacement risk
by occupation and U.S. metropolitan area across 938 occupations and 393 metro areas.

🔗 [Live Interactive Dashboard](https://public.tableau.com/views/AI-Job-Displacement-Risk-Observatory/AIDisplacementObservatory)

---

## What This Builds

A composite **AI Displacement Risk Index** (0–1 scale) built from four features:

| Feature | Source | Weight |
|---|---|---|
| Automation Probability | Frey & Osborne (2013) | 40% |
| AI Task Exposure | O*NET Work Activities | 25% |
| Employment Trend | BLS OEWS 2018–2025 | 20% |
| Wage Stagnation | BLS OEWS 2018–2025 | 15% |

The pipeline ends with GPT-4o generating plain-language risk briefings for the
50 highest-risk occupation × metro combinations, stored in PostgreSQL.

---

## Project Structure

| Part | Description | Status |
|---|---|---|
| 1–3 | Setup, data collection, PostgreSQL ingestion | ✅ |
| 4 | Exploratory Data Analysis (15 charts) | ✅ |
| 5 | Feature engineering & risk index construction | ✅ |
| 6 | Prophet forecasting + XGBoost automation prediction | ✅ |
| 7 | GPT-4o LLM risk briefings (50 briefings) | ✅ |
| 8 | Interactive Tableau dashboard | ✅ |
| 9 | GitHub polish & portfolio writeup | ✅ |

---

## Tech Stack

Python · PostgreSQL · XGBoost · Prophet · OpenAI GPT-4o · Tableau Public · Pandas · SQLAlchemy

---

## Data Sources

| Source | Records |
|---|---|
| BLS OEWS 2018–2025 | ~1.18M rows |
| Frey & Osborne (2013) | 702 occupations |
| O*NET Work Activities | 41,000+ rows |

---

## Setup

```powershell
git clone https://github.com/jabezwilliams/ai-job-displacement-observatory
cd ai-job-displacement-observatory
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

​``` 

DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_observatory
DB_USER=postgres
DB_PASSWORD=your_password
OPENAI_API_KEY=your_key

Run in order:

```powershell
python src/data_ingestion.py
python src/feature_builder.py
python src/forecasting.py
python src/llm_briefings.py
```

Then run notebooks 01 through 05 in Jupyter.

---

## Key Findings

- ~37% of covered workers are in high-risk occupations (automation probability > 0.70)
- Education is the strongest single predictor — less than high school roles average 0.85 vs 0.05 for doctoral-level roles
- Employment was already declining in many high-risk occupations before AI became the headline risk
- Office/Admin, Production, and Sales sectors show the highest concentration of high-risk employment

---

*Built by Jabez Williams · Purdue University B.S. Data Science · 2026*  
[LinkedIn](https://www.linkedin.com/in/jabez-williams-7ab3611b3) · [Tableau Public](https://public.tableau.com/views/AI-Job-Displacement-Risk-Observatory/AIDisplacementObservatory)
