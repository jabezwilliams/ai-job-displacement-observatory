# AI Job Displacement Risk Observatory

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end data analytics project quantifying AI-driven job displacement
risk by occupation and U.S. metropolitan area — across 938 occupations and
393 U.S. metro areas, using 8 years of government labor market data (2018–2025).

🔗 **[View Live Dashboard →](https://public.tableau.com/views/AI-Job-Displacement-Risk-Observatory/AIDisplacementObservatory)**

---

## What This Project Does

This project constructs a composite **AI Displacement Risk Index** scored at
the occupation × metro area level, combining four independently sourced features:

| Feature | Source | Weight |
|---|---|---|
| Automation Probability | Frey & Osborne (2013) | 40% |
| AI Task Exposure Score | O*NET Work Activities | 25% |
| Employment Trend Risk | BLS OEWS 2018–2025 | 20% |
| Wage Stagnation Risk | BLS OEWS 2018–2025 | 15% |

The pipeline culminates in an **LLM-powered briefing engine** that generates
plain-language displacement risk summaries for the 50 highest-risk occupation
× metro combinations using GPT-4o, and Prophet-forecasted employment trends
through 2027.

---

## Live Dashboard

**[→ AI Job Displacement Risk Observatory on Tableau Public](https://public.tableau.com/views/AI-Job-Displacement-Risk-Observatory/AIDisplacementObservatory)**

The interactive dashboard includes:
- **U.S. State Risk Map** — employment-weighted average risk index by state
- **Occupation Risk Ranking** — top 30 highest-risk occupations by composite index
- **Employment Trend** — 2018–2025 historical employment plus Prophet forecasts through 2027
- **Risk Tier Breakdown** — Low/Medium/High risk composition by major sector

All four views are cross-linked — clicking any state, occupation, or risk tier
filters the entire dashboard.

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

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_observatory
DB_USER=postgres
DB_PASSWORD=your_password
OPENAI_API_KEY=your_key
```

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

1. **~37% of covered workers** are employed in high-risk occupations
   (automation probability > 0.70)
2. **Office/Admin, Production, and Sales** sectors show the highest concentration
   of high-risk employment
3. **Education level** is the strongest single predictor of automation risk —
   occupations requiring less than high school education average 0.85
   probability vs. 0.05 for doctoral-level roles
4. **Employment is already declining** in many high-risk occupations
   independent of AI — the displacement signal precedes the technology
5. **Geographic variation** in state-level risk reflects industry mix more than
   regional policy differences

---

*Built by Jabez Williams · Purdue University B.S. Data Science · 2026*
[LinkedIn](https://www.linkedin.com/in/jabez-williams-7ab3611b3) · [Tableau Public](https://public.tableau.com/views/AI-Job-Displacement-Risk-Observatory/AIDisplacementObservatory)
