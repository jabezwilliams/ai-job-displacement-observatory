# AI Job Displacement Risk Observatory

> An end-to-end data analytics project quantifying the risk of AI-driven job displacement by occupation and U.S. metropolitan area — featuring a composite risk index, machine learning forecasts, and LLM-generated risk briefings.

## Project Overview

This project answers a question that matters deeply in 2025–2026: **which workers, in which places, are most exposed to AI-driven displacement — and how is that risk changing?**

Using publicly available labor market data from the Bureau of Labor Statistics (BLS) and O*NET, I construct a composite **AI Displacement Risk Index** scored at the occupation × metro area level. A machine learning model (XGBoost + Prophet) forecasts how that risk evolves over the next 12–18 months, and an LLM-powered briefing engine generates plain-language summaries of findings for each occupation.

## Data Sources

| Source | What It Provides |
|--------|-----------------|
| BLS Occupational Employment & Wage Statistics (OEWS) | Employment counts and wages by occupation and metro area |
| O*NET Automation Risk Data | Task-level automation susceptibility scores by occupation |
| BLS Employment Situation (CPS) | Year-over-year employment trends |

## Tech Stack

- **Languages:** Python, SQL
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Seaborn
- **Database:** PostgreSQL (via SQLAlchemy)
- **AI Integration:** OpenAI API (GPT-4) for automated risk briefings
- **Visualization:** Tableau, Matplotlib, Seaborn
- **Infrastructure:** Docker-ready, GitHub Actions CI

## Project Structure

## Setup Instructions

```bash
git clone https://github.com/yourusername/ai-job-displacement-observatory.git
cd ai-job-displacement-observatory
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*Built by Jabez Williams | [LinkedIn](www.linkedin.com/in/jabez-williams-7ab3611b3)*
