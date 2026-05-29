-- ============================================================
-- AI Job Displacement Observatory — Analysis Queries
-- ============================================================

-- ── 1. Row counts per year (sanity check) ────────────────────
SELECT year, data_type, COUNT(*) AS row_count
FROM bls_oews
GROUP BY year, data_type
ORDER BY year, data_type;

-- ── 2. Top 20 highest-risk occupations (Frey & Osborne) ──────
SELECT occ_code, occ_title, automation_prob, education_req
FROM automation_scores
ORDER BY automation_prob DESC
LIMIT 20;

-- ── 3. Lowest-risk occupations ────────────────────────────────
SELECT occ_code, occ_title, automation_prob, education_req
FROM automation_scores
ORDER BY automation_prob ASC
LIMIT 20;

-- ── 4. National employment by major occupation group (2025) ──
SELECT
    LEFT(occ_code, 2)       AS major_group_code,
    occ_title,
    tot_emp
FROM bls_oews
WHERE year = 2025
  AND data_type = 'national'
  AND occ_group = 'major'
  AND tot_emp NOT IN ('**', '#', '***')
ORDER BY CAST(tot_emp AS NUMERIC) DESC
LIMIT 20;

-- ── 5. Most common O*NET work activities (by occupation count)
SELECT element_name, COUNT(DISTINCT onet_soc_code) AS occupation_count
FROM onet_work_activities
WHERE scale_id = 'IM'
GROUP BY element_name
ORDER BY occupation_count DESC;

-- ── 6. Preview join: BLS + Automation scores ─────────────────
-- This is a preview of the core join you will use throughout
-- the project. SOC codes are the link between the two tables.
SELECT
    b.occ_code,
    b.occ_title,
    b.area_title,
    b.tot_emp,
    a.automation_prob
FROM bls_oews b
JOIN automation_scores a
    ON LEFT(b.occ_code, 7) = LEFT(a.occ_code, 7)
WHERE b.year = 2025
  AND b.data_type = 'msa'
  AND b.occ_group = 'detailed'
  AND b.tot_emp NOT IN ('**', '#', '***')
  AND a.automation_prob > 0.8
ORDER BY a.automation_prob DESC
LIMIT 30;