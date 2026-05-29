-- ============================================================
-- AI Job Displacement Observatory — Database Schema
-- ============================================================

-- Drop tables if they already exist (useful when re-running)
DROP TABLE IF EXISTS bls_oews CASCADE;
DROP TABLE IF EXISTS automation_scores CASCADE;
DROP TABLE IF EXISTS onet_work_activities CASCADE;
DROP TABLE IF EXISTS onet_occupations CASCADE;

-- ── Table 1: BLS OEWS Data (all years combined) ──────────────
-- This is your largest table. Each row is one occupation in one
-- metro area for one survey year.
CREATE TABLE bls_oews (
    id              SERIAL PRIMARY KEY,
    year            INTEGER NOT NULL,
    area_code       TEXT,
    area_title      TEXT,
    area_type       TEXT,
    occ_code        TEXT,           -- SOC code, e.g. "15-1252"
    occ_title       TEXT,
    occ_group       TEXT,           -- "detailed", "broad", "major", "total"
    tot_emp         TEXT,           -- kept as TEXT to preserve BLS suppression flags ("**", "#")
    h_mean          TEXT,
    a_mean          TEXT,
    h_median        TEXT,
    a_median        TEXT,
    data_type       TEXT            -- "msa" or "national"
);

-- ── Table 2: Frey & Osborne Automation Scores ────────────────
-- One row per occupation with its automation probability score
CREATE TABLE automation_scores (
    id                  SERIAL PRIMARY KEY,
    occ_code            TEXT,
    occ_title           TEXT,
    automation_prob     NUMERIC(5,4),  -- decimal between 0 and 1
    median_wage         NUMERIC(12,2),
    education_req       TEXT
);

-- ── Table 3: O*NET Work Activities ───────────────────────────
-- One row per occupation per work activity per scale
-- This gives us 41 activity ratings per occupation
CREATE TABLE onet_work_activities (
    id              SERIAL PRIMARY KEY,
    onet_soc_code   TEXT,
    title           TEXT,
    element_id      TEXT,
    element_name    TEXT,
    scale_id        TEXT,       -- "IM" = Importance, "LV" = Level
    data_value      NUMERIC(6,2),
    recommend_suppress TEXT
);

-- ── Table 4: O*NET Occupation Reference ──────────────────────
-- Simple lookup table mapping O*NET SOC codes to occupation titles
CREATE TABLE onet_occupations (
    id              SERIAL PRIMARY KEY,
    onet_soc_code   TEXT UNIQUE,
    title           TEXT,
    description     TEXT
);

-- ── Indexes ──────────────────────────────────────────────────
-- These speed up the joins you'll write later significantly
CREATE INDEX idx_bls_occ_code  ON bls_oews (occ_code);
CREATE INDEX idx_bls_year      ON bls_oews (year);
CREATE INDEX idx_bls_area      ON bls_oews (area_code);
CREATE INDEX idx_auto_occ_code ON automation_scores (occ_code);
CREATE INDEX idx_onet_soc_code ON onet_work_activities (onet_soc_code);

