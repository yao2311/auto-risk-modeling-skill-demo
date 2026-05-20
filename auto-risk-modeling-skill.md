---
name: auto-risk-modeling-skill
description: >-
  A vertical Agent Skill demo for credit risk modeling pre-analysis.
  This skill supports data preview, column role identification, target variable confirmation,
  feature statistics, missing value checks, business missing value detection, and result export.
  It is designed for credit risk modeling preparation workflows and will be extended to
  automated modeling, model evaluation, and feature comparison scenarios.
---

# Auto Risk Modeling Skill

A vertical Agent Skill demo for credit risk modeling pre-analysis and automated modeling workflows.

This Skill focuses on the pre-analysis stage of credit risk modeling. It packages common modeling preparation tasks into a reusable Agent Skill workflow, including data preview, column role identification, target variable confirmation, feature statistics, missing value checks, business missing value detection, and result export.

> This repository is a sanitized demo project. It does not contain any company data, client data, internal product logic, confidential rules, or real business data.

---

## Use Cases

Trigger this Skill when the user uploads a structured data file and wants to perform one or more of the following tasks:

- Credit risk modeling pre-analysis
- Sample data exploration
- Data quality checks
- Column role identification
- Target variable confirmation
- Feature statistics
- Missing value analysis
- Business missing value detection
- Modeling sample preparation
- Feature screening preparation
- Automated modeling workflow preparation

Typical trigger keywords include:

- credit risk modeling
- risk modeling
- scorecard
- feature analysis
- variable analysis
- data profiling
- missing value check
- feature statistics
- target variable
- IV analysis
- automated modeling

---

## Supported Input Files

The Skill is designed to support common structured data formats:

- CSV
- Excel
- Parquet

---

## Workflow Overview

```text
[Module 1: Data Preview and Column Role Identification]
  1. Load data file
  2. Preview basic dataset information
  3. Detect information columns, target column candidates, and feature columns
  4. Ask the user to confirm column roles
  5. Save intermediate metadata for the next module

        ↓

[Module 2: Feature Statistics and Missing Value Checks]
  1. Generate feature statistics
  2. Calculate valid rate, missing rate, and descriptive statistics
  3. Detect business missing values
  4. Export statistical result files

        ↓

[Future Extension: Automated Modeling]
  1. Identify classification / regression tasks
  2. Train baseline models
  3. Evaluate model performance
  4. Validate feature effectiveness
  5. Compare different data sources or feature sets