# MLOps Assignment 2: Data Contracts

**Student Information:**

- **Name:** Laksh Mendpara
- **Roll Number:** B23CS1037
- **Course:** ML-DL-Ops (CSL7120)
- **Assignment:** Data Contracts & YAML Implementation

---

## 📋 Overview

This repository contains the complete implementation of **Data Contracts** for four distinct industry scenarios following the **Open Data Contract Standard (ODCS)** pattern. The assignment demonstrates the application of data contracts to ensure data quality, governance, and reliability across different domains.

### Scenarios Implemented

1. **Ride-Share (CityMove)** - Preventing ML model crashes from schema changes
2. **E-commerce (Flash Sale)** - Real-time order stream validation
3. **IoT (Smart Thermostat)** - Sensor failure detection and data quality
4. **FinTech (Transaction Log)** - Fraud detection with strict format enforcement

---

## 📁 Project Structure

```
MLOps-Laksh-B23CS1037/
├── contracts/                      # Data contract YAML files
│   ├── rides_contract.yaml         # Scenario 1: Ride-Share
│   ├── orders_contract.yaml        # Scenario 2: E-commerce
│   ├── thermostat_contract.yaml    # Scenario 3: IoT
│   └── fintech_contract.yaml       # Scenario 4: FinTech
│
├── scripts/                        # Validation infrastructure
│   ├── __init__.py
│   └── validate_contracts.py       # Comprehensive validation script
│
├── logs/                           # Validation logs
│   ├── yamllint.log                # YAML syntax validation results
│   └── validation_report.log       # Detailed validation report
│
├── README.md                       # This file
├── report.tex                      # LaTeX assignment report
├── pyproject.toml                  # Project dependencies (uv)
└── uv.lock                         # Dependency lock file
```

---

## 🚀 Setup and Installation

This project uses **`uv`** for fast, reliable Python dependency management.

### Prerequisites

- Python 3.12+
- `uv` package manager ([Installation guide](https://github.com/astral-sh/uv))

### Installation Steps

1. **Clone the repository:**

   ```bash
   cd MLOps-Laksh-B23CS1037
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   This installs:

   - `yamllint>=1.38.0` - YAML syntax validation
   - `pyyaml>=6.0.0` - YAML parsing for validation script

---

## ✅ Running Validation

### 1. YAML Syntax Validation

Validate YAML syntax and formatting:

```bash
uv run yamllint contracts/
```

**Expected Output:** No errors (clean output indicates all contracts are syntactically valid)

**Log Location:** `logs/yamllint.log`

### 2. Comprehensive Contract Validation

Run the custom validation script to check:

- ODCS structural requirements
- Scenario-specific business rules
- Quality constraints
- PII tagging
- SLA definitions

```bash
uv run python scripts/validate_contracts.py
```

**Expected Output:**

```
✓ ALL VALIDATIONS PASSED
All contracts meet assignment requirements!
```

**Log Location:** `logs/validation_report.log`

### 3. View Validation Logs

```bash
# YAML syntax validation log
cat logs/yamllint.log

# Detailed validation report
cat logs/validation_report.log
```

---

## 📊 Implementation Details

### Scenario 1: Ride-Share (40 points)

**Problem:** Schema change (`cost_total` → `fare_final`) crashed ML pricing algorithm

**Solution:**

- **Logical Mapping:** Stable interface with `fare_amount` field
- **PII Tagging:** `passenger_id` marked as sensitive
- **SLA:** 30-minute freshness guarantee
- **Quality Rules:**
  - Fare must be non-negative (`fare_amount >= 0`)
  - Driver rating within [1.0, 5.0]
  - Distance must not be null

**Contract:** [`contracts/rides_contract.yaml`](contracts/rides_contract.yaml)

---

### Scenario 2: E-commerce Orders (20 points)

**Problem:** Unmapped status codes crashed marketing dashboard

**Solution:**

- **Enum Mapping:** `status_code` (2, 5, 9) → `status` (PAID, SHIPPED, CANCELLED)
- **Quality Rules:**
  - Order total must be non-negative
  - Invalid status codes rejected (hard enforcement)

**Contract:** [`contracts/orders_contract.yaml`](contracts/orders_contract.yaml)

---

### Scenario 3: IoT Thermostat (20 points)

**Problem:** Sensor failures send default value (9999°C), skewing analytics

**Solution:**

- **Range Validation:**
  - Temperature: [-30°C, 60°C]
  - Battery level: [0.0, 1.0]
- **Quality Rules:** Hard enforcement to reject invalid readings

**Contract:** [`contracts/thermostat_contract.yaml`](contracts/thermostat_contract.yaml)

---

### Scenario 4: FinTech Transactions (20 points)

**Problem:** Invalid account IDs cause silent fraud detection failures

**Solution:**

- **Regex Pattern:** `^[A-Z0-9]{10}$` (exactly 10 uppercase alphanumeric)
- **Hard Circuit Breaker:** Pipeline blocks on violation
- **Quality Rules:** Both source and destination account IDs validated

**Contract:** [`contracts/fintech_contract.yaml`](contracts/fintech_contract.yaml)

---

## 🔍 Validation Script Features

The `scripts/validate_contracts.py` script performs:

### Structural Validation

- ✅ ODCS specification version
- ✅ Required sections (info, schema, sla, quality)
- ✅ Metadata completeness (owner, contact, classification)

### Scenario-Specific Validation

- ✅ **Scenario 1:** PII tagging, SLA freshness, quality rules
- ✅ **Scenario 2:** Enum mapping, status code rejection
- ✅ **Scenario 3:** Temperature/battery range checks
- ✅ **Scenario 4:** Regex pattern, hard enforcement

### Output

- Detailed pass/fail for each check
- Timestamped validation report
- Error summary with actionable feedback

---

## 🎯 Key Takeaways

### Data Contracts Provide:

1. **Decoupling:** Consumers isolated from physical schema changes
2. **Quality Assurance:** Automated validation at pipeline boundaries
3. **Governance:** PII tagging, ownership, and compliance
4. **Reliability:** Circuit breakers prevent cascading failures

### Best Practices Demonstrated:

- ✅ Logical vs. physical separation
- ✅ Semantic clarity in field descriptions
- ✅ Executable quality rules with thresholds
- ✅ Hard vs. soft enforcement policies
- ✅ Comprehensive validation infrastructure

---

## 📚 References

- [Open Data Contract Standard (ODCS)](https://github.com/bitol-io/open-data-contract-standard)
- [Data Mesh Principles](https://www.datamesh-architecture.com/)
- [yamllint Documentation](https://yamllint.readthedocs.io/)

---

## 📧 Contact

**Laksh Mendpara**  
Roll: B23CS1037  
Course: ML-DL-Ops (CSL7120)
