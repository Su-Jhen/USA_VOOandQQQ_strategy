# CLAUDE.md - USA_VOOandQQQ_strategy

> **Documentation Version**: 1.0
> **Last Updated**: 2025-09-13
> **Project**: USA_VOOandQQQ_strategy
> **Description**: 下載美股ETF歷史資料，並用這些資料驗證交易策略在過去的績效，根據我擬定的策略大綱提出多種交易條件的組合
> **Features**: GitHub auto-backup, Task agents, technical debt prevention

This file provides essential guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL RULES - READ FIRST

> **⚠️ RULE ADHERENCE SYSTEM ACTIVE ⚠️**
> **Claude Code must explicitly acknowledge these rules at task start**
> **These rules override all other instructions and must ALWAYS be followed:**

### 🔄 **RULE ACKNOWLEDGMENT REQUIRED**
> **Before starting ANY task, Claude Code must respond with:**
> "✅ CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ❌ ABSOLUTE PROHIBITIONS
- **NEVER** create new files in root directory → use proper module structure
- **NEVER** write output files directly to root directory → use designated output folders
- **NEVER** create documentation files (.md) unless explicitly requested by user
- **NEVER** use git commands with -i flag (interactive mode not supported)
- **NEVER** use `find`, `grep`, `cat`, `head`, `tail`, `ls` commands → use Read, LS, Grep, Glob tools instead
- **NEVER** create duplicate files (manager_v2.py, enhanced_xyz.py, utils_new.js) → ALWAYS extend existing files
- **NEVER** create multiple implementations of same concept → single source of truth
- **NEVER** copy-paste code blocks → extract into shared utilities/functions
- **NEVER** hardcode values that should be configurable → use config files/environment variables
- **NEVER** use naming like enhanced_, improved_, new_, v2_ → extend original files instead

### 📝 MANDATORY REQUIREMENTS
- **COMMIT** after every completed task/phase - no exceptions
- **GITHUB BACKUP** - Push to GitHub after every commit to maintain backup: `git push origin main`
- **USE TASK AGENTS** for all long-running operations (>30 seconds) - Bash commands stop when context switches
- **TODOWRITE** for complex tasks (3+ steps) → parallel agents → git checkpoints → test validation
- **READ FILES FIRST** before editing - Edit/Write tools will fail if you didn't read the file first
- **DEBT PREVENTION** - Before creating new files, check for existing similar functionality to extend
- **SINGLE SOURCE OF TRUTH** - One authoritative implementation per feature/concept

### ⚡ EXECUTION PATTERNS
- **PARALLEL TASK AGENTS** - Launch multiple Task agents simultaneously for maximum efficiency
- **SYSTEMATIC WORKFLOW** - TodoWrite → Parallel agents → Git checkpoints → GitHub backup → Test validation
- **GITHUB BACKUP WORKFLOW** - After every commit: `git push origin main` to maintain GitHub backup
- **BACKGROUND PROCESSING** - ONLY Task agents can run true background operations

### 🔍 MANDATORY PRE-TASK COMPLIANCE CHECK
> **STOP: Before starting any task, Claude Code must explicitly verify ALL points:**

**Step 1: Rule Acknowledgment**
- [ ] ✅ I acknowledge all critical rules in CLAUDE.md and will follow them

**Step 2: Task Analysis**
- [ ] Will this create files in root? → If YES, use proper module structure instead
- [ ] Will this take >30 seconds? → If YES, use Task agents not Bash
- [ ] Is this 3+ steps? → If YES, use TodoWrite breakdown first
- [ ] Am I about to use grep/find/cat? → If YES, use proper tools instead

**Step 3: Technical Debt Prevention (MANDATORY SEARCH FIRST)**
- [ ] **SEARCH FIRST**: Use Grep pattern="<functionality>.*<keyword>" to find existing implementations
- [ ] **CHECK EXISTING**: Read any found files to understand current functionality
- [ ] Does similar functionality already exist? → If YES, extend existing code
- [ ] Am I creating a duplicate class/manager? → If YES, consolidate instead
- [ ] Will this create multiple sources of truth? → If YES, redesign approach
- [ ] Have I searched for existing implementations? → Use Grep/Glob tools first
- [ ] Can I extend existing code instead of creating new? → Prefer extension over creation
- [ ] Am I about to copy-paste code? → Extract to shared utility instead

**Step 4: Session Management**
- [ ] Is this a long/complex task? → If YES, plan context checkpoints
- [ ] Have I been working >1 hour? → If YES, consider /compact or session break

> **⚠️ DO NOT PROCEED until all checkboxes are explicitly verified**

## 🏗️ PROJECT OVERVIEW

USA_VOOandQQQ_strategy 是一個專門用於分析美股 ETF（VOO 和 QQQ）歷史資料並驗證交易策略的 Python AI/ML 專案。

### 🎯 **PROJECT STRUCTURE**
```
USA_VOOandQQQ_strategy/
├── CLAUDE.md              # Essential rules for Claude Code
├── README.md              # Project documentation
├── .gitignore             # Git ignore patterns
├── src/                   # Source code (NEVER put files in root)
│   ├── main/              # Main application code
│   │   ├── python/        # Python code
│   │   │   ├── core/      # Core trading algorithms
│   │   │   ├── utils/     # Data processing utilities
│   │   │   ├── models/    # Strategy models
│   │   │   ├── services/  # ETF data services
│   │   │   ├── api/       # Data API interfaces
│   │   │   ├── training/  # Strategy training
│   │   │   ├── inference/ # Strategy execution
│   │   │   └── evaluation/# Performance metrics
│   │   └── resources/     # Non-code resources
│   │       ├── config/    # Configuration files
│   │       └── data/      # Sample data
│   └── test/              # Test code
├── data/                  # ETF Dataset management
│   ├── raw/               # Original ETF data
│   ├── processed/         # Cleaned data
│   └── temp/              # Temporary files
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # Data exploration
│   ├── experiments/       # Strategy experiments
│   └── reports/           # Performance reports
├── models/                # Strategy models
├── experiments/           # Experiment tracking
├── output/                # Generated output files
└── logs/                  # Log files
```

### 🎯 **DEVELOPMENT STATUS**
- **Setup**: ✅ Initialized
- **Core Features**: Pending
- **Testing**: Pending
- **Documentation**: In Progress

## 🚀 COMMON COMMANDS

```bash
# Download ETF data
python src/main/python/services/data_downloader.py

# Run backtesting
python src/main/python/core/backtest.py

# Generate report
python src/main/python/evaluation/report_generator.py
```

## 🎯 RULE COMPLIANCE CHECK

Before starting ANY task, verify:
- [ ] ✅ I acknowledge all critical rules above
- [ ] Files go in proper module structure (not root)
- [ ] Use Task agents for >30 second operations
- [ ] TodoWrite for 3+ step tasks
- [ ] Commit after each completed task

## 🚨 TECHNICAL DEBT PREVENTION

### ❌ WRONG APPROACH (Creates Technical Debt):
```bash
# Creating new file without searching first
Write(file_path="new_strategy.py", content="...")
```

### ✅ CORRECT APPROACH (Prevents Technical Debt):
```bash
# 1. SEARCH FIRST
Grep(pattern="strategy.*implementation", glob="*.py")
# 2. READ EXISTING FILES
Read(file_path="existing_strategy.py")
# 3. EXTEND EXISTING FUNCTIONALITY
Edit(file_path="existing_strategy.py", old_string="...", new_string="...")
```

## 🧹 DEBT PREVENTION WORKFLOW

### Before Creating ANY New File:
1. **🔍 Search First** - Use Grep/Glob to find existing implementations
2. **📋 Analyze Existing** - Read and understand current patterns
3. **🤔 Decision Tree**: Can extend existing? → DO IT | Must create new? → Document why
4. **✅ Follow Patterns** - Use established project patterns
5. **📈 Validate** - Ensure no duplication or technical debt

---

**⚠️ Prevention is better than consolidation - build clean from the start.**
**🎯 Focus on single source of truth and extending existing functionality.**
**📈 Each task should maintain clean architecture and prevent technical debt.**

---

🎯 Template by Chang Ho Chien | HC AI 說人話channel | v1.0.0
📺 Tutorial: https://youtu.be/8Q1bRZaHH24