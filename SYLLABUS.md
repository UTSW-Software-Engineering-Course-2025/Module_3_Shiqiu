# Syllabus
## 🧬 LLMOps: A Practical Introduction to Software Engineering with LLMs

**Dates:** June 2-5, 2025 

**Room Assignments**:

- **Monday:** G9.250A
- **Tuesday - Thursday:** NL03.120

**[Daily Schedule](#-daily-schedule)**
  - [Day 1: Local LLMs & Test Harness Setup](#-day-1-monday-local-llms-test-harness-setup-basic-benchmarking--devops-intro)
  - [Day 2: APIs & Structured Outputs](#-day-2-tuesday-proprietary-apis-structured-outputs-advanced-benchmarking-gene-hop--evaluation)
  - [Day 3: Tool Use & RAG Concepts](#️-day-3-wednesday-tool-use-rag-concepts--code-interpreter)
  - [Day 4: Refinement & Demos](#-day-4-thursday-refinement-advanced-integration-documentation--demos)

**[Grading Rubric](#-grading-rubric-20-points-total)**


## 👨‍🏫 Instructors & Contact Info

| 👤 **Instructor**      | 📧 **Email**                           |
|:------------------------|:---------------------------------------|
| Andrew Jamieson        | Andrew.jamieson@utsouthwestern.edu     |
| Michael Holcomb        | Michael.Holcomb@UTSouthwestern.edu     |
| David Hein             | david.hein@UTSouthwestern.edu          |
| Ameer Hamza Shakur     | AmeerHamza.Shakur@UTSouthwestern.edu   |

---

## 📖 Course Overview

This module provides a **practical, hands-on introduction** to large language models (LLMs) in biomedical research and software development. 

Students will incrementally build a single project—a **"BioLLM Explorer"** (inspired by concepts like GeneGPT)—throughout the four days. 

### 🔑 Key Components:
- 🖥️ Setting up local LLM serving (Ollama on BioHPC)
- 🔌 Interacting with local and proprietary LLM APIs
- 🛠️ Implementing tool use for accessing external biomedical data (e.g., NCBI)
- ⚙️ Applying sound software engineering practices
- 📊 Version control, testing, MLflow for experiment tracking
- 🏗️ Modular design, documentation, and evaluation

The module emphasizes **independent problem-solving** with a focus on building a robust "test harness" and iteratively developing capabilities.

---

## 🎯 Learning Objectives

By the end of this module, students will be able to:

- ✅ Set up and run local LLMs via an API (Ollama on BioHPC)
- ✅ Develop a "test harness" to benchmark LLM performance on biomedical tasks (e.g., GeneTuring, Gene Hop)
- ✅ Utilize MLflow for experiment tracking and results logging
- ✅ Integrate proprietary LLM APIs (e.g., OpenAI/Azure) and implement best practices (key management, retries)
- ✅ Implement structured outputs from LLMs (schema-based vs. function calling)
- ✅ Design and implement "tool use" for LLMs to interact with external data sources (NCBI APIs, web search, code interpreters)
- ✅ Apply software engineering best practices: Git, unit/integration testing (pytest), modular design, AI IDE usage
- ✅ Understand and apply various LLM evaluation metrics (exact match, semantic similarity, LLM-as-a-judge concepts)
- ✅ Create maintainable documentation and deliver a project demonstration

---

## 🛠️ Tools & Infrastructure

### 💻 **Development Environment**
- **Python:** 3.10+
- **Environment:** venv
- **IDE:** VSCode (or other AI-assisted IDEs)

### 🔧 **Core Tools**
- **Version Control:** GitHub
- **Compute Resources:** **BioHPC CPU nodes**
- **Local LLM Serving:** **Ollama** (with models like Gemma 3, Granite 3.3)
- **Experiment Tracking:** **MLflow** (initially local, potentially a shared server)

### 📚 **Python Libraries**
- `transformers`
- `sentence-transformers`
- `pytest`
- `OpenAI API`
- **`NCBI Entrez API`**
- `httpx/requests`
- `MLflow client`

### 📦 **Environment Management**
- `uv + pyproject.toml`

---

## 🔬 Project Framework: BioLLM Explorer

Students will build a **command-line tool** and associated "test harness" to evaluate LLM performance on specific biomedical question-answering tasks (e.g., "GeneTuring," "Gene Hop").

### 🏗️ **Project Components:**

1. **🖥️ Local LLM Setup** - Setting up local LLM serving with Ollama on BioHPC
2. **📡 API Integration** - Creating scripts to call local and remote LLM APIs
3. **📊 Benchmarking** - Implementing functions to process benchmark datasets and log results to MLflow
4. **📋 Structured Outputs** - Developing structured output parsing
5. **🔗 Tool Integration** - Integrating various "tools" (e.g., NCBI API clients, web search functions, a Python code interpreter)

Students will progressively build and evaluate their system, applying software engineering best practices throughout.

---

## 📅 Daily Schedule

### 📝 Day 1 (Monday): Local LLMs, Test Harness Setup, Basic Benchmarking & DevOps Intro
**🏢 Room: G9.250A**

#### 🕘 **Morning Sessions**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **9:00–9:45 AM** | 📖 **Course Overview & LLM Fundamentals** (Lecture)<br/>• Broad overview of LLMs, key concepts, model landscape<br/>• [📊 Lecture Slides](https://gamma.app/docs/LLMOps-SWE-Intro-Lecture-1-June-2025-py134s14e8pgphw) | Andrew |
| **9:45–10:30 AM** | 🚀 **Project Introduction & Day 1 Task: GeneTuring Test Harness** (Lecture/Demo)<br/>• Overall project vision: Biomedical Text Analysis Assistant<br/>• Day 1 Goal: Set up a test harness for the "GeneTuring" benchmark<br/>• Brief on basic evaluation (e.g., exact match for GeneTuring) & MLflow setup | Mike |

#### 🔬 **Lab Session**
| Time | Activity |
|:-----|:---------|
| **10:30 AM–2:00 PM** | **🧪 Lab/Self-study** (with lunch break)<br/>• Access BioHPC CPU nodes<br/>• Set up Ollama with a provided model (e.g., Gemma 3)<br/>• Develop Python scripts to:<br/>  - Call the local Ollama API<br/>  - Process the "GeneTuring" dataset<br/>  - Log prompts, responses, and basic evaluation scores to a local MLflow instance<br/>• Initial function wrapping and basic code structure |

#### 🕒 **Afternoon Session**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **2:00–3:00 PM** | ⚙️ **DevOps Concepts & Git Best Practices** (Lecture)<br/>• Git workflows (branching, merging, PRs)<br/>• Introduction to unit testing<br/>• Importance of modular design<br/>• *(Students apply Git to their Day 1 work post-lecture or start of Day 2 lab)*<br/>• [📄 Lecture Slides](day1/DevOps_v1_02JUN2025.pdf) | Andrew |

---

### 🔗 Day 2 (Tuesday): Proprietary APIs, Structured Outputs, Advanced Benchmarking (Gene Hop) & Evaluation
**🏢 Room: NL03.120**

#### 🕘 **Morning Sessions**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **9:00–10:00 AM** | 📊 **Structured Outputs & API Best Practices** (Lecture)<br/>• Schema-based structured outputs vs. function calling (pros/cons)<br/>• Integrating proprietary APIs (OpenAI/Azure): key management, .env files, retry logic, error handling | Dave |
| **10:00–10:30 AM** | 🧬 **Day 2 Task: "Gene Hop" Benchmark & Frontier Models** (Introduction)<br/>• Introduce the more complex "Gene Hop" dataset requiring multi-hop reasoning<br/>• Discuss using frontier models (via API) for more challenging tasks | Mike |

#### 🔬 **Lab Session**
| Time | Activity |
|:-----|:---------|
| **10:30 AM–3:00 PM** | **🧪 Lab/Self-study** (with lunch break)<br/>• Integrate proprietary LLM APIs (e.g., OpenAI) into their test harness<br/>• Implement structured output generation/parsing for "GeneTuring" and/or "Gene Hop" tasks<br/>• Extend their test harness to run the "Gene Hop" benchmark, logging to MLflow<br/>• Write unit tests for API interaction and data processing logic<br/>• Compare local vs. proprietary model performance on tasks |

#### 🕒 **Afternoon Session**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **3:00–4:00 PM** | 📈 **Advanced Evaluation Metrics & LLM-as-a-Judge Teaser** (Mini-Lecture)<br/>• Beyond exact match: fuzzy matching, semantic similarity (e.g., BERTscore)<br/>• Concept of LLM-as-a-judge for more nuanced evaluation | Mike |

---

### 🛠️ Day 3 (Wednesday): Tool Use, RAG Concepts & Code Interpreter
**🏢 Room: NL03.120**

#### 🕘 **Morning Session**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **9:00–10:30 AM** | 🔧 **LLM Tool Use & RAG as a Tool** (Lecture/Demo)<br/>• Core concepts of LLM agents and tool use<br/>• Designing tools: NCBI API wrappers, web search functions<br/>• RAG: Conceptually as retrieval from existing knowledge bases (APIs) or local stores<br/>• Introduction to using a local Python code interpreter as an LLM tool | Mike/Ameer |

#### 🔬 **Lab Session**
| Time | Activity |
|:-----|:---------|
| **10:30 AM–3:00 PM** | **🧪 Lab/Self-study** (with lunch break)<br/>• Develop and integrate "tools" into their assistant for the "Gene Hop" task:<br/>  - Wrapper for NCBI Entrez API (e.g., to search PubMed or Gene databases)<br/>  - Potentially a simple web search tool<br/>• Modify prompts/logic to enable the LLM to utilize these tools<br/>• **🌟 Optional/Bonus:** Set up and integrate a local Python code interpreter (with BioPython) as a tool<br/>• **🌟 Optional/Bonus:** If very advanced, build a small custom vector DB for a specific set of documents and use it as a retrieval tool<br/>• Evaluate the impact of tool use on "Gene Hop" performance, log to MLflow |

#### 🕒 **Afternoon Session**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **3:00-4:00 PM** | 👥 **Peer Code Review / Q&A / TA Session** | Instructor Team |

---

### 🎯 Day 4 (Thursday): Refinement, Advanced Integration, Documentation & Demos
**🏢 Room: NL03.120**

#### 🕘 **Morning Session**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **9:00–9:30 AM** | ❓ **Q&A / Optional Advanced Topic Demo**<br/>(e.g., complex agent loops, advanced MLflow usage) | Instructor Team |

#### 🔬 **Lab Session**
| Time | Activity |
|:-----|:---------|
| **9:30 AM–3:00 PM** | **🧪 Lab/Wrap-up & Demo Preparation** (with lunch break)<br/>• Refine project: improve code quality, modularity, error handling, tests<br/>• Finalize MLflow logging and experiment comparison<br/>• Complete README documentation: project description, setup, how to run, design choices, key findings/evaluation results<br/>• Prepare final demonstration |

#### 🎭 **Final Presentations**
| Time | Session | Instructor |
|:-----|:--------|:-----------|
| **3:00-5:00 PM** | 🎯 **Final Demonstrations & Code Submission**<br/>• Students present their "BioLLM Explorer" and their findings<br/>• Code submission | Instructor Team |

---

## 📊 Grading Rubric (20 points total)

| 📋 **Category** | 🎯 **Points** | 📝 **Details** |
|:----------------|:---------------|:----------------|
| 🖥️ **Local LLM (Ollama) & Test Harness Setup** | **4** | Successfully sets up Ollama on BioHPC, calls local API, processes "GeneTuring" task, logs to MLflow, basic Git usage. |
| 🔗 **Proprietary API & Structured Output** | **4** | Integrates proprietary API, implements structured outputs, handles API keys securely, robust API calling (retries). |
| 🛠️ **Tool Use Implementation & "Gene Hop" Task** | **4** | Implements ≥1 LLM tool (e.g., NCBI API, web search), integrates for "Gene Hop" task, evaluates tool impact. |
| ⚙️ **Software Engineering/DevOps** | **4** | Effective Git use (branches, meaningful commits), 3-5 meaningful unit/integration tests (pytest), modular code design. |
| 📋 **Documentation, Evaluation & Demo** | **4** | Clear README, effective demonstration, explains design choices, discusses evaluation methods applied and results from MLflow for different tasks. |

---

## ⚠️ Implementation Note

> **🎯 Focus on Quality Over Complexity**
> 
> Focus on a working "test harness" and demonstrating core concepts: local/remote API calls, MLflow, structured outputs, and basic tool use. Quality of implementation, testing, and clear articulation of evaluation are key. Advanced tool integration, RAG knowledge base building, or even GUIs are stretch goals.

---

## 🔒 Ethics & Data Privacy

*[Content to be provided]*

---

## 📊 Dataset Information

Students will primarily work with benchmark datasets like **"GeneTuring" and "Gene Hop"** (details/links to be provided). 

### 🔬 **Key Data Sources:**
- **Benchmark Datasets:** GeneTuring, Gene Hop
- **External APIs:** Public NCBI databases (PubMed, Gene, etc.) via their Entrez API
- **Local Models:** Ollama models (e.g., Gemma 3, Granite 3.3) for local LLM serving

*Pending BioHPC availability*

---

## 📚 Pre-Readings & Preparation (Optional but Encouraged)

### 📖 **Core Readings:**
- 📄 **Attention Is All You Need** (Vaswani et al.)
- 📘 **OpenAI API Documentation** - Chat Completions and Embeddings endpoints
- 🔬 **NCBI Entrez Programming Utilities Documentation**
- 🖥️ **Ollama Documentation** (especially API interaction)
- 📊 **MLflow Quickstart/Documentation** (Python API)

### 🔧 **Technical Preparation:**
- 🤗 **Hugging Face Transformers Quickstart** (conceptual)
- 🧪 **PyTest Introduction**

### 📑 **Research Papers (Optional):**
- 🧬 **Gene GPT Paper** (Jin et al., 2023) 
  - [📄 Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153281/)
  - [💻 Code](https://github.com/ncbi/GeneGPT)

---

## 📋 Project Guide

For detailed information about the project, please see the [📋 Project Overview Guide](README.md).

---

*Last updated: Course Syllabus v1.0*
