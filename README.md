# MLflow Agent Evaluation Lab Exercise

A comprehensive educational lab designed to teach students how to evaluate AI agent responses using MLflow, focusing on tool calling capabilities, LLM-as-a-Judge evaluation, and trajectory analysis for modern AI systems.

## 🎯 Learning Objectives

By completing this lab, students will master:
- **MLflow Evaluation Framework**: Hands-on experience with MLflow's evaluation capabilities
- **Tool Calling Assessment**: How to evaluate AI agent tool selection and execution
- **LLM-as-a-Judge Methodology**: Using LLMs to evaluate other LLM responses
- **Trajectory Analysis**: Comprehensive evaluation of agent reasoning chains
- **Real-world Agent Evaluation**: Industry-standard techniques for AI system assessment

## 📚 Lab Structure (3 Progressive Steps)

### 🔍 Step 1: Tool Calling Evaluation for Product Search
**Duration**: 30 minutes  
**Objective**: Evaluate if the agent correctly identifies and calls appropriate tools for product search queries.

**Query Example**: `"I'm looking for an ergonomic wireless mouse"`

**Evaluation Criteria**:
- ✅ **Search Tool Called**: Did the agent invoke the product search function?
- ✅ **Product Found**: Was the correct product (P045) located in the database?  
- ✅ **Tool Parameters**: Were the search parameters correctly formatted?

**Learning Outcomes**:
- Understanding tool calling mechanics
- Evaluating tool selection accuracy
- Assessing parameter passing correctness

### 🔗 Step 2: Context Maintenance and Follow-up Queries  
**Duration**: 30 minutes  
**Objective**: Test the agent's ability to maintain context and handle follow-up questions.

**Follow-up Query**: `"Is the product available in stock?"`

**Evaluation Criteria**:
- ✅ **Context Preservation**: Was the P045 product context maintained?
- ✅ **Stock Tool Called**: Did the agent call the inventory checking function?
- ✅ **Accurate Information**: Was the stock information correctly retrieved and presented?

**Learning Outcomes**:
- Multi-turn conversation evaluation
- Context maintenance assessment
- Information accuracy verification

### 📊 Step 3: Comprehensive MLflow Evaluation
**Duration**: 60 minutes  
**Objective**: Perform complete response evaluation using MLflow metrics and custom scoring.

**Features**:
- **Individual Query Scoring** (1-5 scale)
- **Automated Feedback Generation**
- **MLflow Metrics Integration**
- **Toxicity and Relevance Assessment**
- **Custom Evaluation Metrics**

**Learning Outcomes**:
- MLflow evaluation framework mastery
- Custom metric development
- Automated assessment techniques
- Production-ready evaluation pipelines

## 🚀 Quick Start

### Prerequisites
- Basic Python programming knowledge
- Understanding of AI/LLM concepts
- Google Colab or local Python environment
- Replicate API account (free tier available)
