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

### ⚙️ Step 1: Environment Setup
**Duration**: 15 minutes  
**Objective**: Configure the development environment and establish API connections for agent evaluation.

**Setup Tasks**:
- 🔧 **Google Colab Configuration**: Install required packages and restart runtime
- 🔑 **Replicate API Token**: Set up authentication for LLM access
- 📦 **MLflow Initialization**: Configure experiment tracking and logging
- 🧪 **Connection Testing**: Verify all components are working correctly

**Key Components**:
```python
# Package installation
!pip install mlflow langchain-community replicate ibm-granite-community textstat

# API token configuration
os.environ['REPLICATE_API_TOKEN'] = 'your_token_here'

# MLflow experiment setup
mlflow.set_experiment("agent_evaluation_lab")


### ⚙️ Step 2: Defining Functions (Tools) and TechMart Product Data Structure
**Duration**: 10 minutes  
**Objective**: Build a comprehensive e-commerce tool suite and product database for TechMart's inventory system.

**Setup Tasks**:
- 🔧 **search_products: Intelligent product discovery with keyword matching
- 🔑 **get_product_info: Detailed product specifications and descriptions
- 📦 **get_price: Current pricing with promotional information
- 🧪 **get_delivery_info: Shipping costs and delivery timeframes

**TechMart Product Database Structure**:

```python
# 16 Products across 5 categories
PRODUCTS = [
    {
        "id": "P045",
        "name": "Wireless Mouse", 
        "category": "Electronics",
        "description": "Ergonomic Wireless Mouse with Rechargeable Battery",
        "price": 49.99,
        "stock": 18,
        "delivery": {
            "shipping_cost": 4.99,
            "free_shipping_threshold": 50,
            "estimated_days": "2-3"
        }
    },
    # ... 15 more products
]

**Tool Registry Implementation**:

```python
@tool
def search_products(query: str) -> str:
    """Advanced search with fuzzy matching and relevance scoring"""
    
@tool  
def check_stock(product_id: str) -> str:
    """Stock status with intelligent categorization"""

### 📊 Step 3: Complete Agent Evaluation Pipeline

**Duration**: 15 minutes  
**Objective**: Execute comprehensive evaluation including tool calling assessment, context maintenance, and MLflow integration.

#### 🔍 Step 3.1: Tool Calling Evaluation for Product Search
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

#### 🔗 Step 3.2: Context Maintenance and Follow-up Queries  
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

#### 📊 Step 3.3: Comprehensive MLflow Evaluation
**Objective**: Perform complete response evaluation using MLflow metrics and custom scoring.

**Features**:
- **Individual Query Scoring** (1-5 scale)
- **Automated Feedback Generation**
- **MLflow Metrics Integration**
- **Toxicity and Relevance Assessment**
- **Custom Evaluation Metrics**

MLflow Evaluation Output:

```python
FINAL EVALUATION SUMMARY:
Query 1: I'm looking for an ergonomic wireless mouse
Expected : Product P045 is a wireless mouse with ergonomic support and programmable buttons
Predicted: Based on your query: Found: Wireless Mouse (ID: P045)...
Score    : 5/5
Feedback : Excellent response with accurate product information

Average Score: 4.33/5
MLflow Metrics: {
  "toxicity/v1/mean": 0.00015,
  "exact_match/v1": 0.67,
  "custom_relevance_score": 4.5
}

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
