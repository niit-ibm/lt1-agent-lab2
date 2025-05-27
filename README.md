# Mosaic AI Agent Evaluation Lab Exercise

## Lab Overview
This hands-on lab will guide you through creating, implementing, and evaluating AI agents using Databricks Mosaic AI. You'll learn to build tool-calling agents and evaluate their performance using MLflow's LLM-as-a-Judge approach.

**Duration:** 2-3 hours  
**Prerequisites:** Basic Python knowledge, Databricks workspace access  
**Reference Documentation:** [Databricks Agent Evaluation Guide](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/)

---

## Task 1: Install and Setup the Libraries

### Objective
Set up the necessary environment and libraries for building and evaluating AI agents with Mosaic AI.

### Instructions

#### 1.1 Install Required Libraries
```python
# Install core libraries for agent development and evaluation
%pip install --upgrade databricks-agents mlflow openai langchain langchain-community
%pip install --upgrade databricks-vectorsearch databricks-sdk
%pip install --upgrade pandas numpy matplotlib seaborn

# Restart Python kernel after installation
dbutils.library.restartPython()
```

#### 1.2 Import Essential Libraries
```python
import mlflow
import pandas as pd
import numpy as np
from databricks.agents import CodeInterpreterTool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import BaseMessage
from langchain_community.llms import Databricks
import json
import logging
from typing import List, Dict, Any
```

#### 1.3 Configure MLflow Environment
```python
# Set MLflow tracking URI
mlflow.set_tracking_uri("databricks")

# Create or set experiment for agent evaluation
experiment_name = "/Users/{username}/agent_evaluation_lab"
mlflow.set_experiment(experiment_name)

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("Current experiment:", mlflow.get_experiment_by_name(experiment_name))
```

#### 1.4 Initialize Databricks LLM
```python
# Initialize the Databricks LLM endpoint
llm = Databricks(
    endpoint_name="databricks-dbrx-instruct",  # Adjust based on your available endpoints
    max_tokens=1000,
    temperature=0.1
)

# Test the connection
test_response = llm.invoke("Hello, this is a test connection.")
print("LLM Connection Test:", test_response)
```

### Verification Checklist
- [ ] All libraries installed successfully
- [ ] MLflow experiment created
- [ ] Databricks LLM endpoint connected
- [ ] No import errors

---

## Task 2: Create a Simple Tool Calling Agent

### Objective
Build a functional agent that can use custom tools to perform specific tasks and interact with external resources.

### Instructions

#### 2.1 Define Custom Tools
```python
def calculator_tool(expression: str) -> str:
    """
    A simple calculator tool that evaluates mathematical expressions.
    """
    try:
        # Basic safety check for eval
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def weather_lookup_tool(city: str) -> str:
    """
    A mock weather lookup tool (simulated data).
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 65°F", 
        "tokyo": "Rainy, 68°F",
        "san francisco": "Foggy, 61°F"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Current weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}"

def data_analysis_tool(data_description: str) -> str:
    """
    A mock data analysis tool that provides insights.
    """
    # Simulate data analysis results
    insights = [
        "Trend analysis shows 15% increase over last quarter",
        "Correlation coefficient of 0.82 between variables X and Y",
        "Outliers detected in 3% of the dataset",
        "Statistical significance (p < 0.05) confirmed"
    ]
    
    return f"Analysis of {data_description}: {np.random.choice(insights)}"
```

#### 2.2 Create LangChain Tools
```python
from langchain.tools import Tool

# Define tools for the agent
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    ),
    Tool(
        name="WeatherLookup", 
        func=weather_lookup_tool,
        description="Get current weather information for a city. Input should be the city name."
    ),
    Tool(
        name="DataAnalysis",
        func=data_analysis_tool,
        description="Perform data analysis on described datasets. Input should be a description of the data to analyze."
    )
]

print(f"Created {len(tools)} tools for the agent:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
```

#### 2.3 Initialize the Agent
```python
from langchain.agents import initialize_agent, AgentType

# Create the agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

print("Agent initialized successfully with tool calling capabilities")
```

#### 2.4 Test the Agent
```python
# Test cases for the agent
test_queries = [
    "What's the weather like in New York?",
    "Calculate 25 * 4 + 10",
    "Analyze sales data from Q3 2024",
    "What's 15% of 200, and what's the weather in London?"
]

# Run test queries
test_results = []
for i, query in enumerate(test_queries):
    print(f"\n=== Test Query {i+1}: {query} ===")
    try:
        response = agent.run(query)
        test_results.append({
            "query": query,
            "response": response,
            "status": "success"
        })
        print(f"Response: {response}")
    except Exception as e:
        test_results.append({
            "query": query,
            "response": str(e),
            "status": "error"
        })
        print(f"Error: {e}")

# Display results summary
results_df = pd.DataFrame(test_results)
print("\n=== Test Results Summary ===")
print(results_df[['query', 'status']])
```

### Verification Checklist
- [ ] Custom tools defined and working
- [ ] Agent initialized with tools
- [ ] Agent can successfully call tools
- [ ] Test queries executed without critical errors

---

## Task 3: Use MLflow Evaluation LLM as a Judge Approach

### Objective
Implement MLflow's evaluation framework using LLM-as-a-Judge to assess agent responses for quality, relevance, and correctness.

### Instructions

#### 3.1 Define Evaluation Metrics
```python
import mlflow.evaluate

def create_judge_prompt(query: str, response: str, expected_tools: List[str] = None) -> str:
    """Create a structured prompt for LLM judge evaluation."""
    
    prompt = f"""
You are an expert evaluator assessing AI agent responses. Please evaluate the following interaction:

QUERY: {query}
AGENT RESPONSE: {response}

Please rate the response on the following criteria (1-5 scale, where 5 is excellent):

1. RELEVANCE: How well does the response address the original query?
2. ACCURACY: Is the information provided factually correct?
3. COMPLETENESS: Does the response fully answer the question?
4. CLARITY: Is the response clear and well-structured?
5. TOOL_USAGE: If tools were needed, were they used appropriately?

Provide your evaluation in the following JSON format:
{{
    "relevance": <score>,
    "accuracy": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "tool_usage": <score>,
    "overall_score": <average_score>,
    "reasoning": "<explanation of your evaluation>"
}}
"""
    return prompt

def llm_judge_evaluator(query: str, response: str) -> Dict[str, Any]:
    """Use LLM as a judge to evaluate agent responses."""
    
    judge_prompt = create_judge_prompt(query, response)
    
    try:
        # Get evaluation from LLM judge
        judge_response = llm.invoke(judge_prompt)
        
        # Parse JSON response
        # In practice, you might need more robust JSON parsing
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
        else:
            # Fallback scoring if JSON parsing fails
            evaluation = {
                "relevance": 3,
                "accuracy": 3,
                "completeness": 3,
                "clarity": 3,
                "tool_usage": 3,
                "overall_score": 3,
                "reasoning": "Failed to parse detailed evaluation"
            }
            
        return evaluation
        
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return {
            "relevance": 1,
            "accuracy": 1,
            "completeness": 1,
            "clarity": 1,
            "tool_usage": 1,
            "overall_score": 1,
            "reasoning": f"Evaluation failed: {str(e)}"
        }
```

#### 3.2 Create MLflow Custom Metrics
```python
from mlflow.metrics import make_genai_metric

# Define custom evaluation metrics
def relevance_metric(eval_df, model_output):
    """Custom relevance metric using LLM judge."""
    scores = []
    for _, row in eval_df.iterrows():
        evaluation = llm_judge_evaluator(row['inputs'], row[model_output])
        scores.append(evaluation.get('relevance', 1))
    return pd.Series(scores)

def accuracy_metric(eval_df, model_output):
    """Custom accuracy metric using LLM judge."""
    scores = []
    for _, row in eval_df.iterrows():
        evaluation = llm_judge_evaluator(row['inputs'], row[model_output])
        scores.append(evaluation.get('accuracy', 1))
    return pd.Series(scores)

def tool_usage_metric(eval_df, model_output):
    """Custom tool usage metric using LLM judge."""
    scores = []
    for _, row in eval_df.iterrows():
        evaluation = llm_judge_evaluator(row['inputs'], row[model_output])
        scores.append(evaluation.get('tool_usage', 1))
    return pd.Series(scores)

# Create MLflow GenAI metrics
relevance_genai_metric = make_genai_metric(
    name="relevance_judge",
    definition="Evaluates how relevant the agent response is to the query using LLM judge",
    grading_prompt=create_judge_prompt("", ""),
    examples=None,
    model="databricks-dbrx-instruct",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True
)
```

#### 3.3 Prepare Evaluation Dataset
```python
# Create evaluation dataset from our test results
eval_data = []
for result in test_results:
    if result['status'] == 'success':
        eval_data.append({
            'inputs': result['query'],
            'outputs': result['response']
        })

eval_df = pd.DataFrame(eval_data)
print(f"Created evaluation dataset with {len(eval_df)} examples")
print("\nSample evaluation data:")
print(eval_df.head())
```

#### 3.4 Run MLflow Evaluation
```python
# Define a wrapper function for our agent
def agent_model(inputs):
    """Wrapper function for MLflow evaluation."""
    try:
        if isinstance(inputs, pd.DataFrame):
            results = []
            for query in inputs['inputs']:
                response = agent.run(query)
                results.append(response)
            return results
        else:
            return agent.run(inputs)
    except Exception as e:
        return f"Error: {str(e)}"

# Run evaluation with MLflow
with mlflow.start_run(run_name="agent_evaluation_llm_judge"):
    
    # Log model parameters
    mlflow.log_param("agent_type", "ZERO_SHOT_REACT_DESCRIPTION")
    mlflow.log_param("llm_model", "databricks-dbrx-instruct")
    mlflow.log_param("num_tools", len(tools))
    mlflow.log_param("evaluation_method", "LLM_as_Judge")
    
    # Run evaluation
    evaluation_results = mlflow.evaluate(
        model=agent_model,
        data=eval_df,
        targets="outputs",  # Ground truth column (if available)
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[relevance_genai_metric],
        evaluator_config={
            "col_mapping": {
                "inputs": "inputs",
                "outputs": "outputs"
            }
        }
    )
    
    # Log custom metrics
    manual_evaluations = []
    for _, row in eval_df.iterrows():
        evaluation = llm_judge_evaluator(row['inputs'], row['outputs'])
        manual_evaluations.append(evaluation)
    
    # Calculate aggregate metrics
    avg_relevance = np.mean([e['relevance'] for e in manual_evaluations])
    avg_accuracy = np.mean([e['accuracy'] for e in manual_evaluations])
    avg_tool_usage = np.mean([e['tool_usage'] for e in manual_evaluations])
    avg_overall = np.mean([e['overall_score'] for e in manual_evaluations])
    
    mlflow.log_metric("avg_relevance_judge", avg_relevance)
    mlflow.log_metric("avg_accuracy_judge", avg_accuracy)
    mlflow.log_metric("avg_tool_usage_judge", avg_tool_usage)
    mlflow.log_metric("avg_overall_judge", avg_overall)
    
    print(f"Evaluation completed!")
    print(f"Average Relevance: {avg_relevance:.2f}")
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Tool Usage: {avg_tool_usage:.2f}")
    print(f"Average Overall Score: {avg_overall:.2f}")
```

### Verification Checklist
- [ ] LLM judge evaluation function created
- [ ] Custom MLflow metrics defined
- [ ] Evaluation dataset prepared
- [ ] MLflow evaluation run completed
- [ ] Metrics logged and visible in MLflow UI

---

## Task 4: Evaluating the Trajectory for Tool Calling

### Objective
Implement comprehensive trajectory evaluation to assess not just final outputs, but the entire reasoning and tool-calling process of the agent.

### Instructions

#### 4.1 Create Trajectory Capture System
```python
class TrajectoryCapture:
    """Capture and analyze agent decision-making trajectories."""
    
    def __init__(self):
        self.trajectories = []
    
    def capture_trajectory(self, query: str, agent_steps: List[Dict]) -> Dict:
        """Capture the full trajectory of agent execution."""
        trajectory = {
            "query": query,
            "steps": agent_steps,
            "num_steps": len(agent_steps),
            "tools_used": self._extract_tools_used(agent_steps),
            "reasoning_quality": self._assess_reasoning(agent_steps),
            "efficiency_score": self._calculate_efficiency(agent_steps)
        }
        
        self.trajectories.append(trajectory)
        return trajectory
    
    def _extract_tools_used(self, steps: List[Dict]) -> List[str]:
        """Extract tools used in the trajectory."""
        tools_used = []
        for step in steps:
            if 'tool' in step and step['tool']:
                tools_used.append(step['tool'])
        return tools_used
    
    def _assess_reasoning(self, steps: List[Dict]) -> float:
        """Assess the quality of reasoning in the trajectory."""
        # Simple heuristic: longer reasoning chains might indicate more thorough thinking
        total_reasoning_length = sum(len(step.get('thought', '')) for step in steps)
        return min(total_reasoning_length / 100, 5.0)  # Normalize to 0-5 scale
    
    def _calculate_efficiency(self, steps: List[Dict]) -> float:
        """Calculate efficiency score based on number of steps."""
        # Fewer steps with successful completion = higher efficiency
        if len(steps) == 0:
            return 0
        elif len(steps) <= 2:
            return 5.0
        elif len(steps) <= 4:
            return 4.0
        elif len(steps) <= 6:
            return 3.0
        else:
            return 2.0

# Initialize trajectory capture
trajectory_capture = TrajectoryCapture()
```

#### 4.2 Enhanced Agent with Trajectory Logging
```python
class TrajectoryLoggingAgent:
    """Agent wrapper that logs detailed trajectories."""
    
    def __init__(self, base_agent, trajectory_capture):
        self.base_agent = base_agent
        self.trajectory_capture = trajectory_capture
    
    def run_with_trajectory(self, query: str) -> Dict:
        """Run agent and capture detailed trajectory."""
        
        # Mock trajectory steps (in practice, you'd integrate with agent internals)
        steps = []
        
        try:
            # Simulate capturing agent's internal steps
            # This would require deeper integration with LangChain's agent execution
            
            # Step 1: Initial reasoning
            steps.append({
                "step_type": "reasoning",
                "thought": f"I need to analyze the query: '{query}' and determine what tools to use.",
                "action": None,
                "tool": None
            })
            
            # Determine expected tools based on query content
            expected_tools = []
            if any(word in query.lower() for word in ['calculate', 'math', 'compute', '*', '+', '-', '/']):
                expected_tools.append('Calculator')
            if any(word in query.lower() for word in ['weather', 'temperature', 'climate']):
                expected_tools.append('WeatherLookup')
            if any(word in query.lower() for word in ['analyze', 'data', 'analysis']):
                expected_tools.append('DataAnalysis')
            
            # Step 2: Tool selection and execution
            for tool in expected_tools:
                steps.append({
                    "step_type": "tool_call",
                    "thought": f"Using {tool} to help answer the query",
                    "action": f"call_{tool.lower()}",
                    "tool": tool
                })
            
            # Execute the actual agent
            response = self.base_agent.run(query)
            
            # Step 3: Final response generation
            steps.append({
                "step_type": "response",
                "thought": "Synthesizing information to provide final response",
                "action": "respond",
                "tool": None
            })
            
            # Capture trajectory
            trajectory = self.trajectory_capture.capture_trajectory(query, steps)
            
            return {
                "query": query,
                "response": response,
                "trajectory": trajectory,
                "success": True
            }
            
        except Exception as e:
            error_step = {
                "step_type": "error",
                "thought": f"Encountered error: {str(e)}",
                "action": "error_handling",
                "tool": None
            }
            steps.append(error_step)
            
            trajectory = self.trajectory_capture.capture_trajectory(query, steps)
            
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "trajectory": trajectory,
                "success": False
            }

# Create trajectory-enabled agent
trajectory_agent = TrajectoryLoggingAgent(agent, trajectory_capture)
```

#### 4.3 Trajectory Evaluation Metrics
```python
def evaluate_trajectory(trajectory: Dict) -> Dict[str, float]:
    """Comprehensive trajectory evaluation."""
    
    evaluation = {}
    
    # 1. Tool Selection Appropriateness
    query = trajectory['query'].lower()
    tools_used = trajectory['tools_used']
    
    # Determine expected tools
    expected_tools = set()
    if any(word in query for word in ['calculate', 'math', 'compute']):
        expected_tools.add('Calculator')
    if any(word in query for word in ['weather', 'temperature']):
        expected_tools.add('WeatherLookup')
    if any(word in query for word in ['analyze', 'data']):
        expected_tools.add('DataAnalysis')
    
    tools_used_set = set(tools_used)
    
    # Tool precision and recall
    if expected_tools:
        precision = len(tools_used_set & expected_tools) / len(tools_used_set) if tools_used_set else 0
        recall = len(tools_used_set & expected_tools) / len(expected_tools)
        tool_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        tool_f1 = 1.0 if not tools_used_set else 0.5  # No tools needed, penalize if tools used
    
    evaluation['tool_selection_score'] = tool_f1 * 5  # Scale to 1-5
    
    # 2. Efficiency Score (from trajectory capture)
    evaluation['efficiency_score'] = trajectory['efficiency_score']
    
    # 3. Reasoning Quality (from trajectory capture)  
    evaluation['reasoning_score'] = trajectory['reasoning_quality']
    
    # 4. Step Coherence (steps follow logical order)
    steps = trajectory['steps']
    coherence_score = 5.0
    
    # Check for logical step progression
    step_types = [step['step_type'] for step in steps]
    if 'reasoning' not in step_types:
        coherence_score -= 1.0
    
    # Tool calls should come after reasoning
    reasoning_index = step_types.index('reasoning') if 'reasoning' in step_types else -1
    tool_indices = [i for i, st in enumerate(step_types) if st == 'tool_call']
    
    if reasoning_index >= 0 and tool_indices:
        if any(ti <= reasoning_index for ti in tool_indices):
            coherence_score -= 1.0
    
    evaluation['coherence_score'] = max(coherence_score, 1.0)
    
    # 5. Overall trajectory score
    evaluation['overall_trajectory_score'] = np.mean([
        evaluation['tool_selection_score'],
        evaluation['efficiency_score'],
        evaluation['reasoning_score'],
        evaluation['coherence_score']
    ])
    
    return evaluation
```

#### 4.4 Run Comprehensive Trajectory Evaluation
```python
# Test queries with trajectory evaluation
trajectory_test_queries = [
    "What's 25 * 8 + 15?",
    "What's the weather in Tokyo and calculate 50% of 200?",
    "Analyze customer satisfaction data and tell me the weather in San Francisco",
    "Just say hello"  # Query that shouldn't need tools
]

trajectory_results = []

print("=== Running Trajectory Evaluation ===\n")

for i, query in enumerate(trajectory_test_queries):
    print(f"Query {i+1}: {query}")
    
    # Run agent with trajectory capture
    result = trajectory_agent.run_with_trajectory(query)
    
    # Evaluate trajectory
    trajectory_eval = evaluate_trajectory(result['trajectory'])
    
    # Combine results
    full_result = {
        **result,
        'trajectory_evaluation': trajectory_eval
    }
    
    trajectory_results.append(full_result)
    
    # Print summary
    print(f"  Response: {result['response'][:100]}...")
    print(f"  Tools Used: {result['trajectory']['tools_used']}")
    print(f"  Trajectory Score: {trajectory_eval['overall_trajectory_score']:.2f}")
    print(f"  Tool Selection: {trajectory_eval['tool_selection_score']:.2f}")
    print(f"  Efficiency: {trajectory_eval['efficiency_score']:.2f}")
    print()

# Create trajectory evaluation DataFrame
trajectory_eval_data = []
for result in trajectory_results:
    eval_row = {
        'query': result['query'],
        'response': result['response'],
        'success': result['success'],
        'num_steps': result['trajectory']['num_steps'],
        'tools_used': ', '.join(result['trajectory']['tools_used']),
        **{f"trajectory_{k}": v for k, v in result['trajectory_evaluation'].items()}
    }
    trajectory_eval_data.append(eval_row)

trajectory_df = pd.DataFrame(trajectory_eval_data)
print("=== Trajectory Evaluation Summary ===")
print(trajectory_df[['query', 'trajectory_overall_trajectory_score', 'trajectory_tool_selection_score', 'trajectory_efficiency_score']].round(2))
```

#### 4.5 Log Trajectory Results to MLflow
```python
# Log trajectory evaluation results
with mlflow.start_run(run_name="trajectory_evaluation"):
    
    # Log parameters
    mlflow.log_param("evaluation_type", "trajectory_analysis")
    mlflow.log_param("num_test_queries", len(trajectory_test_queries))
    
    # Calculate and log aggregate metrics
    avg_trajectory_score = trajectory_df['trajectory_overall_trajectory_score'].mean()
    avg_tool_selection = trajectory_df['trajectory_tool_selection_score'].mean()
    avg_efficiency = trajectory_df['trajectory_efficiency_score'].mean()
    avg_reasoning = trajectory_df['trajectory_reasoning_score'].mean()
    avg_coherence = trajectory_df['trajectory_coherence_score'].mean()
    
    mlflow.log_metric("avg_trajectory_score", avg_trajectory_score)
    mlflow.log_metric("avg_tool_selection_score", avg_tool_selection)
    mlflow.log_metric("avg_efficiency_score", avg_efficiency)
    mlflow.log_metric("avg_reasoning_score", avg_reasoning)
    mlflow.log_metric("avg_coherence_score", avg_coherence)
    
    # Log detailed results as artifact
    trajectory_df.to_csv("trajectory_evaluation_results.csv", index=False)
    mlflow.log_artifact("trajectory_evaluation_results.csv")
    
    # Create and log trajectory visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Agent Trajectory Evaluation Results')
    
    # Tool selection scores
    axes[0,0].bar(range(len(trajectory_df)), trajectory_df['trajectory_tool_selection_score'])
    axes[0,0].set_title('Tool Selection Scores')
    axes[0,0].set_ylabel('Score')
    
    # Efficiency scores
    axes[0,1].bar(range(len(trajectory_df)), trajectory_df['trajectory_efficiency_score'])
    axes[0,1].set_title('Efficiency Scores')
    axes[0,1].set_ylabel('Score')
    
    # Overall trajectory scores
    axes[1,0].bar(range(len(trajectory_df)), trajectory_df['trajectory_overall_trajectory_score'])
    axes[1,0].set_title('Overall Trajectory Scores')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_xlabel('Query Index')
    
    # Score distribution
    axes[1,1].hist(trajectory_df['trajectory_overall_trajectory_score'], bins=10, alpha=0.7)
    axes[1,1].set_title('Score Distribution')
    axes[1,1].set_xlabel('Overall Trajectory Score')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig("trajectory_evaluation_chart.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("trajectory_evaluation_chart.png")
    plt.show()
    
    print(f"\n=== Final Trajectory Evaluation Results ===")
    print(f"Average Trajectory Score: {avg_trajectory_score:.2f}")
    print(f"Average Tool Selection Score: {avg_tool_selection:.2f}")
    print(f"Average Efficiency Score: {avg_efficiency:.2f}")
    print(f"Average Reasoning Score: {avg_reasoning:.2f}")
    print(f"Average Coherence Score: {avg_coherence:.2f}")
```

### Verification Checklist
- [ ] Trajectory capture system implemented
- [ ] Enhanced agent with trajectory logging created
- [ ] Trajectory evaluation metrics defined
- [ ] Comprehensive evaluation completed
- [ ] Results logged to MLflow with visualizations



**Lab Complete!** You now have hands-on experience with comprehensive agent evaluation using Databricks Mosaic AI.
