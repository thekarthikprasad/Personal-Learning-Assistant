# Personal Learning Assistant
A GenAI based personal learning assistant.

A learning assistant to assist in conceptual learning.

## Features

- Extracts concepts from the given document and their relationships.
- Generates quizzes based on the concepts.
- Evaluates answers using agentic workflows.
- Provides report about conceptual understanding of the given document.
- Finds weak areas of understanding.

## Set up

1. **`Clone the Repository`**:
   ```sh
   git clone https://github.com/thekarthikprasad/Personal-Learning-Assistant.git
   

2. **`Install requirements`**:
   ```sh
   pip install requirements.txt
   

3. **`Configure environment variables`**:
   ```sh
   GROQ_API_KEY_M = <Your groq key>
   GROQ_API_KEY_L = <Your groq key>
   GROQ_API_KEY_L1 = <Your groq key>
   NEO4J_URL = <Your neo4j url>
   NEO4J_PASS = <Your neo4j db password>
   
4. **`Run`**:
   ```sh
   cd Personal-learning-Assistant
   streamlit run app_v1.py
   
