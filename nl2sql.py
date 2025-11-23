import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_agent  # ✅ The standard v1.0 method

# ==========================================
# STEP 1: DataFrame to SQLite File
# ==========================================

data = {
    'id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'department': ['HR', 'IT', 'IT', 'Marketing'],
    'salary': [55000, 85000, 90000, 62000]
}
df = pd.DataFrame(data)

engine = create_engine("sqlite:///local_database.db")
df.to_sql("staff", engine, index=False, if_exists='replace')
print("✅ DataFrame saved to 'local_database.db'")

# ==========================================
# STEP 2: Setup LLM
# ==========================================

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3-4b-thinking-2507",
    temperature=0
)

# ==========================================
# STEP 3: Create Tools
# ==========================================

db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# ==========================================
# STEP 4: Create Agent (using create_agent)
# ==========================================

# Define a system prompt to guide the agent's behavior for SQL tasks
system_prompt = """You are a helpful assistant designed to interact with a SQL database.
You have access to tools to list tables, get schema, and run queries.
Given a user question, use these tools to find the answer.
ALWAYS check the schema of relevant tables before running a query.
"""

# ✅ create_agent replaces the old specific factories (like create_react_agent)
# and automatically handles the execution loop (replacing AgentExecutor).
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# ==========================================
# STEP 5: Run Query
# ==========================================

query = "Department with maximum average salary?"
print(f"\n🤖 User: {query}")

try:
    # ✅ Invocation now expects a standard 'messages' format
    # The input is a list of messages, and the output state contains the full history
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract the final response from the last message in the history
    print(f"\n✅ Answer: {result['messages'][-1].content}")
    
except Exception as e:
    print(f"Error: {e}")