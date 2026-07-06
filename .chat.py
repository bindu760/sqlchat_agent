import os
import sqlite3
from typing import Literal

# Core imports
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# === Groq Setup ===
from langchain_groq import ChatGroq

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

# ====================== 1. CREATE STUDENT DATABASE ======================
def create_student_database():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS Enrollments;
        DROP TABLE IF EXISTS Students;
        DROP TABLE IF EXISTS Courses;
        DROP TABLE IF EXISTS Departments;

        CREATE TABLE Departments (department_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE Students (student_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, gender TEXT, department_id INTEGER);
        CREATE TABLE Courses (course_id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER);
        CREATE TABLE Enrollments (enrollment_id INTEGER PRIMARY KEY, student_id INTEGER, course_id INTEGER, grade TEXT);
    """)

    cursor.executescript("""
        INSERT INTO Departments VALUES (1, 'Computer Science'), (2, 'Mathematics');
        INSERT INTO Students VALUES 
        (1, 'Emma Wilson', 20, 'F', 1),
        (2, 'Liam Garcia', 21, 'M', 1),
        (3, 'Olivia Martinez', 19, 'F', 2),
        (4, 'Noah Chen', 22, 'M', 1);
        INSERT INTO Courses VALUES 
        (1, 'Python Programming', 1),
        (2, 'Data Structures', 1),
        (3, 'Calculus I', 2);
        INSERT INTO Enrollments VALUES 
        (1,1,1,'A'), (2,2,1,'B+'), (3,3,3,'A'), (4,4,2,'A-');
    """)
    conn.commit()
    conn.close()
    print("✅ Student database created successfully!")


# ====================== 2. SQL TOOLS ======================
@tool
def sql_db_list_tables() -> str:
    """List all tables in the database."""
    con = sqlite3.connect("students.db")
    try:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return ", ".join(row[0] for row in cursor.fetchall())
    finally:
        con.close()


@tool
def sql_db_schema(table_names: str) -> str:
    """Get schema for the given tables."""
    con = sqlite3.connect("students.db")
    try:
        cursor = con.cursor()
        results = []
        for table in [t.strip() for t in table_names.split(",") if t.strip()]:
            cursor.execute("SELECT sql FROM sqlite_master WHERE name=?;", (table,))
            schema = cursor.fetchone()
            if schema:
                results.append(schema[0])
        return "\n\n".join(results)
    finally:
        con.close()


@tool
def sql_db_query(query: str) -> str:
    """Execute SQL query and return the result."""
    con = sqlite3.connect("students.db")
    try:
        cursor = con.cursor()
        cursor.execute(query)
        return str(cursor.fetchall())
    except Exception as e:
        return f"Error: {e}"
    finally:
        con.close()


# ====================== 3. CREATE SQL AGENT ======================
def create_sql_agent():
    # === Groq Model ===
    # Get your key from https://console.groq.com/keys and set it as an
    # environment variable: export GROQ_API_KEY="your key"
    model = ChatGroq(
        model="llama-3.3-70b-versatile",   # other options: "llama-3.1-8b-instant", "mixtral-8x7b-32768"
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    tools = [sql_db_list_tables, sql_db_schema, sql_db_query]

    def list_tables(state: MessagesState):
        result = sql_db_list_tables.invoke("")
        return {"messages": [AIMessage(content=f"Available tables: {result}")]}

    def call_get_schema(state: MessagesState):
        schema_tool = next(t for t in tools if t.name == "sql_db_schema")
        llm = model.bind_tools([schema_tool], tool_choice="any")
        return {"messages": [llm.invoke(state["messages"])]}

    def generate_query(state: MessagesState):
        system_prompt = {
            "role": "system",
            "content": "You are a helpful SQL expert. Generate correct SQLite queries and use the sql_db_query tool."
        }
        llm = model.bind_tools([sql_db_query])
        return {"messages": [llm.invoke([system_prompt] + state["messages"])]}

    def should_continue(state: MessagesState) -> Literal[END, "run_query"]:
        last_message = state["messages"][-1]
        return "run_query" if getattr(last_message, 'tool_calls', None) else END

    # Build Graph
    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", ToolNode([next(t for t in tools if t.name == "sql_db_schema")]))
    builder.add_node("generate_query", generate_query)
    builder.add_node("run_query", ToolNode([sql_db_query]))

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("run_query", "generate_query")

    return builder.compile(checkpointer=InMemorySaver())


# ====================== 4. RUN THE AGENT ======================
if __name__ == "__main__":
    create_student_database()

    agent = create_sql_agent()

    print("\n" + "="*70)
    print("🎉 Groq SQL Agent is Ready!")
    print("Make sure you have set your GROQ_API_KEY environment variable.")
    print("="*70)

    questions = [
        "How many students are there?",
        "List all students with their grades and courses.",
        "Which department has more students?",
        "Show me students who got grade A."
    ]

    config = {"configurable": {"thread_id": "groq_agent"}}

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 60)

        inputs = {"messages": [HumanMessage(content=question)]}

        for event in agent.stream(inputs, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content.strip():
                print(last_msg.content)

        print("-" * 60)