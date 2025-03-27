import os
import openai
import pandas as pd

# Explicitly set OpenAI API Key
openai.api_key = " your key here "
# Import llama-index components
from llama_index.query_engine import PandasQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI

# Import your custom modules
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from pdf import canada_engine

def setup_rag_agent():
    # Load population data
    population_path = os.path.join("data", "population.csv")
    
    # Error handling for file loading
    try:
        population_df = pd.read_csv(population_path)
    except FileNotFoundError:
        print(f"Error: Population CSV file not found at {population_path}")
        return None
    except Exception as e:
        print(f"Error loading population data: {e}")
        return None

    # Setup query engine
    population_query_engine = PandasQueryEngine(
        df=population_df, 
        verbose=True, 
        instruction_str=instruction_str
    )
    population_query_engine.update_prompts({"pandas_prompt": new_prompt})

    # Prepare tools
    tools = [
        note_engine,
        QueryEngineTool(
            query_engine=population_query_engine,
            metadata=ToolMetadata(
                name="population_data",
                description="provides information on world population and demographics",
            ),
        ),
        QueryEngineTool(
            query_engine=canada_engine,
            metadata=ToolMetadata(
                name="canada_data",
                description="provides detailed information about Canada",
            ),
        ),
    ]

    # Initialize LLM and Agent
    try:
        llm = OpenAI(model="gpt-3.5-turbo-0613")
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
        return agent
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return None

def main():
    # Setup the agent
    agent = setup_rag_agent()
    
    if not agent:
        print("Failed to setup RAG agent. Exiting.")
        return

    # Interactive loop
    while True:
        try:
            prompt = input("Enter a prompt (q to quit): ")
            
            if prompt.lower() == 'q':
                break
            
            result = agent.query(prompt)
            print(result)
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()