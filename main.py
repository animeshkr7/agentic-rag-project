
import sys
from agent import Agent

def main():
    """
    Main function to run the agent.
    """
    agent = Agent()
    if len(sys.argv) > 1:
        query = sys.argv[1]
        response = agent.run(query)
        print(f"Agent: {response['messages'][-1].content}")
    else:
        print("Please provide a query as a command-line argument.")

if __name__ == "__main__":
    main()
