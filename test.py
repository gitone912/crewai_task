import os

os.environ["OPENAI_MODEL_NAME"]="gpt-4-turbo"
import requests
import json
from crewai import Agent
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

# Creating a senior researcher agent with memory and verbose mode
def get_movie_data():
    api_url = "https://www.omdbapi.com/?apikey=f0f8a529&s=comedy&type=movie"  # Example using TMDB API
    response = requests.get(api_url)
    if response.status_code == 200:
        data = json.loads(response.text)
        print(data["Search"])
        return data["Search"]  # Return list of movies
    else:
        print(f"Error fetching data: {response.status_code}")
        return None
    
data  = get_movie_data()

movie_agent = Agent(
    role="Movie Agent",
    goal="Use Movie data to get their description",
    verbose=True,
    memory=True,
    backstory=(
                "You use data of movies "
                "find description of every movie"

            ),
    tools=[search_tool],
    allow_delegation=True
)

# Create a GPT Agent
gpt_agent = Agent(
    role="GPT Recommender",
    goal="Recommend the best movie based on the comedy prefernce",
    verbose=True,
    memory=True,
    backstory=(
                "You use data of movies "
                "analyze description of every movie"
                "give one best movie"

            ),
    tools=[search_tool],
    allow_delegation=True
)

from crewai import Task

# Research task
research_task = Task(
  description=(
    "Get description of these movie "+ str(data)
  ),
  expected_output='5 movies name and their description formatted as markdown.',
  agent=movie_agent,
  async_execution=False,
  output_file='5_best.md'
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "analyze every movie from this list and recommend one best movie." + str(data)
  ),
  expected_output='A best movie  formatted as markdown.',
  agent=gpt_agent,
  async_execution=False,
  output_file='one_best.md'  # Example of output customization
)

from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[gpt_agent, movie_agent],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(result)