from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class DoctorPatient:
    """DoctorPatient crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # from config/agents.yaml
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],  # from config/agents.yaml
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # from config/tasks.yaml
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # from config/tasks.yaml
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DoctorPatient crew"""
        return Crew(
            agents=self.agents,   # from @agent methods
            tasks=self.tasks,     # from @task methods
            process=Process.sequential,
            verbose=True,
        )
