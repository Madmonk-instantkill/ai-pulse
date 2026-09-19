"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import ExecutiveSummary


@CrewBase
class SummaryCrew:
    """Executive summary stage of Editorial Crew: reads every finished
    PaperSection and writes the report title plus a short introduction.

    Invoked once per run via crew().kickoff(inputs={"sections": ...}),
    where sections is the list of PaperSections serialized as JSON. The
    report date is not written here -- code adds it.
    """

    agents_config = "config/summary_agents.yaml"
    tasks_config = "config/summary_tasks.yaml"

    @agent
    def executive_summary_agent(self) -> Agent:
        """The Executive Summary Agent: pure LLM reasoning, no tools."""
        return Agent(
            config=self.agents_config["executive_summary_agent"],
            verbose=True,
        )

    @task
    def executive_summary_task(self) -> Task:
        """Write the title and intro from the sections passed in via
        kickoff(inputs={...})."""
        return Task(
            config=self.tasks_config["executive_summary_task"],
            output_pydantic=ExecutiveSummary,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
