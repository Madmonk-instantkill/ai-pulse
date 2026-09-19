"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import RelevanceFilterResult


@CrewBase
class DiscoveryCrew:
    """Discovery Crew: judges whether each candidate paper genuinely
    belongs in an AI/LLM digest (spec section 4's negative exclusions).

    Collecting and deduplicating the papers is plain Python (main.py calls
    DiscoveryTool directly). An earlier version had the agent call the
    tool and re-type the whole list as its answer, but a ~100-paper answer
    is long enough to be cut off mid-way. Now the crew only receives a
    small batch per call via kickoff(inputs={"papers": ...}) and returns
    short keep/discard decisions.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def scout_agent(self) -> Agent:
        """The Scout Agent: pure LLM reasoning, no tools -- judging
        whether a paper is genuinely about AI/LLM research requires
        reading its title and abstract."""
        return Agent(
            config=self.agents_config["scout_agent"],
            verbose=True,
        )

    @task
    def relevance_filter_task(self) -> Task:
        """Judge every paper in the batch passed in via
        kickoff(inputs={...}). output_pydantic is set here (as a real
        Python class reference) rather than in YAML, since a Pydantic
        class can't be written directly inside YAML."""
        return Task(
            config=self.tasks_config["relevance_filter_task"],
            output_pydantic=RelevanceFilterResult,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
