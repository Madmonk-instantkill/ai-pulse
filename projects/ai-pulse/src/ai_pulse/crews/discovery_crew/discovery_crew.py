"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import PaperCollectionResult, RelevanceFilterResult
from ai_pulse.tools.discovery_tool import DiscoveryTool


@CrewBase
class DiscoveryCrew:
    """Discovery Crew: collects candidate AI/LLM papers from arXiv,
    OpenAlex, and Hugging Face, deduplicates them, and filters out papers
    that are not genuinely relevant AI/LLM research (spec section 4's
    negative exclusions).

    agents.yaml and tasks.yaml supply role/goal/backstory and
    description/expected_output as plain text; the string references in
    tasks.yaml (agent, context) are resolved automatically by CrewAI
    against the @agent/@task methods below with matching names.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def scout_agent(self) -> Agent:
        """The Scout Agent: collects and deduplicates candidate papers
        (Task 1), then judges each one's relevance (Task 2)."""
        return Agent(
            config=self.agents_config["scout_agent"],
            tools=[DiscoveryTool()],
            verbose=True,
        )

    @task
    def collect_and_deduplicate_task(self) -> Task:
        """Task 1: call the Paper Discovery tool and return the
        deduplicated candidate list. output_pydantic is set explicitly
        here (a real Python class reference, same reason as Task 2) so
        the Flow can reliably read the full paper list back afterward,
        not just free-form text."""
        return Task(
            config=self.tasks_config["collect_and_deduplicate_task"],
            output_pydantic=PaperCollectionResult,
        )

    @task
    def relevance_filter_task(self) -> Task:
        """Task 2: judge every paper from Task 1 for genuine AI/LLM
        relevance. output_pydantic is set explicitly here (as a real
        Python class reference) rather than as a YAML string, since a
        Pydantic class can't be written directly inside YAML."""
        return Task(
            config=self.tasks_config["relevance_filter_task"],
            output_pydantic=RelevanceFilterResult,
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the Discovery Crew: both tasks run in sequence, since
        Task 2 depends on Task 1's output (wired via tasks.yaml's
        context: field)."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
