"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ai_pulse.schemas import PaperVerification


@CrewBase
class VerifierCrew:
    """Verifier stage of Editorial Crew: checks each claim in one paper's
    ledger against the text of the pages it cites.

    Invoked once per paper via crew().kickoff(inputs={"arxiv_id": ...,
    "claims": ..., "source_text": ...}), where claims is the ledger
    serialized as JSON and source_text comes from verification.py.
    """

    agents_config = "config/verifier_agents.yaml"
    tasks_config = "config/verifier_tasks.yaml"

    @agent
    def claim_verifier_agent(self) -> Agent:
        """The Claim Verifier Agent: pure LLM reasoning, no tools --
        it only compares the claims to the page text it is given."""
        return Agent(
            config=self.agents_config["claim_verifier_agent"],
            verbose=True,
        )

    @task
    def verify_claims_task(self) -> Task:
        """Verify the ledger passed in via kickoff(inputs={...})."""
        return Task(
            config=self.tasks_config["verify_claims_task"],
            output_pydantic=PaperVerification,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
