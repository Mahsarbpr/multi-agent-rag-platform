from langgraph.graph import END, START, StateGraph

from rag_assistant.agents import ResearchAgent, AnalysisAgent, EvaluationAgent
from rag_assistant.graph.state import GraphState
from rag_assistant.llm.base_provider import LLMProvider


def create_rag_workflow(llm: LLMProvider):
    analysis_agent = AnalysisAgent(llm)
    evaluation_agent = EvaluationAgent(llm)

    def run_research(state: GraphState) -> GraphState:
        research_agent = ResearchAgent(
            llm=llm,
            vectorstore=state["vectorstore"],
        )

        research_result = research_agent.run(
            question=state["question"],
        )

        return {
            **state,
            "documents": research_result["documents"],
            "context": research_result["context"],
            "sources": research_result["sources"],
            "tool_used": research_result["tool_used"],
        }

    def run_analysis(state: GraphState) -> GraphState:
        answer = analysis_agent.run(
            question=state["question"],
            context=state["context"],
        )

        return {
            **state,
            "answer": answer,
        }

    def run_evaluation(state: GraphState) -> GraphState:
        evaluation = evaluation_agent.run(
            question=state["question"],
            context=state["context"],
            answer=state["answer"],
        )

        return {
            **state,
            "evaluation": evaluation,
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("research", run_research)
    workflow.add_node("analysis", run_analysis)
    workflow.add_node("evaluation", run_evaluation)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "evaluation")
    workflow.add_edge("evaluation", END)

    return workflow.compile()