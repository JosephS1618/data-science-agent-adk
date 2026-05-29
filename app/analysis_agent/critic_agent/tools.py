import uuid
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content, Part

def invoke_critic_agent(evaluation_prompt: str) -> str:
    """
    Invokes the Critic Agent natively to evaluate an execution trace.
    You must pass the evaluation_prompt constructed from the trace template.
    """
    # Lazy import to avoid circular dependencies
    from .agent import critic_agent
    
    session_service = InMemorySessionService()
    
    runner = Runner(
        agent=critic_agent,
        app_name="critic_agent_tool",
        session_service=session_service,
        auto_create_session=True
    )
    
    content = Content(role="user", parts=[Part(text=evaluation_prompt)])
    events = list(runner.run(
        user_id="internal_tool",
        session_id=str(uuid.uuid4()),
        new_message=content
    ))
    
    # Iterate backwards through events to find the agent's final text response
    for event in reversed(events):
        if hasattr(event, "get_text"):
            text = event.get_text()
            if text:
                return text
                
    return "[REJECTED] Error: Critic agent returned no text response."
