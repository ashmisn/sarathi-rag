import sys
import os
import json
import datetime
import traceback
from typing import List, Dict, Any, Optional # Added Optional

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------

try:
    # Use relative import assuming agent.py is in src/
    from . import config, tools, prompts
except ImportError:
    # Fallback for running script directly or different structure
    try:
        from src import config, tools, prompts
    except ImportError:
        import config
        import tools
        import prompts

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# NOTE: JsonOutputParser and OutputParserException are removed as they were for the reranker

# --- 1. Initialize the LLM ---
try:
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
    # reranker_llm is removed
    print(f"LLM initialized with model: {config.OLLAMA_MODEL}")
except Exception as e:
    print(f"--- ERROR initializing LLM: {e} ---")
    print(f"Ensure Ollama is running and the model '{config.OLLAMA_MODEL}' is pulled.")
    sys.exit(1)

# --- 2. Knowledge Gap Logger ---
def log_knowledge_gap(query: str):
    """Logs a user query that returned no results to the knowledge gap file."""
    try:
        log_dir = os.path.dirname(config.KNOWLEDGE_GAP_LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        with open(config.KNOWLEDGE_GAP_LOG_PATH, "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] {query}\n")
        print(f"Logged knowledge gap for query: {query}")
    except Exception as e:
        print(f"Error logging knowledge gap: {e}")
    # Return the specific message expected by downstream chains
    return "No information found in the database for this query."

# --- 3. The "General Query" Chain ---

# --- Planner Prompt ---
try:
    PLANNER_SYSTEM_PROMPT = prompts.PLANNER_SYSTEM_PROMPT
except AttributeError:
    print("--- ERROR: PLANNER_SYSTEM_PROMPT not found in prompts module ---")
    PLANNER_SYSTEM_PROMPT = "Error: Planner prompt missing." # Basic fallback

# --- execute_plan Function (MODIFIED - LLM RERANKING REMOVED) ---
def execute_plan(input_dict: dict) -> dict:
    """
    Executes planned search AND keyword search (with category hint),
    combines and deduplicates the results, and passes the top N
    directly to the synthesis step. (LLM Reranking REMOVED).
    Returns dict with context and query.
    """
    agent_plan = input_dict.get('plan', None)
    # Make sure to get the original query correctly passed through
    query_input = input_dict.get('query', "Query unavailable") # This is the 'dict' object

    # --- FIX for AttributeError: 'dict' object has no attribute 'lower' ---
    # Create a new 'query_string' variable for tools that expect text,
    # while preserving the 'query_input' dict for the final return value.
    if isinstance(query_input, dict):
        # Adjust 'text' or 'query' if your key is different
        if 'text' in query_input:
            query_string = query_input['text']
        elif 'query' in query_input:
            query_string = query_input['query']
        else:
            print(f"Error: Query is a dict but has no 'text' or 'query' key: {query_input}")
            query_string = "" # Set to empty string to avoid new errors
    else:
        query_string = str(query_input) # It's already a string or something else
    # --- END FIX ---

    if not agent_plan:
        print("Error: Agent Plan is missing.")
        # Use query_string for log_knowledge_gap
        return {"context": log_knowledge_gap(query_string), "query": query_input}

    planned_results = []
    tool_map = {
        "structured_search": tools.structured_search,
        "semantic_search": tools.semantic_search,
    }

    # --- Use a slightly larger k for initial retrieval ---
    INITIAL_K = 5 # Retrieve top 5 from each source initially

    print(f"\n--- Executing General Plan (Hybrid Search - No Reranking) ---")
    print(f"Planner Thought: {getattr(agent_plan, 'thought', 'N/A')}")

    # --- Step 1: Execute Planned Search (Semantic/Structured) ---
    plan_steps = getattr(agent_plan, 'plan', [])
    if not isinstance(plan_steps, list): plan_steps = []

    for tool_call in plan_steps:
        if not (hasattr(tool_call, 'tool_name') and hasattr(tool_call, 'arguments')):
                print(f"Warning: Invalid tool_call format: {tool_call}")
                continue
        tool_name = tool_call.tool_name
        arguments = tool_call.arguments
        tool_function = tool_map.get(tool_name)
        if tool_function:
            try:
                print(f"Calling (Planner): {tool_name} with args {arguments}")
                if not isinstance(arguments, dict):
                        print(f"Warning: Invalid arguments format for {tool_name}: {arguments}")
                        continue
                if tool_name == "semantic_search":
                    search_query = arguments.get('query')
                    if not search_query:
                            print("Warning: Missing 'query' argument for semantic_search.")
                            continue
                    # Fetch INITIAL_K results
                    result = tool_function(query=search_query, n_results=INITIAL_K)
                elif tool_name == "structured_search":
                    valid_args = {k: v for k, v in arguments.items() if k in ['problem', 'category', 'type'] and v}
                    if not valid_args:
                            print(f"Warning: No valid arguments provided for structured_search: {arguments}")
                            continue
                    # Structured search might return many, that's okay here
                    result = tool_function(**valid_args)
                else:
                    print(f"Warning: Unknown tool name '{tool_name}' encountered in plan.")
                    continue
                if result is not None and isinstance(result, list):
                    planned_results.extend(result)
                elif result is not None:
                        print(f"Warning: Tool {tool_name} returned non-list result: {result}")
            except Exception as e:
                print(f"--- Error executing planned tool {tool_name}: {e} ---")
        else:
            print(f"Warning: Planned tool '{tool_name}' not found in tool_map.")
    print(f"Planned search returned {len(planned_results)} results.")

    # --- Step 2: Execute Keyword Search (WITH CATEGORY HINT) ---
    keyword_results = []
    try:
        # Use query_string for the print statement
        print(f"Calling (Keyword): keyword_search_sqlite for query '{query_string}'")
        # Ensure tools module has necessary functions
        if hasattr(tools, 'extract_keywords_and_category') and hasattr(tools, 'keyword_search_sqlite'):
            # Use query_string for the tool
            extracted_keywords, guessed_category = tools.extract_keywords_and_category(query_string)
            if extracted_keywords:
                # --- FIX: Pass keywords, not the query ---
                # Pass the extracted keywords directly to avoid re-running extraction
                keyword_results = tools.keyword_search_sqlite(
                    keywords=extracted_keywords, 
                    category_hint=guessed_category, 
                    n_results=INITIAL_K
                )
                # --- END FIX ---
            else:
                print("No keywords extracted, skipping keyword search.")
        else:
            print("Error: Keyword search or extraction function not found in tools module.")
    except Exception as e:
        print(f"Error during keyword search execution: {e}")
        traceback.print_exc()

    # --- Step 3: Combine and Deduplicate Results ---
    combined_results = []
    seen_ids = set()
    print("Combining and deduplicating initial search results...")
    # Prioritize planned results slightly by adding them first
    for res in planned_results + keyword_results:
        doc_id = res.get('S. No.') # Use the correct column name 'S. No.'
        if doc_id is not None:
            try:
                # Store by int ID to ensure uniqueness
                doc_id_int = int(doc_id)
                if doc_id_int not in seen_ids:
                    combined_results.append(res)
                    seen_ids.add(doc_id_int)
            except (ValueError, TypeError):
                    print(f"Warning: Invalid 'S. No.' format encountered: {doc_id}. Skipping.")

    # --- Step 4: Select Top N Results (NO LLM RERANKING) ---
    FINAL_CONTEXT_K = 7 # Choose how many top results to send to the final LLM
    final_results = combined_results[:FINAL_CONTEXT_K] # Simply take the top N unique results

    print(f"Selected top {len(final_results)} unique results from hybrid search to send to LLM.")
    print("----------------------------------------------------------\n")

    # --- Step 5: Format Final Context ---
    if not final_results:
        # Use query_string for log_knowledge_gap
        context_str = log_knowledge_gap(query_string)
    else:
        try:
            # Convert list of dicts to JSON string for the prompt
            context_str = json.dumps(final_results, indent=2)
        except TypeError as e:
                print(f"Error serializing final results to JSON: {e}")
                # Fallback to simple string representation
                context_str = "\n---\n".join([str(item) for item in final_results])

    # Return dict structure expected by the synthesis chain
    # We return the original 'query_input' dict
    return {"context": context_str, "query": query_input}

# --- create_general_query_chain (Uses the execute_plan without reranking) ---
def create_general_query_chain():
    """Builds the chain for handling general queries using planning, hybrid search (no rerank), and synthesis."""
    try:
        agent_plan_schema = prompts.AgentPlan
        final_answer_prompt_template = prompts.FINAL_ANSWER_PROMPT # Use the improved synthesizer prompt
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        agent_plan_schema = None
        final_answer_prompt_template = "Error: Synthesis prompt missing. Context: {context} Query: {query}"
        # Consider exiting if critical components are missing
        # sys.exit(1)

    if agent_plan_schema is None:
         print("Exiting due to missing AgentPlan schema.")
         sys.exit(1)

    # 1. Planner sub-chain
    plan_chain = (
        ChatPromptTemplate.from_template(PLANNER_SYSTEM_PROMPT)
        | llm.with_structured_output(agent_plan_schema)
    )

    # 2. Synthesis sub-chain (uses the final answer prompt)
    synthesis_chain = (
        PromptTemplate.from_template(final_answer_prompt_template)
        | llm
        | StrOutputParser()
    )

    # 3. Complete chain structure: Plan -> Execute (Hybrid, No Rerank) -> Synthesize
    complete_general_chain = RunnablePassthrough.assign(
        # Pass only query to planner, assign output to 'plan'
        plan= RunnableLambda(lambda x: {"query": x["query"]}) | plan_chain
    # Pass {"query": ..., "plan": ...} to execute_plan, get {"query": ..., "context": ...}
    ) | execute_plan | synthesis_chain # Pass context and query to synthesizer

    return complete_general_chain


# --- create_extractor_chain (Ensure k=1 fix is applied) ---
def create_extractor_chain():
    """Builds the chain for extracting specific facts (uses semantic search k=1)."""
    try:
        extractor_output_schema = prompts.ExtractorOutput
        extractor_prompt_template = prompts.EXTRACTOR_PROMPT
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        extractor_output_schema = None
        extractor_prompt_template = "Error: Extractor prompt missing. Context: {context} Query: {query}"
        # sys.exit(1)

    if extractor_output_schema is None:
        print("Exiting due to missing ExtractorOutput schema.")
        sys.exit(1)

    # semantic_search_tool_for_extractor needs to use k=1
    def semantic_search_tool_for_extractor(input_dict: dict) -> str:
        query = input_dict.get("query", "")
        print(f"\n--- Executing Extractor Search ---")
        try:
            if hasattr(tools, 'semantic_search'):
                # --- Ensure n_results=1 ---
                results = tools.semantic_search(query=query, n_results=1)
            else:
                 print("Error: semantic_search function not found in tools module.")
                 results = []
            print(f"Found {len(results)} extraction context.")
            if not results:
                log_knowledge_gap(query)
                return "[[EXTRACTION_CONTEXT_NOT_FOUND]]"
            # Ensure context is serializable
            try:
                context_str = json.dumps(results, indent=2)
            except TypeError:
                context_str = str(results) # Fallback if complex objects in metadata
            return context_str
        except Exception as e:
             print(f"Error in extractor semantic search: {e}")
             return "[[EXTRACTION_CONTEXT_NOT_FOUND]]"

    # format_extractor_output (No significant changes needed, ensure robustness)
    def format_extractor_output(output: Any) -> str: # Use Any for robustness
        if not isinstance(output, extractor_output_schema):
            print(f"Warning: Extractor LLM did not return expected Pydantic object. Output: {output}")
            if isinstance(output, str) and "not found" in output.lower():
                return "I could not find a specific answer for that in the database."
            return "Could not extract the answer due to an internal formatting error."

        if output.answer is None or "not found" in output.answer.lower():
             return "I could not find a specific answer for that in the database."
        source_str = ""
        if output.source_code and output.source_clause:
            source_str = f" (Source: {output.source_code}, Clause: {output.source_clause})"
        elif output.source_code:
             source_str = f" (Source: {output.source_code})"
        return f"The answer is: **{output.answer}**{source_str}"

    extractor_prompt_obj = ChatPromptTemplate.from_template(extractor_prompt_template)

    # route_extractor_logic (No changes needed)
    def route_extractor_logic(input_dict: dict):
        context = input_dict.get("context", "")
        if context == "[[EXTRACTION_CONTEXT_NOT_FOUND]]":
            return RunnableLambda(lambda x: "I could not find relevant context in the database for that specific query.")
        else:
            extractor_llm_chain = extractor_prompt_obj | llm.with_structured_output(extractor_output_schema, include_raw=False) | format_extractor_output
            return extractor_llm_chain

    # Final extractor chain structure
    return (
        {"context": semantic_search_tool_for_extractor, "query": lambda x: x["query"]}
        | RunnableLambda(route_extractor_logic)
    )


# --- create_supervisor_chain (No changes needed here) ---
def create_supervisor_chain():
    """Builds the master chain that routes tasks."""
    try:
        query_router_schema = prompts.QueryRouter
        router_prompt_template = prompts.ROUTER_PROMPT # Use the refined few-shot router prompt
    except AttributeError as e:
        print(f"--- ERROR: Missing Pydantic model or prompt in prompts module: {e} ---")
        query_router_schema = None
        router_prompt_template = "Error: Router prompt missing. Query: {query}"
        # sys.exit(1)

    if query_router_schema is None:
        print("Exiting due to missing QueryRouter schema.")
        sys.exit(1)

    router_prompt_obj = ChatPromptTemplate.from_template(router_prompt_template)
    router_chain = router_prompt_obj | llm.with_structured_output(query_router_schema)

    general_chain = create_general_query_chain()
    extractor_chain = create_extractor_chain()

    branch = RunnableBranch(
        (lambda x: x.get('task_type') == 'extraction_query', extractor_chain),
        general_chain
    )

    return (
        {"task": {"query": RunnablePassthrough()} | router_chain,
         "query": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"task_type": getattr(x.get("task"), "task_type", "general_query"), "query": x.get("query")})
        | branch
    )

# --- Main execution block (No changes needed) ---
if __name__ == "__main__":
    print("Road Safety Supervisor Agent Initialized.")
    print("This file is for logic. Run 'streamlit run app.py' to use the app.")

    if tools.sql_engine is None or tools.langchain_vector_store is None:
        print("\n--- WARNING: One or more tools failed to initialize. ---")

    try:
        agent_chain = create_supervisor_chain()
    except Exception as e:
        print(f"--- FATAL ERROR during agent chain creation: {e} ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    while True:
        try:
            query = input("\nYour query (or 'exit'): ")
            if query.lower() == 'exit':
                break
            if not query.strip():
                print("Please enter a query.")
                continue

            print("\n--- Agent Answer ---")
            full_response = ""
            for chunk in agent_chain.stream(query):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
             import traceback
             print(f"\n--- An Error Occurred During Query Processing ---")
             query_at_error = 'N/A'
             try: query_at_error = query
             except NameError: pass
             print(f"Query: {query_at_error}")
             print(f"Error Type: {type(e).__name__}")
             print(f"Error Details: {e}")
             traceback.print_exc()
             print(f"-------------------------------------------------")
