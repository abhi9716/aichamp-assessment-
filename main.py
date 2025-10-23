import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


# class PythonExpressionToolResult(TypedDict):
#     result: Any
#     error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool

class LoadDataToolResult(TypedDict):
    result: Any
    error: str | None


def load_data_from_drive_tool(google_drive_link: str) -> LoadDataToolResult:
    """
    Tool that loads a CSV dataset from a Google Drive shareable link.
    The link must be in the format: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    """
    try:
        import pandas as pd
        file_id = google_drive_link.split("/d/")[1].split("/")[0]
        csv_url = f"https://drive.google.com/uc?id={file_id}"
        df = pd.read_csv(csv_url)
        return {"result": df.head().to_string(), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=1000, tools=tools, messages=messages
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # ✅ Safe handling for malformed tool inputs
                    if tool_name == "python_expression":
                        if isinstance(tool_input, dict):
                            expression = tool_input.get("expression") or json.dumps(tool_input)
                        elif isinstance(tool_input, str):
                            expression = tool_input
                        else:
                            expression = str(tool_input)

                        if verbose:
                            print("\nInput:")
                            print("```")
                            print(expression)
                            print("```")

                        result = handler(expression)

                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")

                    elif tool_name == "submit_answer":
                        if isinstance(tool_input, dict) and "answer" in tool_input:
                            result = handler(tool_input["answer"])
                            submitted_answer = result["answer"]
                        else:
                            result = {"error": "Invalid input format for submit_answer"}

                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: Any,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=5,
        verbose=verbose,
    )

    success = result == expected_answer

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")

    return run_id, success, result


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "load_data_from_drive",
            "description": "Loads a CSV file from a Google Drive link into a pandas DataFrame.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "google_drive_link": {
                        "type": "string",
                        "description": "Google Drive shareable link of the CSV file.",
                    }
                },
                "required": ["google_drive_link"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "load_data_from_drive": load_data_from_drive_tool,
        "submit_answer": submit_answer_tool,
    }

    # 🔗 Your Google Drive dataset prompt
    drive_link = "https://drive.google.com/file/d/18aRqId7sKY-R4dyRt2dd6ackibk1IUhg/view?usp=sharing"

    prompt = f"""
        You are an ML engineer analyzing a dataset.
        1. Load the CSV data from this Google Drive link: {drive_link}.
        2. Use load_data_from_drive tool to know about columns.
        2. Clean the data by dropping rows with missing values and removing duplicates.
        3. Which CSAT Score has highest average issue response time?
        4. Give output in single string and unique value.
"""

    expected_answer = "1"  # Adjust this based on your actual dataset result

    num_runs = 10
    print(f"Running {num_runs} dataset analysis test iterations...\n{'=' * 60}")

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run all tests (parallel or sequential)
    results = []
    if concurrent:
        results = [await coro for coro in asyncio.as_completed(tasks)]
    else:
        for task in tasks:
            results.append(await task)

    # ✅ Compute success/failure correctly
    successes = sum(1 for (_, success, _) in results if success)
    failures = len(results) - successes
    pass_rate = (successes / len(results)) * 100

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(concurrent=False))
