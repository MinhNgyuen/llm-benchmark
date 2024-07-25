import argparse
from typing import List

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided, potentially due to prompt caching. For more info, see https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(
    model_name: str, prompt: str, verbose: bool
) -> OllamaResponse:

    last_element = None

    if verbose:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    # with open("data/ollama/ollama_res.json", "w") as outfile:
    #     outfile.write(json.dumps(last_element, indent=4))

    return OllamaResponse.model_validate(last_element)


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
        model_response.prompt_eval_count + model_response.eval_count
    ) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models(models_to_run: List[str] = []) -> List[str]:
    all_models = ollama.list().get("models", [])
    all_model_names = [model["name"] for model in all_models]
    
    if not models_to_run:
        model_names = all_model_names
    else:
        model_names = [model for model in models_to_run if model in all_model_names]
        if len(model_names) != len(models_to_run):
            missing_models = set(models_to_run) - set(model_names)
            print(f"Warning: The following models were not found: {', '.join(missing_models)}")
    
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="List of model names to benchmark. If not specified, all available models will be used.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            "Why is the sky blue?",
            "Write a report on the financials of Apple Inc.",
        ],
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    models = args.models
    prompts = args.prompts
    print(
        f"\nVerbose: {verbose}\nModels: {models}\nPrompts: {prompts}"
    )

    model_names = get_benchmark_models(args.models)
    benchmarks = {}

    for model_name in model_names:
        responses: List[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            responses.append(response)

            if verbose:
                print(f"Response: {response.message.content}")
                inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M llama2:latest --prompts "What color is the sky" "Write a report on the financials of Microsoft"
