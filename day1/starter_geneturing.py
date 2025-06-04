import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union, cast

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

os.environ["no_proxy"] = "*"
load_dotenv()
mlflow.set_tracking_uri("http://198.215.61.34:8153/")
mlflow.set_experiment("s440708")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

print(
    "Loaded API key:", AZURE_OPENAI_KEY[:5] + "..." if AZURE_OPENAI_KEY else "MISSING"
)


# 2.1 Data Configuration
data_config = {
    "dataset_path": "../data/geneturing.json",
    "output_path": "outputs/eval_outputs.json",
}


# 2.2 Model Configuration

model_config = {
    "model_name": AZURE_OPENAI_DEPLOYMENT_NAME,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "temperature": 1.0,
    "max_tokens": 800,
    "top_p": 1.0,
}


# 2.3 Evaluation and Logging Configuration


# 3.1 Load the JSON file
# Load the data here


def load_geneturing(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        raw_data = json.load(f)

    flat_data = []
    counter = 0
    for task_name, qa_pairs in raw_data.items():
        for question, answer in qa_pairs.items():
            flat_data.append(
                {
                    "id": counter,
                    "task": task_name,
                    "question": question,
                    "answer": answer,
                }
            )

            counter += 1
    return flat_data


dataset = load_geneturing(data_config["dataset_path"])
print(f"Loaded {len(dataset)} examples from {data_config['dataset_path']}")
# TASKS = set()

# # 3.2 Iterate through the JSON data recursively
# #      to collect each of the rows into a list
# #     Each row should have a dictionary with keys of the columsn in the table above

# def flatten_geneturing_json(raw_data: dict) -> List[Dict[str, str]]:

#     flat_data = []
#     counter = 0

#     for task_name, qa_pairs in raw_data.items():
#         for question, answer in qa_pairs.items():
#             row = {
#                 "id": counter,
#                 "task": task_name,
#                 "question": question,
#                 "answer": answer
#             }
#             flat_data.append(row)
#             counter += 1


#     return flat_data


# dataset = flatten_geneturing_json(dataset)
# print(f"Flattened to {len(dataset)} rows.")


# 3.3 Create the pandas dataframe from the collection of rows


def build_dataframe(flat_data: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(flat_data)

    df = df[["id", "task", "question", "answer"]]
    df.set_index("id", inplace=True)
    return df


df = build_dataframe(dataset)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# from ollama import Client
# client = Client(
#   host='http://localhost:11434',
# )
# response = client.chat(model='qwen3:4b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])

# 4.1 Setting up the large language model Ollama model client

OllamaClient = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

response = OllamaClient.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        },
    ],
    max_completion_tokens=model_config["max_tokens"],
    temperature=model_config["temperature"],
    top_p=model_config["top_p"],
    frequency_penalty=model_config["frequency_penalty"],
    presence_penalty=model_config["presence_penalty"],
    model=model_config["model_name"],
)

print(response.choices[0].message.content)


# 4.2 Draft your own system prompt for our generic genomics question answering system.
#     Replace the system message `content` below with your own.
system_message = [
    {
        "role": "system",
        "content": "You are a genomic assistant."
        "When answering questions, return only the final answer."
        "Do not include explanations, references, or formatting."
        "For example, only reply 'chr1:123-456'"
        "or 'GNAS' without additional context.",
    }
]

# 4.3 Appending the few-shot examples to the `messages` list

example_messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hello"},
]


def query_model(
    client: Any,
    system_message: dict[str, str],
    few_shot_examples: List[dict[str, str]],
    user_query: str,
) -> str:
    messages = (
        [system_message] + few_shot_examples + [{"role": "user", "content": user_query}]
    )

    response = client.chat.completions.create(
        model=model_config["model_name"],
        messages=messages,
        max_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
        top_p=model_config["top_p"],
        frequency_penalty=model_config["frequency_penalty"],
        presence_penalty=model_config["presence_penalty"],
    )

    return str(response.choices[0].message.content)


# 5.1 Implement metrics


def exact_match(pred: str, true: str) -> float:
    # if isinstance(pred, list):
    #     pred = [p.strip().lower() for p in pred]
    #     return float(true.strip().lower() in pred)
    # else:
    #     return float(pred.strip().lower() == true.strip().lower())
    return float(pred.strip().lower() == true.strip().lower())


def gene_disease_association(pred: list[str], true: list[str]) -> float:
    pred_set = set(map(str.lower, map(str.strip, pred)))
    true_set = set(map(str.lower, map(str.strip, true)))
    if not true_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & true_set) / len(true_set)


def disease_gene_location(pred: list[str], true: list[str]) -> float:
    pred_set = set(map(str.lower, map(str.strip, pred)))
    true_set = set(map(str.lower, map(str.strip, true)))
    if not true_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & true_set) / len(true_set)


def human_genome_dna_alignment(pred: str, true: str) -> float:
    pred = pred.strip().lower()
    true = true.strip().lower()
    if pred == true:
        return 1.0
    pred_chr = pred.split(":")[0]
    true_chr = true.split(":")[0]
    return 0.5 if pred_chr == true_chr else 0.0


MetricFunc = Callable[[Union[str, list[str]], Union[str, list[str]]], float]

metric_task_map: Dict[str, MetricFunc] = defaultdict(
    lambda: cast(MetricFunc, exact_match),
    {
        "Gene disease association": cast(MetricFunc, gene_disease_association),
        "Gene location": cast(MetricFunc, disease_gene_location),
        "Human genome DNA aligment": cast(MetricFunc, human_genome_dna_alignment),
    },
)


def get_answer(answer: str | list[str], task: str) -> str:
    mapper = {
        "Caenorhabditis elegans": "worm",
        "Homo sapiens": "human",
        "Danio rerio": "zebrafish",
        "Mus musculus": "mouse",
        "Saccharomyces cerevisiae": "yeast",
        "Rattus norvegicus": "rat",
        "Gallus gallus": "chicken",
    }
    if isinstance(answer, list):
        answer = " ".join(answer)

    if task == "SNP location":
        answer = answer.strip().split()[-1]
        if "chr" not in answer:
            answer = "chr" + answer

    elif task == "Gene location":
        answer = answer.strip().split()[-1]
        if "chr" not in answer:
            answer = "chr" + answer

    elif task == "Gene disease association":
        answer = answer.strip().replace("Answer: ", "")
        answer = answer.split(", ")

    elif task == "Disease gene location":
        answer = answer.strip().replace("Answer: ", "")
        answer = answer.split(", ")

    elif task == "Protein-coding genes":
        answer = answer.strip().replace("Answer: ", "")
        if answer == "Yes":
            answer = "TRUE"
        elif answer == "No":
            answer = "NA"

    elif task == "Multi-species DNA aligment":
        answer = answer.strip().replace("Answer: ", "")
        answer = mapper.get(answer, answer)

    else:
        answer = answer.strip().replace("Answer: ", "")

    if isinstance(answer, list):
        answer = answer[0]
    return answer


# 6.1 Set up data structures for results


@dataclass
class Result:
    id: int
    task: str
    question: str
    answer: str
    raw_prediction: Optional[str]
    processed_prediction: Optional[str]
    score: Optional[float]
    success: bool


def save_results(results: List[Result], results_csv_filename: str) -> None:
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(results_csv_filename, index=False)


# 6.2 Loop over the dataset with a progress bar

# * Do not forget to add the results to our Result list,
#   both successful and failed predictions
# * API calls will not always work,
#   so make sure we capture the exceptions from failed calls
#   and add them to the Result list with a `status=False`


# def evaluate_with_progress(
#     client,
#     dataset: List[Dict],
#     system_message: Dict,
#     few_shot_examples: List[Dict],
# ) -> List[Result]:
#     results: List[Result] = []

#     for item in tqdm(dataset, desc="Evaluating"):
#         item_id = item.get("id")
#         task = item.get("task")
#         question = item.get("question")
#         gold_answer = item.get("answer")


#         raw_pred = query_model(client, system_message, few_shot_examples, question)

#         # Post-process
#         print("task:", task)
#         # print("raw pred:", raw_pred)
#         pred = get_answer(raw_pred, task)
#         print("pred:", pred)
#         true = get_answer(gold_answer, task)
#         print("true:",true)


#         metric_fn = metric_task_map[task]
#         score = metric_fn(pred, true)
#         try:
#             result = Result(
#                 id=item_id,
#                 task=task,
#                 question=question,
#                 answer=gold_answer,
#                 raw_prediction=raw_pred,
#                 processed_prediction=str(pred),
#                 score=score,
#                 success=True
#             )

#         except Exception as e:

#             result = Result(
#                 id=item_id,
#                 task=task,
#                 question=question,
#                 answer=gold_answer,
#                 raw_prediction=None,
#                 processed_prediction=None,
#                 score=0.0,
#                 success=False
#             )

#         results.append(result)


#     return results
def evaluate_with_progress(
    client: Any,
    dataset: List[Dict[str, str]],
    system_message: Dict[str, str],
    few_shot_examples: List[Dict[str, str]],
) -> List[Result]:
    results: List[Result] = []

    with mlflow.start_run(run_name="geneturing-eval"):
        mlflow.log_param("model", model_config["model_name"])
        mlflow.log_param("temperature", model_config["temperature"])
        mlflow.log_param("max_tokens", model_config["max_tokens"])
        mlflow.log_param("top_p", model_config["top_p"])

        for item in tqdm(dataset, desc="Evaluating"):
            item_id = item.get("id")
            task = item.get("task")
            question = item.get("question")
            gold_answer = item.get("answer")

            try:
                raw_pred = query_model(
                    client, system_message, few_shot_examples, question
                )
                pred = get_answer(raw_pred, task)
                true = get_answer(gold_answer, task)
                metric_fn = metric_task_map[task]
                score = metric_fn(pred, true)

                result = Result(
                    id=item_id,
                    task=task,
                    question=question,
                    answer=gold_answer,
                    raw_prediction=raw_pred,
                    processed_prediction=str(pred),
                    score=score,
                    success=True,
                )

            except Exception:
                result = Result(
                    id=item_id,
                    task=task,
                    question=question,
                    answer=gold_answer,
                    raw_prediction=None,
                    processed_prediction=None,
                    score=0.0,
                    success=False,
                )

            results.append(result)

        # Calculate and log metrics
        df = pd.DataFrame([asdict(r) for r in results])
        success_rate = df["success"].mean()
        valid_scores_df = df[df["score"].notna()]
        overall_score = valid_scores_df["score"].mean()

        mlflow.log_metric("success_rate", success_rate)
        mlflow.log_metric("overall_score", overall_score)

        for task_name, task_df in valid_scores_df.groupby("task"):
            mlflow.log_metric(
                f"score_{task_name.replace(' ', '_')}", task_df["score"].mean()
            )

        # Save to CSV and log as artifact
        result_csv_path = "geneturing_results.csv"
        df.to_csv(result_csv_path, index=False)
        mlflow.log_artifact(result_csv_path)

    return results


# 6.3 Save the results

few_shot_examples = [
    {"role": "user", "content": "What is the official gene symbol of LMP10?"},
    {"role": "assistant", "content": "SLC38A6"},
    {
        "role": "user",
        "content": "What are genes related to Pseudohypoparathyroidism Ic?",
    },
    {"role": "assistant", "content": "GNAS"},
    {
        "role": "user",
        "content": "Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAA",
    },
    {"role": "assistant", "content": "chr15:91950805-91950932"},
    {
        "role": "user",
        "content": "Which organism does the DNA sequence come from:AGGGGCAGCAAACACC",
    },
    {"role": "assistant", "content": "worm"},
    {"role": "user", "content": "Convert ENSG00000215251 to official gene symbol."},
    {"role": "assistant", "content": "FASTKD5"},
    {"role": "user", "content": "Is ATP5F1EP2 a protein-coding gene?"},
    {"role": "assistant", "content": "NA"},
    {"role": "user", "content": "Which gene is SNP rs1217074595 associated with?"},
    {"role": "assistant", "content": "LINC01270"},
    {
        "role": "user",
        "content": "Which chromosome does SNP rs1430464868 locate on human genome?",
    },
    {"role": "assistant", "content": "chr13"},
]
sample = random.sample(dataset, 50)

results = evaluate_with_progress(
    OllamaClient, dataset, system_message[0], few_shot_examples
)
save_results(results, "geneturing_results.csv")


df = pd.read_csv("geneturing_results.csv")
success_rate = df["success"].mean()
print(f"Fraction of successful predictions: {success_rate:.2%}")


# 7.2 Calculate the overall score and the score by task

valid_scores_df = df[df["score"].notna()]
overall_score = valid_scores_df["score"].mean()
score_by_task = valid_scores_df.groupby("task")["score"].mean()
print(f"Overall average score: {overall_score:.3f}\n")
print("Score by task:")
print(score_by_task)


# 7.3 Create a bar chart of the scores by task with a horizontal line for
# the overall score

plt.figure(figsize=(10, 6))
score_by_task.plot(kind="bar", color="cornflowerblue")
plt.axhline(
    y=overall_score, color="red", linestyle="--", label=f"Overall: {overall_score:.2f}"
)
plt.title("Average Score by Task")
plt.xlabel("Task")
plt.ylabel("Average Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
