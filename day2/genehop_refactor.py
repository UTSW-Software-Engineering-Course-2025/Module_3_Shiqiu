#!/usr/bin/env python
# coding: utf-8

# # Import

# In[35]:


import json
import os
import random
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["no_proxy"] = "*"

mlflow.set_tracking_uri("http://198.215.61.34:8153/")
mlflow.set_experiment("s440708_Shiqiu")
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

print(
    "Loaded API key:", AZURE_OPENAI_KEY[:5] + "..." if AZURE_OPENAI_KEY else "MISSING"
)


# # Config

# In[ ]:


data_config = {
    "dataset_path": "../data/geneturing.json",
    "output_path": "/work/bioinformatics/s440708/MODULE_3_MATERIALS/outputs/",
    "dataset_name": "geneturing",
}

model_config = {
    # "model_name": AZURE_OPENAI_DEPLOYMENT_NAME,
    "api": "OpenAI",
    "model_name": AZURE_OPENAI_DEPLOYMENT_NAME,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "temperature": 1.0,
    "max_tokens": 800,
    "top_p": 1.0,
}


# # Load data

# In[3]:


def load_geneturing(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        raw_data = json.load(f)

    flat_data = []
    counter = 0
    for task_name, qa_pairs in raw_data.items():
        for question, answer in qa_pairs.items():
            flat_data.append(
                {
                    "task": task_name,
                    "id": counter,
                    "question": question,
                    "answer": answer,
                }
            )
            counter += 1

    return flat_data


# config = Config()
dataset = load_geneturing(data_config["dataset_path"])
print(f"Loaded {len(dataset)} examples from {data_config['dataset_path']}")


# In[4]:


# 3.3 Create the pandas dataframe from the collection of rows


def build_dataframe(flat_data: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(flat_data)

    df = df[["id", "task", "question", "answer"]]
    df.set_index("id", inplace=True)
    return df


df = build_dataframe(dataset)
print(f"DataFrame shape: {df.shape}")
print(df.head())


# # Create a Structured Output

# In[ ]:


system_message = (
    "You are a genomics assistant that returns structured answers. "
    "You will be provided with a gene-related question and should return your answer "
    "in structured JSON format following a schema including task, "
    "question, answer, and explanation."
)


# In[ ]:


class GeneHopAnswer(BaseModel):
    """
    Structured answer format for gene-level LLM QA.
    """

    task: str = Field(..., description="GeneHop task name, e.g., 'Gene alias'")
    question: str = Field(..., description="The user query")
    answer: str = Field(
        ..., description="Structured, atomic answer returned by the model"
    )
    explanation: Optional[str] = Field(
        None, description="Optional reasoning or justification provided by the model"
    )


query = {
    "task": "SNP gene function",
    "question": "What is the function of the gene associated with SNP rs1217074595? "
    "Let's decompose the question to sub-questions and solve them step by step.",
}

prompt = f"""Here is a question: "{query['question']}".
        Return the answer in valid JSON format matching this schema:
        {{"task": str, "question": str, "answer": str, "explanation": Optional[str]}}.
T       he 'task' should be '{query['task']}'.
        """


messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt},
]


# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     api_version="2024-08-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     timeout=10,
#     max_retries=3,
# )

# response = client.beta.chat.completions.parse(
#     model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#     messages=messages,
#     temperature=0,
#     max_tokens=1024,
#     response_format=GeneHopAnswer,
# )

# from ollama import chat
# response = chat(
#   messages=messages,
#   model= model_config['model_name'],
#   format=GeneHopAnswer.model_json_schema(),
# )

# genehop_answer = GeneHopAnswer.model_validate_json(response.message.content)
# answer = genehop_answer.model_dump()
# pp.pprint(answer)

# pp.pprint(response.model_dump())


# # Create Query Model Function

# In[ ]:


def query_model(
    system_message: str, user_query: dict[str, str], api: str
) -> dict[str, str | bool]:
    from ollama import chat

    prompt = f"""Here is a question: "{user_query['question']}".
    Return the answer in valid JSON format matching this schema:
    {{"task": str, "question": str, "answer": str, "explanation": Optional[str]}}.
    The 'task' should be '{user_query['task']}'.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    if api == "ollama":
        response = chat(
            messages=messages,
            model=model_config["model_name"],
            format=GeneHopAnswer.model_json_schema(),
        )
        genehop_answer = GeneHopAnswer.model_validate_json(response.message.content)
        answer = genehop_answer.model_dump()

    if api == "OpenAI":
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=10,
            max_retries=3,
        )

        response = client.beta.chat.completions.parse(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=0,
            max_tokens=1024,
            response_format=GeneHopAnswer,
        )

        answer = response.model_dump()["choices"][0]["message"]["content"]
        if isinstance(answer, str):
            answer = json.loads(answer)
        # answer = json.loads(answer)

    return answer


# In[ ]:


def collect_structured_output(
    # client: Any,
    dataset: List[Dict[str, str]],
    system_message: str,
) -> List[Dict[str, str | bool]]:
    answers = []

    with mlflow.start_run(run_name="geneturing-eval"):
        mlflow.log_param("model", model_config["model_name"])
        mlflow.log_param("temperature", model_config["temperature"])
        mlflow.log_param("max_tokens", model_config["max_tokens"])
        mlflow.log_param("top_p", model_config["top_p"])

        for item in tqdm(dataset, desc="Evaluating"):
            item_id = item.get("id")
            task = item.get("task")
            question = item.get("question")

            # gold_answer = item.get("answer")
            query = {"task": task, "question": question}
            print(query)
            try:
                answer = query_model(system_message, query, str(model_config["api"]))
                answer["success"] = True
                answer["id"] = item_id

            except Exception:
                answer = {
                    "success": False,
                    "id": item_id,
                    "task": task,
                    "question": question,
                    "answer": "",
                    "explanation": "",
                }

            answers.append(answer)

        # Calculate and log metrics
        df = pd.DataFrame(answers)
        success_rate = df["success"].mean()
        print("success rate:", success_rate)
        # valid_scores_df = df[df["score"].notna()]
        # overall_score = valid_scores_df["score"].mean()

        mlflow.log_metric("success_rate", success_rate)
        # mlflow.log_metric("overall_score", overall_score)

        # for task_name, task_df in df.groupby("task"):
        #     mlflow.log_metric(
        #         f"score_{task_name.replace(' ', '_')}", task_df["score"].mean()
        #     )

        # Save to json and log as artifact
        result_json_path = os.path.join(
            data_config["output_path"],
            f"{data_config['dataset_name']}_{model_config['api']}_rawresults.csv",
        )
        with open(result_json_path, "w") as f:
            json.dump(answers, f, indent=2)

        # df.to_csv(result_csv_path, index=False)
        mlflow.log_artifact(result_json_path)

    return answers


# In[ ]:

sample = random.sample(dataset, 50)

results = collect_structured_output(dataset, system_message[0])


# # calculate cosine similarity

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")
model = AutoModel.from_pretrained("FremyCompany/BioLORD-2023")


with open(
    os.path.join(
        data_config["output_path"],
        f"{data_config['dataset_name']}_{model_config['api']}_rawresults.csv",
    ),
    "r",
) as f:
    pred_data = json.load(f)

with open(data_config["dataset_path"], "r") as f:
    gold_data = json.load(f)


# In[ ]:


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")
model = AutoModel.from_pretrained("FremyCompany/BioLORD-2023")


# Helper function for mean pooling
def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


results = []
gold_entries = []
for task_name, task_dict in gold_data.items():
    for question, answers in task_dict.items():
        gold_entries.append(
            {
                "task": task_name,
                "question": question.replace(
                    "Let's decompose the question to sub-questions \
                          and solve them step by step.",
                    "",
                ).strip(),
                "gold_answers": answers,
            }
        )

# Compute similarity
for i, pred_entry in enumerate(pred_data):
    question = pred_entry["question"].strip()
    predicted_answer = pred_entry["answer"].strip()
    task_name = pred_entry["task"]

    match = gold_entries[i]
    gold_answer = match["gold_answers"]
    if isinstance(gold_answer, list):
        gold_answer = " ".join(gold_answer)

    inputs = tokenizer(
        [predicted_answer, gold_answer],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        cosine_sim = F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        print(cosine_sim)
        results.append(
            {
                "task": task_name,
                "question": question,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "cosine_similarity": cosine_sim,
            }
        )


df = pd.DataFrame(results)


# In[30]:


print("cosine similarity", df["cosine_similarity"].mean())


# In[36]:


df.to_csv(
    os.path.join(
        data_config["output_path"],
        f"{data_config['dataset_name']}_{model_config['api']}_results.csv",
    )
)


# #
