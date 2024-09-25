import random
import datasets
import os
import ujson as json
from typing import List, Dict
from datasets import load_dataset

from tasks.mcqa import MCQASample
from tasks.ner import NERSample
from tasks.mt import MTSample
from tasks.nli import NLISample
from tool_utils import main_print
from config_file import DATA_ROOT_DIR


BENCHMARK_NAME = "Coldog2333/JMedBench"


def load_mcqa_samples(dataset_name: str) -> Dict[str, List[MCQASample]]:

    mcqa_samples = {"train": [], "test": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)

        for split in ["train", "test"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    mcqa_samples[split].append(
                        MCQASample(
                            sample_id=sample["sample_id"],
                            question=sample["question"],
                            options=sample["options"],
                            answer_idx=sample["answer_idx"],
                            n_options=sample["n_options"],
                            metadata=sample["metadata"]
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):

                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        mcqa_samples[split].append(
                            MCQASample(
                                sample_id=sample["sample_id"],
                                question=sample["question"],
                                options=sample["options"],
                                answer_idx=sample["answer_idx"],
                                n_options=sample["n_options"],
                                metadata=sample["metadata"]
                            )
                        )

            else:
                main_print(f"File {data_filename} does not exist.")

    main_print(f"Loaded {len(mcqa_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(mcqa_samples['train'])} samples for training set.")

    return mcqa_samples


def load_nli_samples(dataset_name: str) -> Dict[str, List[NLISample]]:

    nli_samples = {"train": [], "test": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "test"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    nli_samples[split].append(
                        NLISample(
                            sample_id=sample["sample_id"],
                            premise=sample["premise"],
                            hypothesis=sample["hypothesis"],
                            label=sample["label"],
                            n_label=sample["n_label"],
                            metadata=sample["metadata"]
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        nli_samples[split].append(
                            NLISample(
                                sample_id=sample["sample_id"],
                                premise=sample["premise"],
                                hypothesis=sample["hypothesis"],
                                label=sample["label"],
                                n_label=sample["n_label"],
                                metadata=sample["metadata"]
                            )
                        )

    main_print(f"Loaded {len(nli_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(nli_samples['train'])} samples for training set.")

    return nli_samples


def load_ner(dataset_name: str) -> Dict[str, List[NERSample]]:
    ner_samples = {"train": [], "test": [], "validation": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "validation", "test"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    ner_samples[split].append(
                        NERSample(
                            text=sample["text"],
                            labels=sample["labels"],
                            entity_type=sample["entity_type"]
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        ner_samples[split].append(
                            NERSample(
                                text=sample["text"],
                                labels=sample["labels"],
                                entity_type=sample["entity_type"]
                            )
                        )

    main_print(f"Loaded {len(ner_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(ner_samples['train'])} samples for training set.")
    main_print(f"Loaded {len(ner_samples['validation'])} samples for validation set.")

    return ner_samples


def load_blurb(subset_name):
    """
    :param subset_name: Options: "bc2gm", "bc5chem", "bc5disease", "jnlpba", "ncbi_disease"
    :return:
    """
    dataset = load_dataset('bigbio/blurb', subset_name)
    ner_samples = {"train": [], "test": [], "validation": []}
    for split in dataset.keys():
        for sample in dataset[split]:
            text = " ".join(sample["tokens"])

            # collect labels
            labels = []
            cache = []
            assert len(sample["ner_tags"]) == len(sample["tokens"])
            for tag, token in zip(sample["ner_tags"], sample["tokens"]):
                if tag == 0:
                    if cache:
                        labels.append(" ".join(cache))
                        cache = []
                elif tag == 1:
                    if cache:
                        labels.append(" ".join(cache))
                        cache = []
                    cache.append(token)
                else:
                    assert len(cache) > 0
                    cache.append(token)

            if cache:
                labels.append(" ".join(cache))
                cache = []

            if len(labels) == 0:
                labels = ["none"]

            ner_samples[split].append(
                NERSample(
                    text=text,
                    labels=labels,
                    entity_type=sample["type"]
                )
            )

    main_print(f"Loaded {len(ner_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(ner_samples['train'])} samples for training set.")
    main_print(f"Loaded {len(ner_samples['validation'])} samples for validation set.")

    return ner_samples




def load_ejmmt(
    dataset_name="ejmmt",
    source_lang="english",
    target_lang="japanese",
) -> Dict[str, List[MTSample]]:

    mt_samples = {"train": [], "test": [], "validation": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "test", "validation"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    mt_samples[split].append(
                        MTSample(
                            source_text=sample[source_lang],
                            target_text=sample[target_lang],
                            source_language=source_lang,
                            target_language=target_lang
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test", "validation"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        mt_samples[split].append(
                            MTSample(
                                source_text=sample[source_lang],
                                target_text=sample[target_lang],
                                source_language=source_lang,
                                target_language=target_lang
                            )
                        )

            else:
                main_print(f"File {data_filename} does not exist.")

    main_print(f"Loaded {len(mt_samples['test'])} samples for testing set.")

    return mt_samples


VALID_DATASET_NAMES = ["medmcqa", "usmleqa", "medqa", "mmlu", "mmlu_medical"]

if __name__ == "__main__":
    for dataset_name in VALID_DATASET_NAMES:
        mcqa_samples = load_mcqa_samples(dataset_name)
        # print(mcqa_samples)
