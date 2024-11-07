import argparse
import copy
import json
import os
from typing import List
import faiss
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from templates import Template, MCQATemplate, NLITemplate
from data_loaders.base import load_mcqa_samples, load_nli_samples


class TextDataset(Dataset):
    def __init__(
            self,
            samples: List,
            template: Template,
            tokenizer,
    ):
        self.samples = samples
        self.template = template
        self.tokenizer = tokenizer

    def instantiate_template(self, sample):
        return self.template.instantiate_template_full(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.instantiate_template(self.samples[index])
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        return dict(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            token_type_ids=encodings["token_type_ids"],
            sample=self.samples[index]
        )


class InputDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids = [example["input_ids"].squeeze() for example in examples]
        attention_mask = [example["attention_mask"].squeeze() for example in examples]
        token_type_ids = [example["token_type_ids"].squeeze() for example in examples]
        samples = [example["sample"] for example in examples]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "samples": samples
        }


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


class KNNDataMaker:
    def __init__(
            self,
            samples,
            demo_samples,
            model_name_or_path="facebook/contriever-msmarco",
            template=None
    ):
        self.samples = samples
        self.demo_samples = demo_samples
        self.template = template

        self.tokenizer, self.model = self.__prepare_tokenizer_and_model__(model_name_or_path)

        self.embeddings, self.samples = self.get_embeddings(samples, template)
        self.demo_embeddings, self.demo_samples = self.get_embeddings(demo_samples, template)

    def __prepare_tokenizer_and_model__(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation_side="left")

        model = AutoModel.from_pretrained(model_name_or_path)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        model = model.to("cuda")
        model.eval()

        return tokenizer, model

    def get_embeddings(self, samples, template):
        dataset = TextDataset(
            samples=samples,
            template=template,
            tokenizer=self.tokenizer
        )
        data_collator = InputDataCollator(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=2048, num_workers=8,
            collate_fn=data_collator, shuffle=False,
            pin_memory=True,
        )

        all_embeddings = []
        all_samples = []
        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader)):
                samples = batch["samples"]
                batch = {k: v.to("cuda") for k, v in batch.items() if k != "samples"}

                outputs = self.model(**batch)
                embeddings = mean_pooling(outputs[0], batch["attention_mask"])
                embeddings = embeddings.cpu().numpy()

                all_embeddings.append(embeddings)
                all_samples.extend(samples)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return all_embeddings, all_samples

    def search_knn_demo_samples(self, K=3):
        index = faiss.IndexFlatL2(self.demo_embeddings.shape[1])
        # gpu faiss
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)

        gpu_index = faiss.index_cpu_to_all_gpus(index)  # 构建 GPU 索引

        gpu_index.add(self.demo_embeddings)  # 添加到 GPU 索引

        few_shot_demo_samples = []
        few_shot_indices = []
        batch_size = 128
        for i in tqdm(range(0, self.embeddings.shape[0], batch_size), total=self.embeddings.shape[0]//batch_size, desc="Searching"):
            D, I = gpu_index.search(self.embeddings[i:i + batch_size], K)
            for j in range(I.shape[0]):
                few_shots, indices = [], []
                for k in range(I.shape[1]):
                    few_shots.append(self.demo_samples[I[j, k]])
                    indices.append(I[j, k].tolist())
                few_shot_demo_samples.append(few_shots)
                few_shot_indices.append(indices)

        print(few_shot_demo_samples[0])
        print(self.samples[0])
        print("====================================")

        for demo_sample in few_shot_demo_samples[0] + [self.samples[0]]:
            t = self.template.instantiate_template_full(demo_sample)
            print(t + "\n")

        return few_shot_demo_samples, few_shot_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--template_name", type=str, default="mcqa_with_options")
    parser.add_argument("--task_name", type=str, default="usmleqa")
    parser.add_argument("--task_category", type=str, default="mcqa")
    parser.add_argument("--icl_source", type=str, default="train")
    parser.add_argument("--topk", type=int, default=32)

    parser.add_argument("--output_cache_dir", type=str, default="../cache")
    args = parser.parse_args()

    if args.task_category == "mcqa":

        template = MCQATemplate(args.template_name)
        samples = load_mcqa_samples(args.task_name)

    elif args.task_category == "nli":

        template = NLITemplate(args.template_name)
        samples = load_nli_samples(args.task_name)

    else:
        raise ValueError("Invalid task category: [%s]" % args.task_category)

    print(len(samples["test"]))
    print(len(samples["train"]))

    knn_data_maker = KNNDataMaker(
        samples=samples["test"],
        demo_samples=samples["train"],
        model_name_or_path=args.model_name_or_path,
        template=template
    )

    few_shot_demo_samples, few_shot_indices = knn_data_maker.search_knn_demo_samples(K=args.topk)

    if not os.path.exists(os.path.join(args.output_cache_dir, args.task_name, args.template_name)):
        os.makedirs(os.path.join(args.output_cache_dir, args.task_name, args.template_name))

    retriever_id = args.model_name_or_path.replace("/", "-")
    dump_filename = os.path.join(args.output_cache_dir, args.task_name, args.template_name, f"{retriever_id}_knn_{args.topk}.csv")
    with open(dump_filename, "w", encoding="utf-8") as writer:
        for i, indices in enumerate(few_shot_indices):
            writer.write(f"{i}," + ",".join([str(index) for index in indices]) + "\n")

    print(f"Dumped to -> [{dump_filename}]")

    del knn_data_maker
