TASK_SPECIFIC_TEMPLATE_SET = {
    "MCQA": {
        "Minimal": "{question}\n",
        "Standard": "質問：{question}\n{options}\n答え：",
        "English-centric": "Question: {question}\n{options}\nAnswer: ",
        "Instructed": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、以下の選択式問題に答えなさい。"
                      "以下の選択肢から正しいものを1つ選びなさい。医療ガイドラインに記載されている、現在行われている標準的な治療法に基づいて答えなさい。\n"
                      "質問：{question}\n選択肢：\n{options}\n答え："
    },
    "CMCQA": {
        "Minimal": "{context}\n{question}\n",
        "Standard": "要旨：{context}\n質問：{question}\n答え：",
        "English-centric": "Abstract: {context}\nQuestion: {question}\nAnswer: ",
        "Instructed": "臨床科学と医学知識の専門家である医師として、次の文が正しいかどうか教えてください。「はい/いいえ/たぶん」のいずれかでお答えください。\n"
                      "要旨：{context}\n質問：{question}\n答え："
    },
    "NER": {
        "Minimal": "段落：{text} => {entity_type}：",
        "Standard": "以下の段落において、{entity_type}は？\n段落：{text} => {entity_type}：",
        "English-centric": "Please extract all {entity_type}s mentioned in the paragraph.\nParagraph: {text} => {entity_type}: ",
        "Instructed": "あなたは医療分野の専門家です。\n"
                      "あなたは{entity_type}のフレーズを含む段落を与えられます。\n"
                      "あなたのタスクは段落からこれらすべてのフレーズを抽出することです。\n"
                      "抽出されたフレーズのみを返し、それらを英語のカンマ（,）で区切る必要があります。\n"
                      "段落：{text} => {entity_type}："
    },
    "MT(e2j)": {
        "Minimal": "{source_text} => ",
        "Standard": "翻訳（English => 日本語）：{source_text} => ",
        "English-centric": "Translation (English => Japanese): {source_text} => ",
        "Instructed": "あなたは生物医学文書を翻訳する医学博士です。"
                      "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                      "以下の英文を和訳しなさい。\n"
                      "{source_text} => "
    },
    "MT(j2e)": {
        "Minimal": "{source_text} => ",
        "Standard": "Translation (日本語 => English): {source_text} => ",
        "English-centric": "Translation (Japanese => English): {source_text} => ",
        "Instructed": "あなたは生物医学文書を翻訳する医学博士です。"
                      "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                      "以下の和文を英訳しなさい。\n"
                      "{source_text} => ",
    },
    "DC": {
        "Minimal": "{context}\n{question}\n",
        "Standard": "文脈：{context}\n質問：{question}\n{options}\n答え：",
        "English-centric": "Context: {context}\nQuestion: {question}\n{options}\nAnswer: ",
        "Instructed": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、以下の選択式問題に答えなさい。"
                        "以下の選択肢から正しいものを1つ選びなさい。\n文脈：{context}\n質問：{question}\n選択肢：\n{options}\n答え："
    },
    "STS": {
        "Minimal": "{premise}\n{hypothesis}\n",
        "Standard": "テキスト1：{premise}\nテキスト2：{hypothesis}\n類似度 (0-5)：",
        "English-centric": "Text 1: {premise}\nText 2: {hypothesis}\nSemantic Text Similarity (0-5): ",
        "Instructed": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、"
                      "次の2つの文の意味的類似度を0から5の範囲で判断してください。\n"
                      "0：二つの文は完全に似ていない。\n"
                      "5：二つの文は完全に同等で、意味が同じである。\n"
                      "テキスト1: {premise}\nテキスト2: {hypothesis}\n類似度 (0-5)："
    }
}

DATASET_NAME_TO_TASK = {
    "igakuqa": "MCQA",
    "jmmlu_medical": "MCQA",
    "medmcqa_jp": "MCQA",
    "usmleqa_jp": "MCQA",
    "medqa_jp": "MCQA",
    "mmlu_medical_jp": "MCQA",
    "pubmedqa_jp": "CMCQA",

    "ejmmt": "MT(e2j)",
    "ejmmt_e2j": "MT(e2j)",
    "ejmmt_j2e": "MT(j2e)",

    "mrner_medicine": "NER",
    "mrner_disease": "NER",
    "nrner": "NER",
    "bc2gm_jp": "NER",
    "bc5chem_jp": "NER",
    "bc5disease_jp": "NER",
    "jnlpba_jp": "NER",
    "ncbi_disease_jp": "NER",

    "crade": "DC",
    "rrtnm": "DC",
    "smdis": "DC",

    "jcsts": "STS"
}


class UnifiedTemplate:
    def __init__(self):
        pass

    def instantiate_template(self, sample, template_name="Standard"):
        if not isinstance(sample, dict):
            sample = sample.to_dict()

        try:
            task_name = DATASET_NAME_TO_TASK[sample["metadata"]["source"]]
            template = TASK_SPECIFIC_TEMPLATE_SET[task_name][template_name]
        except:
            raise ValueError(f"Dataset name is not given.")

        if task_name == "MCQA":

            options = []
            for i in range(sample["n_options"]):
                option_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
                option = sample["options"][i]
                # options.append(f"({option_idx}) {option}")
                options.append(f"{option_idx}. {option}")
            option_text = "\n".join(options)

            return template.format(
                question=sample["question"],
                options=option_text
            )

        elif task_name == "CMCQA":

            return template.format(
                context=sample["metadata"]["context"],
                question=sample["question"]
            )

        elif task_name in ["MT(e2j)", "MT(j2e)"]:

            return template.format(
                source_text=sample["source_text"]
            )

        elif task_name == "NER":

            return template.format(
                entity_type=sample["entity_type"],
                text=sample["text"]
            )

        elif task_name == "DC":

            options = []
            for i in range(sample["n_options"]):
                option_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
                option = sample["options"][i]
                # options.append(f"({option_idx}) {option}")
                options.append(f"{option_idx}. {option}")
            option_text = "\n".join(options)

            return template.format(
                context=sample["metadata"]["context"],
                question=sample["question"],
                options=option_text
            )

        elif task_name == "STS":

            return template.format(
                premise=sample["premise"],
                hypothesis=sample["hypothesis"]
            )

        else:

            raise NotImplementedError

    def instantiate_template_full(self, sample, template_name="Standard"):
        if not isinstance(sample, dict):
            sample = sample.to_dict()

        try:
            task_name = DATASET_NAME_TO_TASK[sample["metadata"]["source"]]
        except:
            raise ValueError(f"Dataset name is not given.")

        if task_name in ["MCQA", "CMCQA", "DC"]:

            return self.instantiate_template(sample, template_name) + f"{sample['options'][sample['answer_idx']]}"

        elif task_name in ["MT(e2j)", "MT(j2e)"]:

            return self.instantiate_template(sample, template_name) + f"{sample['target_text']}"

        elif task_name == "NER":

            if sample["labels"] == []:
                return self.instantiate_template(sample, template_name) + f"なし"
            else:
                return self.instantiate_template(sample, template_name) + f"{','.join(sample['labels'])}"

        elif task_name == "STS":

            return self.instantiate_template(sample, template_name) + f"{sample['label']}"

        else:

            raise NotImplementedError
