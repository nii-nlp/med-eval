from templates.base import Template


TEMPLATE_SET = {

## MCQA
### Mini

    "mcqa": "Question: {question}\nAnswer:",
    "mcqa_minimal": "{question}\n",

### With options

    "mcqa_with_options": "Question: {question}\n{options}\nAnswer:",

    "mcqa_with_options_jp": "質問：{question}\n{options}\n答え：",

## Context-based MCQA
    "context_based_mcqa": "Abstract: {context}\nQuestion: {question}\nAnswer:",   # The answer to the question given the abstract is",
    "context_based_mcqa_minimal": "{context}\n{question}\n",
    "context_based_mcqa_jp": "要旨：{context}\n質問：{question}\n答え：",

    "context_based_mcqa_with_options": "Abstract: {context}\nQuestion: {question}\n{options}\nAnswer:",

## Japanese

    "mcqa_jp": "質問: {question}\n回答:",


## Instructed Version

    "4o_mcqa_instructed": "You are a medical doctor answering real-world medical entrance exam questions. "
                          "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, "
                          "disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one "
                          "correct answer from the following options. Base your answer on the current and standard practices referenced in medical guidelines.\n"
                          "Question: {question}\nOptions: {options}\nAnswer:",

    "4o_mcqa_instructed_jp": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、以下の選択式問題に答えなさい。"
                             "以下の選択肢から正しいものを1つ選びなさい。医療ガイドラインに記載されている、現在行われている標準的な治療法に基づいて答えなさい。\n"
                             "質問：{question}\n選択肢：\n{options}\n答え：",

    "context_based_mcqa_instructed": "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? "
                                        "Answer yes, no, or maybe.\n"
                                        "Abstract: {context}\nQuestion: {question}\nAnswer:",

    "context_based_mcqa_instructed_jp": "臨床科学と医学知識の専門家である医師として、次の文が正しいかどうか教えてください。「はい/いいえ/たぶん」のいずれかでお答えください。\n"
                                        "要旨：{context}\n質問：{question}\n答え：",

    "llama-2-chat": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.  
Please answer the following Japanese medical question, by selecting the most appropriate answer from the options below.
<</SYS>>

Question: {question}
Options: """,

    "llm-jp": """

### 指示:
質問と回答の選択肢を入力として受け取り、選択肢から正しい回答を選択してください。

### 質問:
{question}
### 選択肢: {option_text}

### 応答:
""",


############################ DC #############################

    #  "dc_minimal" -> "context_based_mcqa_minimal"
    "dc_with_options": "Context: {context}\nQuestion: {question}\n{options}\nAnswer:",
    "dc_with_options_jp": "文脈：{context}\n質問：{question}\n{options}\n答え：",
    "dc_instructed_jp": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、以下の選択式問題に答えなさい。"
                        "以下の選択肢から正しいものを1つ選びなさい。\n文脈：{context}\n質問：{question}\n選択肢：\n{options}\n答え："
}


class MCQATemplate(Template):
    def __init__(self, template_name: str):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        options = []
        for i in range(sample.n_options):
            option_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
            option = sample.options[i]
            # options.append(f"({option_idx}) {option}")
            options.append(f"{option_idx}. {option}")
        # option_text = " ".join(options)
        option_text = "\n".join(options)

        if self.template_name in ["mcqa", "mcqa_minimal"]:
            return self.template.format(
                question=sample.question
            )

        elif self.template_name in ["mcqa_with_options", "mcqa_with_options_jp", "4o_mcqa_instructed", "4o_mcqa_instructed_jp"]:
            return self.template.format(
                question=sample.question,
                options=option_text
            )

        elif self.template_name in ["context_based_mcqa", "context_based_mcqa_minimal", "context_based_mcqa_jp", "context_based_mcqa_instructed_jp"]:
            return self.template.format(
                context=sample.metadata["context"],
                question=sample.question
            )

        elif self.template_name in ["context_based_mcqa_with_options", "dc_with_options", "dc_with_options_jp", "dc_instructed_jp"]:
            return self.template.format(
                context=sample.metadata["context"],
                question=sample.question,
                options=option_text
            )

        else:
            raise NotImplementedError

    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f" {sample.options[sample.answer_idx]}"
