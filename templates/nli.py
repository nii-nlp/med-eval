from templates.base import Template

TEMPLATE_SET = {

    "standard": "Premise: {premise}\nHypothesis: {hypothesis}\nEntailment:",

    "fact_verification": "Explanation: {premise}\n"
                         "Claim: {hypothesis}\n"
                         "Which veracity label would you give to the claim taking into account the entire explanation? (A) Mixture (B) False (C) True (D) Unproven\n"
                         "Answer:",

    "rqe": "Is the answer to 'Q1: {hypothesis}' also a complete or partial answer to 'Q2: {premise}'? =>",

    "nli_as_qa_for_healthver": "Context: {premise}\nQuestion: Based on the given context, if the following claim is supported or contradict or not enough information?\n{hypothesis}\nAnswer:",

    "nli_as_qa_for_pubhealth": "Question: Here is a claim: {hypothesis}\nBased on the following explanation: {premise}\nDo you think the claim is true or false or partially true (mixture) or unproven?\nAnswer:",


######################### STS #########################
    "sts_as_nli": "Text 1: {premise}\nText 2: {hypothesis}\nSemantic Text Similarity (0-5):",
    "sts_minimal": "{premise}\n{hypothesis}\n",
    "sts_as_nli_jp": "テキスト1：{premise}\nテキスト2：{hypothesis}\n類似度 (0-5)：",
    "sts_instructed_jp": "あなたは医学博士です。基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムについて理解した上で、"
                         "次の2つの文の意味的類似度を0から5の範囲で判断してください。\n"
                         "0：二つの文は完全に似ていない。\n"
                         "5：二つの文は完全に同等で、意味が同じである。\n"
                         "テキスト1: {premise}\nテキスト2: {hypothesis}\n類似度 (0-5)："
}


class NLITemplate(Template):
    def __init__(self, template_name: str):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        if self.template_name in ["standard", "fact_verification", "rqe", "nli_as_qa"]:
            return self.template.format(
                premise=sample.premise,
                hypothesis=sample.hypothesis
            )

        else:
            return self.template.format(
                premise=sample.premise,
                hypothesis=sample.hypothesis
            )

    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f" {sample.label}"
