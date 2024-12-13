from templates.base import Template


TEMPLATE_SET = {

    "english_japanese": "翻訳（English => 日本語）：{source_text} => ",

    "japanese_english": "Translation (日本語 => English): {source_text} => ",

    "mt_minimal": "{source_text} => ",

    "mt_english_centric_j2e": "Translation (Japanese => English): {source_text} => ",
    "mt_english_centric_e2j": "Translation (English => Japanese): {source_text} => ",

    "mt_english_centric_instructed_e2j": "You are a medical doctor translating biomedical documents. "
                                         "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, "
                                         "disease, patient care, and modes of therapy, translate the following English text to Japanese.\n"
                                         "{source_text} => ",

    "mt_instructed_e2j": "あなたは生物医学文書を翻訳する医学博士です。"
                         "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                         "以下の英文を和訳しなさい。\n"
                         "{source_text} => ",

    "mt_instructed_j2e": "あなたは生物医学文書を翻訳する医学博士です。"
                         "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                         "以下の和文を英訳しなさい。\n"
                         "{source_text} => ",
}


class MTTemplate(Template):
    def __init__(
            self,
            template_name: str,
    ):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        if not isinstance(sample, dict):
            sample = sample.to_dict()

        return self.template.format(
            source_text=sample["source_text"]
        )

    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f"{sample['target_text']}"
