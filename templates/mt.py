from templates.base import Template


TEMPLATE_SET = {

    "english_japanese": "翻訳（English => 日本語）：{source_text} =>",

    "japanese_english": "Translation (日本語 => English): {source_text} =>",

    "mt_minimal": "{source_text} =>",

    "mt_english_centric_j2e": "Translation (Japanese => English): {source_text} =>",
    "mt_english_centric_e2j": "Translation (English => Japanese): {source_text} =>",

    "mt_english_centric_instructed_e2j": "You are a medical doctor translating biomedical documents. "
                                         "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, "
                                         "disease, patient care, and modes of therapy, translate the following English text to Japanese.\n"
                                         "{source_text} =>",

    "mt_instructed_e2j": "あなたは生物医学文書を翻訳する医学博士です。"
                         "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                         "以下の英文を和訳しなさい。\n"
                         "{source_text} =>",

    "mt_instructed_j2e": "あなたは生物医学文書を翻訳する医学博士です。"
                         "基礎科学、臨床科学、医学知識、健康、病気、患者ケア、治療法の基礎となるメカニズムを理解した上で、"
                         "以下の和文を英訳しなさい。\n"
                         "{source_text} =>",


#     "few-shot_english_japanese": """翻訳（English => 日本語）：
# At present, the document manual is used in transfer assistance of the muscular dystrophy patient. => 現在，筋ジストロフィー患者の移動介助において文書マニュアルを使用している。
# The use concentration and the amount of lidocaine were 0.5～10% and 0.1～1.0ml(10～60mg) respectively. => リドカイン使用濃度，使用量は0.5～10％，0.1～1.0ml（10～60mg）であった。
# A 43‐year‐old female was seen at our clinic with complaints of high fever and erythroderma like skin rashes, which have developed in 3 weeks after her heart operation. => 症例は43歳の女性で，心臓弁膜症手術後22日目頃より，発熱と共に全身に紅斑が出現した。
# {source_text} =>""",
#
#     "few-shot_japanese_english": """Translation (日本語 => English):
# 現在，筋ジストロフィー患者の移動介助において文書マニュアルを使用している。 => At present, the document manual is used in transfer assistance of the muscular dystrophy patient.
# リドカイン使用濃度，使用量は0.5～10％，0.1～1.0ml（10～60mg）であった。 => The use concentration and the amount of lidocaine were 0.5～10% and 0.1～1.0ml(10～60mg) respectively.
# 症例は43歳の女性で，心臓弁膜症手術後22日目頃より，発熱と共に全身に紅斑が出現した。 => A 43‐year‐old female was seen at our clinic with complaints of high fever and erythroderma like skin rashes, which have developed in 3 weeks after her heart operation.
# {source_text} =>"""

}


class MTTemplate(Template):
    def __init__(
            self,
            template_name: str,
    ):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        return self.template.format(
            source_text=sample.source_text
        )

    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f" {sample.target_text}"
