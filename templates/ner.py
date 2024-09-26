from templates.base import Template

TEMPLATE_SET = {

    "standard": "段落: {text} => {entity_type}:",   #  {entities}
    "minimal": "段落: {text} => {entity_type}:",
    "english-centric": "Paragraph: {text} => {entity_type}:",
    "instructed": "段落: {text} => {entity_type}:"
}


class NERTemplate(Template):
    def __init__(self, template_name: str):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        en2jp_mapping = {
            "Disease": "患者に実際に認められた病変や症状の存在を示す異常などの所見を表す表現",
            "Medicine Value": "処方量など薬品にかかわる値",
            "Medicine Key": "薬品名"
        }
        if sample.entity_type in en2jp_mapping.keys():
            sample.entity_type = en2jp_mapping[sample.entity_type]
        
        # print(self.template.format(
        #         text=sample.text,
        #         entity_type=entity_type
        #     ))
        if self.template_name in ["standard"]:
            return self.template.format(
                text=sample.text,
                entity_type=sample.entity_type
            )

        else:
            return self.template.format(
                text=sample.text,
                entity_type=sample.entity_type
            )
    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f" {', '.join(sample.labels)}"
