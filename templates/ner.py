from templates.base import Template

TEMPLATE_SET = {

    "standard": "Paragraph: {text} => {entity_type}:",   #  {entities}

}


class NERTemplate(Template):
    def __init__(self, template_name: str):
        super().__init__(template_name)
        self.template = TEMPLATE_SET[template_name]

    def instantiate_template(self, sample):
        if self.template_name in ["standard"]:
            return self.template.format(
                text=sample.text,
                entity_type=sample.entity_type[0].capitalize() + sample.entity_type[1:]
            )

        else:
            return self.template.format(
                text=sample.text,
                entity_type=sample.entity_type[0].capitalize() + sample.entity_type[1:]
            )

    def instantiate_template_full(self, sample):
        return self.instantiate_template(sample) + f" {', '.join(sample.labels)}"
