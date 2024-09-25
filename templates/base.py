class Template:
    def __init__(self, template_name: str):
        self.template_name = template_name

    def instantiate_template(self, sample):
        raise NotImplementedError

    def instantiate_template_full(self, sample):
        raise NotImplementedError
