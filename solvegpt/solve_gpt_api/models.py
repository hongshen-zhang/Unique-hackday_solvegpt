from tortoise import fields
from tortoise.models import Model


class Question(Model):
    id = fields.IntField(pk=True)
    question = fields.TextField()
    answer = fields.TextField()

    def __init__(self, *args, accuracy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy = accuracy

    class Meta:
        table = "question"

    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "accuracy": getattr(self, 'accuracy', None),
        }
