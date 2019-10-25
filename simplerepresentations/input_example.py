class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, text_a, text_b=None):
        """
        Constructs an InputExample.

        Args:
            guid: integer. Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """

        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
