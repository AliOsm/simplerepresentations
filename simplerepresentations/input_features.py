class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
