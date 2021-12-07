class Metrics():
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.loss_value = 0
        self.batches = 0

    def add_batch(self, prediction, target):
        # TODO(fix this)
        loss_value = self.loss_function(prediction, target)
        self.loss_value = (self.loss_value * self.batches + loss_value) / \
                          (self.batches + 1)
        self.batches += 1

    def reset_metrics(self):
        self.loss_value = 0
        self.batches = 0
