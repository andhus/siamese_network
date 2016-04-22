from __future__ import division, print_function

from theano import tensor
from blocks.bricks.cost import CostMatrix


class ContrastiveLoss(CostMatrix):

    def __init__(self, q, *args, **kwargs):
        super(ContrastiveLoss, self).__init__(*args, **kwargs)
        self.q = q
        # could default to prod(x1.shape[1:]) assuming max size 1 of each
        # feature

    def cost_matrix(self, x1, x2, y1, y2):
        y = tensor.switch(abs(y1-y2) > 0, 0, 1)  # y = 1 means genuine pair
        contrastive_loss = (
            (1 - y) * (2 / self.q) * tensor.square(x1 - x2).sum(axis=1, keepdims=True) +
            y * 2 * self.q * tensor.exp(-(2.77 / self.q) * abs(x1 - x2).sum(axis=1, keepdims=True))
        )

        return contrastive_loss
