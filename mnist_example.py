from __future__ import division, print_function


import numpy as np
import theano

from theano import tensor

from blocks.algorithms import GradientDescent, AdaDelta
from blocks.bricks.cost import SquaredError, BinaryCrossEntropy
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, \
    DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.bricks import MLP, Rectifier, Tanh, Logistic, Identity, BatchNormalizedMLP
from blocks.initialization import IsotropicGaussian, Constant
from blocks_extras.extensions.plot import Plot

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift, Rename, Merge


seed = 123
batch_size = 1000
np.random.seed(seed=seed)

mnist_train = MNIST(which_sets=('train',), subset=range(10000))
mnist_test = MNIST(which_sets=('test',), subset=range(1000))


def _data_stream(
    dataset,
    batch_size
):
    data_stream_ = DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=ShuffledScheme(
            examples=dataset.num_examples,
            batch_size=batch_size
        )
    )

    return data_stream_


def pair_data_stream(
    dataset,
    batch_size
):
    data_streams = [
        Rename(
            _data_stream(dataset=dataset, batch_size=batch_size),
            names={source: '{}_{}'.format(source, i) for source in dataset.sources}
        ) for i in [1, 2]
    ]
    data_stream = Merge(
        data_streams=data_streams,
        sources=data_streams[0].sources + data_streams[1].sources
    )
    _ = data_streams[0].get_epoch_iterator()  # make sure not same order

    return data_stream


def apply_transformers(data_stream):

    data_stream_ = Flatten(
        data_stream,
        which_sources=['features_1', 'features_2']
    )
    data_stream_ = ScaleAndShift(
        data_stream_,
        which_sources=['features_1', 'features_2'],
        scale=2.0,
        shift=-1.0
    )

    return data_stream_


train_stream = apply_transformers(pair_data_stream(mnist_train, batch_size))
test_stream = apply_transformers(pair_data_stream(mnist_test, batch_size))


dims = [28*28, 128, 128, 64]

encoder = MLP(
    activations=[Rectifier(), Rectifier(), Logistic()],
    dims=dims,
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
encoder.initialize()

f1 = tensor.matrix('features_1')
f2 = tensor.matrix('features_2')
y1 = tensor.matrix('targets_1')
y2 = tensor.matrix('targets_2')

x1 = encoder.apply(f1)
x2 = encoder.apply(f2)

from cost import ContrastiveLoss

cost = ContrastiveLoss(q=dims[-1]).apply(
    x1=x1,
    x2=x2,
    y1=y1,
    y2=y2
)
cost.name = 'contrastive_loss'
cost_test = cost.copy('contrastive_loss_test')

cg = ComputationGraph(cost)

algorithm = GradientDescent(
    cost=cost,
    step_rule=AdaDelta(),
    parameters=cg.parameters
)

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=train_stream,
    extensions=[
        TrainingDataMonitoring(
            variables=[cost],
            after_epoch=True
        ),
        DataStreamMonitoring(
            data_stream=test_stream,
            variables=[cost_test],
            after_epoch=True
        ),
        Printing(after_epoch=True),
        FinishAfter(after_n_epochs=50),
        Plot(
            document='siamese network larger',
            channels=[[cost.name, cost_test.name]],
            start_server=True,
            after_epoch=True
        )
    ]
)
