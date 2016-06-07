from __future__ import division, print_function


from fuel.transformers import Transformer


class SplitSource(Transformer):

    def __init__(self, datastream, which_sources=None, sufixes=('1', '2')):
        super(SplitSource, self).__init__(
            data_stream=datastream,
            produces_examples=False,
            which_sources=which_sources
        )

        self._verify_sufixes(sufixes)
        self.suffixes = sufixes
        self.bypass_sources = tuple(filter(
            lambda source: source not in self.which_sources,
            self.data_stream.sources
        ))
        split_sources = []
        for source in self.which_sources:
            for sufix in self.suffixes:
                self.split_sources.append(
                    '{}_{}'.format(source, sufix)
                )
        self.split_sources = tuple(split_sources)

    @property
    def sources(self):
        return self.bypass_sources + self.split_sources

    @sources.setter
    def sources(self, value):
        raise AttributeError('sources can not be set')

    def _verify_sufixes(self, sufixes):
        assert isinstance(sufixes, tuple)
        assert len(sufixes, 2)
        assert isinstance(sufixes[0], str)
        assert isinstance(sufixes[1], str)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError()
        source_to_batch = zip(
            self.data_stream.sources,
            next(self.child_epoch_iterator)
        )

