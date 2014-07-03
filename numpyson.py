"""
transparent serialization of numpy/pandas data via jsonpickle.
compatible to python2.7 and python3.3 and allows to serialize
between the two interpreters.

majorly based on code and ideas of David Moss in his MIT licensed pdutils
repository: https://github.com/drkjam/pdutils

Note that the serialization/deserialization is not space-efficient
due to the nature of json/jsonpickle. You could certainly save space
by compressing/decompressing the resulting json output if you need to.

(C) David Moss, Holger Krekel 2014
"""

__version__ = '0.3'
import numpy as np
import pandas as pd

import jsonpickle.handlers
import jsonpickle.util


class BaseHandler(jsonpickle.handlers.BaseHandler):
    def nrestore(self, arg, reset=False):
        return self.context.restore(arg, reset=reset)

    def nflatten(self, arg, reset=False):
        return self.context.flatten(arg, reset=reset)


class NumpyNumber(BaseHandler):
    def flatten(self, obj, data):
        data["__reduce__"] = (self.nflatten(type(obj)), [float(obj)])
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        return cls(args[0])


class NumpyArrayHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising arrays."""
    def flatten(self, obj, data):
        buf = jsonpickle.util.b64encode(obj.tostring())
        #TODO: should probably also consider including other parameters in future such as byteorder, etc.
        #TODO: see numpy.info(obj) and obj.__reduce__() for details.
        shape = self.nflatten(obj.shape)
        dtype = str(obj.dtype)
        order = 'F' if obj.flags.fortran else 'C'
        args = [shape, dtype, buf, order]
        data['__reduce__'] = (self.nflatten(np.ndarray), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        shape = self.nrestore(args[0])
        dtype = np.dtype(self.nrestore(args[1]))
        buf = jsonpickle.util.b64decode(args[2])
        order = args[3]
        return cls(shape=shape, dtype=dtype, buffer=buf, order=order)


class PandasTimeSeriesHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas TimeSeries objects."""

    def flatten(self, obj, data):
        values = self.nflatten(obj.values)
        index = self.nflatten(obj.index.values)
        args = [values, index]
        data['__reduce__'] = (self.nflatten(pd.TimeSeries), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        cls = self.nrestore(cls)
        values = self.nrestore(args[0])
        index = self.nrestore(args[1])
        return cls(data=values, index=index)


class PandasDateTimeIndexHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas DateTimeIndex objects."""

    def flatten(self, obj, data):
        values = self.nflatten(obj.values)
        freq = self.nflatten(obj.freq)
        args = [values, freq]
        data['__reduce__'] = (self.nflatten(pd.DatetimeIndex), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls, reset=False)
        values = self.nrestore(args[0])
        freq = self.nrestore(args[1])
        return cls(data=values, freq=freq)


def build_index_handler_for_type(index_class):
    """A class factor that builds jsonpickle handlers for various index types."""
    if not issubclass(index_class, pd.Index) or index_class == pd.DatetimeIndex:
        raise TypeError('expected a subclass of pandas.Index, got %s' % type(index_class))

    class _IndexHandler(BaseHandler):
        """A jsonpickle handler for numpy (de)serialising pandas Index objects."""
        def flatten(self, obj, data):
            values = self.nflatten(obj.values)
            args = [values]
            data['__reduce__'] = (self.nflatten(index_class), args)
            return data

        def restore(self, obj):
            cls, args = obj['__reduce__']
            cls = self.nrestore(cls)
            values = self.nrestore(args[0])
            return cls(data=values)

    return _IndexHandler

PandasInt64IndexHandler = build_index_handler_for_type(pd.Int64Index)
PandasFloat64IndexHandler = build_index_handler_for_type(pd.Float64Index)
PandasIndexHandler = build_index_handler_for_type(pd.Index)


class PandasDataFrameHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas DataFrame objects."""

    def flatten(self, obj, data):
        pickler = self.context
        flatten = pickler.flatten
        values = [flatten(obj[col].values) for col in obj.columns]
        index = flatten(obj.index.values)
        columns = flatten(obj.columns.values)
        args = [values, index, columns]
        data['__reduce__'] = (flatten(pd.DataFrame), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        values = self.nrestore(args[0])
        index = self.nrestore(args[1])
        columns = self.nrestore(args[2])
        return cls(dict(zip(columns, values)), index=index)


def register_handlers():
    """Call this function to register handlers with jsonpickle module."""
    NumpyNumber.handles(np.float64)
    NumpyNumber.handles(np.int64)
    NumpyArrayHandler.handles(np.ndarray)

    PandasIndexHandler.handles(pd.Index)
    PandasDateTimeIndexHandler.handles(pd.DatetimeIndex)
    PandasInt64IndexHandler.handles(pd.Int64Index)
    PandasFloat64IndexHandler.handles(pd.Float64Index)

    PandasTimeSeriesHandler.handles(pd.TimeSeries)

    PandasDataFrameHandler.handles(pd.DataFrame)


def dumps(obj):
    register_handlers()
    return jsonpickle.encode(obj, unpicklable=True).encode("utf-8")

    #from jsonpickle.pickler import _make_backend, Pickler
    #backend = _make_backend(None)
    #context = Pickler(unpicklable=True,
    #                  make_refs=True,
    #                  keys=False,
    #                  backend=backend,
    #                  max_depth=None)
    #context._mkref = lambda x: True
    #return backend.encode(context.flatten(obj, reset=False)).encode("utf-8")


def loads(obj):
    register_handlers()
    return jsonpickle.decode(obj.decode("utf-8"))
