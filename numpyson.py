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

import numpy as np
import pandas as pd

import jsonpickle.handlers
import jsonpickle.util


class NumpyArrayHandler(jsonpickle.handlers.BaseHandler):
    """A jsonpickle handler for numpy (de)serialising arrays."""

    def flatten(self, obj, data):
        pickler = self.context
        flatten = pickler.flatten
        buffer = jsonpickle.util.b64encode(obj.tostring())
        #TODO: should probably also consider including other parameters in future such as byteorder, etc.
        #TODO: see numpy.info(obj) and obj.__reduce__() for details.
        shape = flatten(obj.shape)
        dtype = str(obj.dtype)
        strides = flatten(obj.strides)
        args = [shape, dtype, strides, buffer]
        data['__reduce__'] = (flatten(np.ndarray, reset=False), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        unpickler = self.context
        restore = unpickler.restore
        cls = restore(cls, reset=False)
        shape = restore(args[0])
        dtype = np.dtype(restore(args[1]))
        strides = restore(args[2])
        buffer = jsonpickle.util.b64decode(args[3])
        return cls(shape=shape, dtype=dtype, buffer=buffer, strides=strides)


class PandasTimeSeriesHandler(jsonpickle.handlers.BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas TimeSeries objects."""

    def flatten(self, obj, data):
        pickler = self.context
        flatten = pickler.flatten
        values = flatten(obj.values)
        index = flatten(obj.index.values)
        args = [values, index]
        data['__reduce__'] = (flatten(pd.TimeSeries), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        unpickler = self.context
        restore = unpickler.restore
        cls = restore(cls, reset=False)
        values = restore(args[0])
        index = restore(args[1])
        return cls(data=values, index=index)


class PandasDataFrameHandler(jsonpickle.handlers.BaseHandler):
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
        unpickler = self.context
        restore = unpickler.restore
        cls = restore(cls, reset=False)
        values = restore(args[0])
        index = restore(args[1])
        columns = restore(args[2])
        return cls(dict(zip(columns, values)), index=index)

def register_handlers():
    """Call this function to register handlers with jsonpickle module."""
    NumpyArrayHandler.handles(np.ndarray)
    PandasTimeSeriesHandler.handles(pd.TimeSeries)
    PandasDataFrameHandler.handles(pd.DataFrame)
