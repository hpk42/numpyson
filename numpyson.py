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

class BaseHandler(jsonpickle.handlers.BaseHandler):
    def nrestore(self, arg, reset=False):
        return self.context.restore(arg, reset=reset)


class NumpyArrayHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising arrays."""

    def flatten(self, obj, data):
        buffer = jsonpickle.util.b64encode(obj.tostring())
        #TODO: should probably also consider including other parameters in future such as byteorder, etc.
        #TODO: see numpy.info(obj) and obj.__reduce__() for details.
        flatten = self.context.flatten
        shape = flatten(obj.shape)
        dtype = str(obj.dtype)
        strides = flatten(obj.strides)
        args = [shape, dtype, strides, buffer]
        data['__reduce__'] = (flatten(np.ndarray, reset=False), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        shape = self.nrestore(args[0])
        dtype = np.dtype(self.nrestore(args[1]))
        strides = self.nrestore(args[2])
        buffer = jsonpickle.util.b64decode(args[3])
        return cls(shape=shape, dtype=dtype, buffer=buffer, strides=strides)


class PandasTimeSeriesHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas TimeSeries objects."""

    def flatten(self, obj, data):
        flatten = self.context.flatten
        values = flatten(obj.values)
        index = flatten(obj.index.values)
        args = [values, index]
        data['__reduce__'] = (flatten(pd.TimeSeries), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls)
        values = self.nrestore(args[0])
        index = self.nrestore(args[1])
        return cls(data=values, index=index)

class PandasDateTimeIndexHandler(BaseHandler):
    """A jsonpickle handler for numpy (de)serialising pandas TimeSeries objects."""

    def flatten(self, obj, data):
        flatten = self.context.flatten
        values = flatten(obj.values)
        freq = flatten(obj.freq)
        args = [values, freq]
        data['__reduce__'] = (flatten(pd.DatetimeIndex), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        cls = self.nrestore(cls, reset=False)
        values = self.nrestore(args[0])
        freq = self.nrestore(args[1])
        return cls(data=values, freq=freq)



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
        cls = self.nrestore(cls, reset=False)
        values = self.nrestore(args[0])
        index = self.nrestore(args[1])
        columns = self.nrestore(args[2])
        return cls(dict(zip(columns, values)), index=index)

def register_handlers():
    """Call this function to register handlers with jsonpickle module."""
    PandasDateTimeIndexHandler.handles(pd.DatetimeIndex)
    NumpyArrayHandler.handles(np.ndarray)
    PandasTimeSeriesHandler.handles(pd.TimeSeries)
    PandasDataFrameHandler.handles(pd.DataFrame)

def dumps(obj):
    register_handlers()
    return jsonpickle.encode(obj).encode("utf-8")

def loads(obj):
    register_handlers()
    return jsonpickle.decode(obj.decode("utf-8"))
