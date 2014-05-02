numpyson
========

cross py2/py3 python object serializer, currently using and extending jsonpickle 
to dump/load python objects with a primary focus on numpy/pandas types.

The main purpose of ``numpyson`` is to allow dumping under a python3 interpreter
and loading under python2 and vice versa.  This is a use case that is not covered
by current serializers ASFAIK.

quick simple API example::

    import numpyson
    import numpy
    s = numpyson.dumps(numpy.array([1,2,3]))
    data = numpyjson.loads(s)
    assert data.to_list() == [1,2,3]

Currently supported:

- ``numpy arrays`` and some other numpy types
- ``pandas.DataFrame``
- ``pandas.date_range``
- ``pandas.TimeSeries``
- nested python data structures with the above types
