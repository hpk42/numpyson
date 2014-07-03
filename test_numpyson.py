import datetime as dt
from functools import partial
import inspect

import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_index_equal, assert_series_equal, assert_frame_equal
from numpy.testing import assert_equal

assert_series_equal_strict = partial(assert_series_equal, check_dtype=True, check_index_type=True,
                                     check_series_type=True, check_less_precise=False)

assert_frame_equal_strict = partial(assert_frame_equal, check_dtype=True, check_index_type=True,
                                    check_column_type=True, check_frame_type=True, check_less_precise=False,
                                    check_names=True)

from numpyson import dumps, loads, build_index_handler_for_type


def test_version():
    import numpyson
    assert numpyson.__version__


@pytest.mark.parametrize('arr_before', [
    np.array([1, 2, 3]),
    np.array([1., 2., 3.]),
    np.array(['foo', 'bar', 'baz']),
    np.array([dt.datetime(1970, 1, 1, 12, 57), dt.datetime(1970, 1, 1, 12, 58), dt.datetime(1970, 1, 1, 12, 59)]),
    np.array([dt.date(1970, 1, 1), dt.date(1970, 1, 2), dt.date(1970, 1, 3)]),
    np.array([True, False, True]),
    np.arange(10).T,
    np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
    np.array([[[1., 10.], [4., 40.], [7., 70.]], [[2., 20.], [5., 50.], [8., 80.]], [[3., 30.], [6., 60.], [9., 90.]]]),
    np.reshape(np.arange(100), (10, 10)),
    np.reshape(np.arange(100).T, (10, 10)),
])
def test_numpy_array_handler(arr_before):
    buf = dumps(arr_before)
    arr_after = loads(buf)
    assert_equal(arr_before, arr_after)


def test_nested_array():
    data_before = {"1": np.array([1, 2])}
    buf = dumps(data_before)
    data_after = loads(buf)
    assert_equal(data_before["1"], data_after["1"])



@pytest.mark.parametrize('ts_before', [
    pd.TimeSeries([1, 2, 3], index=[0, 1, 2]),
    pd.TimeSeries([1., 2., 3.], pd.date_range('1970-01-01', periods=3, freq='S')),
    pd.TimeSeries([1., 2., 3.], pd.date_range('1970-01-01', periods=3, freq='D')),
])
def test_pandas_timeseries_handler(ts_before):
    buf = dumps(ts_before)
    ts_after = loads(buf)
    assert_series_equal_strict(ts_before, ts_after)


@pytest.mark.parametrize('index_before', [
    pd.Index([0, 1, 2]),
    pd.Index([0., 1., 2.]),  # not sure why you would want to index by floating point numbers; here for completeness
    pd.Index(['a', 'b', 'c']),
])
def test_pandas_index_handler(index_before):
    buf = dumps(index_before)
    index_after = loads(buf)
    assert_index_equal(index_before, index_after)


@pytest.mark.parametrize('index_before', [
    pd.date_range('1970-01-01', periods=3, freq='S'),
    pd.date_range('1970-01-01', periods=3, freq='D'),
])
def test_pandas_datetime_index_handler(index_before):
    buf = dumps(index_before)
    index_after = loads(buf)
    assert_index_equal(index_before, index_after)


@pytest.mark.parametrize('data_before', [
    {"1": pd.date_range('1970-01-01', periods=3, freq='S')},
    {"1": pd.date_range('1970-01-01', periods=3, freq='D')},
])
def test_datetime_index_nested(data_before):
    buf = dumps(data_before)
    data_after = loads(buf)
    assert_index_equal(data_before["1"], data_after["1"])


TEST_DATA_FRAMES = (
    pd.DataFrame({0: [1, 2, 3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='S')),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='D')),
    pd.DataFrame({'a': [1, 2, 3], 'b': [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='D')),
    pd.DataFrame({
            'i': [1, 2, 3],
            'f': [1.1, 2.2, 3.3],
            's': ['ham', 'spam', 'eggs'],
            'b': [True, False, True],
            'o': [{'a': 1}, {'b': 2}, {'c': 3}],
        },
        index=pd.date_range('1970-01-01', periods=3, freq='S')),
    pd.DataFrame(np.ones(shape=(10,15)), index=pd.date_range('1970-01-01', periods=10))
)


@pytest.mark.parametrize('df_before', TEST_DATA_FRAMES)
def test_pandas_dataframe_handler(df_before):
    buf = dumps(df_before)
    df_after = loads(buf)
    assert_frame_equal_strict(df_before, df_after)


def test_mixed_python_and_pandas_types():
    data_before = TEST_DATA_FRAMES
    buf = dumps(data_before)
    data_after = loads(buf)

    assert isinstance(data_after, tuple)
    assert len(data_after) == len(TEST_DATA_FRAMES)
    assert len(data_before) == len(data_after)
    for df_before, df_after in zip(data_before, data_after):
        assert_frame_equal_strict(df_before, df_after)


def test_build_index_handler_for_type():
    for index_class in ():
        handler_cls = build_index_handler_for_type(index_class)
        assert inspect.isclass(handler_cls)
        assert hasattr(handler_cls, 'flatten')
        assert hasattr(handler_cls, 'restore')

    with pytest.raises(TypeError):
        build_index_handler_for_type(pd.DatetimeIndex)

    with pytest.raises(TypeError):
        build_index_handler_for_type(pd.TimeSeries)

@pytest.mark.xfail(reason='failing preserve underlying array state when it is wrapped inside a Pandas object')
def test_preservation_of_specific_array_ordering():
    df_c = pd.DataFrame(np.array([[1,2],[3,4]], order='C'))
    df_c_after = loads(dumps(df_c))
    assert_frame_equal_strict(df_c, df_c_after)
    assert_equal(df_c.values, df_c_after.values)
    assert not df_c.values.flags.fortran
    assert not df_c_after.values.flags.fortran

    df_f = pd.DataFrame(np.array([[1,2],[3,4]], order='F'))
    df_f_after = loads(dumps(df_f))
    assert_frame_equal_strict(df_f, df_f_after)
    assert_equal(df_f.values, df_f_after.values)
    assert df_f.values.flags.fortran
    assert df_f_after.values.flags.fortran

def test_preservation_of_specific_array_ordering_simple():
    arr_c = np.array([[1,2],[3,4]], order='C')
    arr_f = np.array([[1,2],[3,4]], order='F')

    assert_equal(arr_c, arr_f)

    assert arr_c.strides != arr_f.strides

    #   C array ordering
    arr_c_after = loads(dumps(arr_c))
    assert arr_c.strides == arr_c_after.strides
    assert not arr_c.flags.fortran
    assert not arr_c_after.flags.fortran

    #   Fortran array order
    arr_f_after = loads(dumps(arr_f))
    assert arr_f.strides == arr_f_after.strides
    assert arr_f.flags.fortran
    assert arr_f_after.flags.fortran




@pytest.mark.parametrize("val", [np.float64(4.2), np.int64(5)])
def test_number(val):
    dumped = dumps(val)
    loaded = loads(dumped)
    assert loaded == val
    assert type(loaded) == type(val)

def test_datetime_identity():
    import datetime
    date = datetime.datetime(2013, 11, 1, 0, 0)
    val = {
        'start': date,
        'end': date,
        'd': {"ttf": pd.TimeSeries([1.],
                                pd.date_range("1970-1-1", periods=1, freq='S'))
        }
    }
    dumped = dumps(val)
    loaded = loads(dumped)
    assert loaded["start"] == val["start"], dumped
    assert loaded["end"] == val["end"]
    assert loaded["end"] == val["end"]

