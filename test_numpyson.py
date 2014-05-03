import datetime as dt
from functools import partial

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

from numpyson import dumps, loads


@pytest.mark.parametrize('arr_before', [
    np.array([1, 2, 3]),
    np.array([1., 2., 3.]),
    np.array(['foo', 'bar', 'baz']),
    np.array([dt.datetime(1970, 1, 1, 12, 57), dt.datetime(1970, 1, 1, 12, 58), dt.datetime(1970, 1, 1, 12, 59)]),
    np.array([dt.date(1970, 1, 1), dt.date(1970, 1, 2), dt.date(1970, 1, 3)]),
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
    assert len(data_after) == 5
    assert len(data_before) == len(data_after)
    for df_before, df_after in zip(data_before, data_after):
        assert_frame_equal_strict(df_before, df_after)
