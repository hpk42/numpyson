import datetime as dt

import pytest
import numpy as np
import pandas as pd

from numpyson import dumps, loads

@pytest.mark.parametrize('arr', [
    np.array([1, 2, 3]),
    np.array([1., 2., 3.]),
    np.array(['foo', 'bar', 'baz']),
    np.array([dt.datetime(1970, 1, 1, 12, 57), dt.datetime(1970, 1, 1, 12, 58), dt.datetime(1970, 1, 1, 12, 59)]),
    np.array([dt.date(1970, 1, 1), dt.date(1970, 1, 2), dt.date(1970, 1, 3)]),
])
def test_numpy_array_handler(arr):
    buf = dumps(arr)
    arr_after = loads(buf)
    assert (arr == arr_after).all()

def test_nested_array():
    d = {"1": np.array([1,2])}
    buf = dumps(d)
    d_after = loads(buf)
    assert (d["1"] == d_after["1"]).all()
    

@pytest.mark.parametrize('ts', [
    pd.TimeSeries([1, 2, 3], index=[0, 1, 2]),
    pd.TimeSeries([1., 2., 3.], pd.date_range('1970-01-01', periods=3, freq='S')),
    pd.date_range('1970-01-01', periods=3, freq='S'),
])
def test_pandas_timeseries_handler(ts):
    buf = dumps(ts)
    ts_after = loads(buf)
    assert (ts == ts_after).all()

def test_timeseries_nested():
    d = {"1": pd.date_range('1970-01-01', periods=3, freq='S')}
    buf = dumps(d)
    d_after = loads(buf)
    assert (d["1"] == d_after["1"]).all()

@pytest.mark.parametrize('df', [
    pd.DataFrame({0: [1, 2, 3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='S')),
])
def test_pandas_dataframe_handler(df):
    buf = dumps(df)
    ts_after = loads(buf)
    assert (df == ts_after).all().all()


def test_mixed_python_and_pandas_types():
    data = (
        np.array([1., 2., 3.]),
        pd.TimeSeries([1, 2, 3], index=[0, 1, 2]),
        pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='S'))
    )
    buf = dumps(data)
    data_after = loads(buf)

    assert isinstance(data, tuple)
    assert len(data) == 3
    assert (data[0] == data_after[0]).all()
    assert (data[1] == data_after[1]).all()
    assert (data[2] == data_after[2]).all().all()
