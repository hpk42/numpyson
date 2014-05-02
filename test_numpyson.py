import datetime as dt

import pytest
import jsonpickle
import numpy as np
import pandas as pd

from numpyson import register_handlers

@pytest.fixture(scope="session", autouse=True)
def reg():
    register_handlers()

@pytest.mark.parametrize('arr', [
    np.array([1, 2, 3]),
    np.array([1., 2., 3.]),
    np.array(['foo', 'bar', 'baz']),
    np.array([dt.datetime(1970, 1, 1, 12, 57), dt.datetime(1970, 1, 1, 12, 58), dt.datetime(1970, 1, 1, 12, 59)]),
    np.array([dt.date(1970, 1, 1), dt.date(1970, 1, 2), dt.date(1970, 1, 3)]),
])
def test_numpy_array_handler(arr):
    buf = jsonpickle.encode(arr)
    arr_after = jsonpickle.decode(buf)
    assert (arr == arr_after).all()

@pytest.mark.parametrize('ts', [
    pd.TimeSeries([1, 2, 3], index=[0, 1, 2]),
    pd.TimeSeries([1., 2., 3.], pd.date_range('1970-01-01', periods=3, freq='S'))
])
def test_pandas_timeseries_handler(ts):
    buf = jsonpickle.encode(ts)
    ts_after = jsonpickle.decode(buf)
    assert (ts == ts_after).all()


@pytest.mark.parametrize('df', [
    pd.DataFrame({0: [1, 2, 3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
    pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='S')),
])
def test_pandas_dataframe_handler(df):
    buf = jsonpickle.encode(df)
    ts_after = jsonpickle.decode(buf)
    assert (df == ts_after).all().all()


def test_mixed_python_and_pandas_types():
    data = (
        np.array([1., 2., 3.]),
        pd.TimeSeries([1, 2, 3], index=[0, 1, 2]),
        pd.DataFrame({0: [1, 2, 3], 1: [1.1, 2.2, 3.3]}, index=pd.date_range('1970-01-01', periods=3, freq='S'))
    )
    buf = jsonpickle.encode(data)
    data_after = jsonpickle.decode(buf)

    assert isinstance(data, tuple)
    assert len(data) == 3
    assert (data[0] == data_after[0]).all()
    assert (data[1] == data_after[1]).all()
    assert (data[2] == data_after[2]).all().all()
