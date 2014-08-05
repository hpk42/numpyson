0.4
--------------

- fix bug: respect ordering when serializing/deserializing ndarrays


0.3
--------------

- depend on pandas>=0.13.1 because we are using Float64Index which
  does not exist in pandas-0.12

- support numpy.float64 and numpy.int64
