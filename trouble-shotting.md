## Known Issues

1. ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject

```bash
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Traceback (most recent call last):
  File "/home/xin/Documents/projects/TensorCrypt/src/aes/ecb_cipher.py", line 14, in <module>
    import tensorflow as tf
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/tensorflow/__init__.py", line 468, in <module>
    importlib.import_module("keras.src.optimizers")
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/__init__.py", line 2, in <module>
    from keras.api import DTypePolicy
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/api/__init__.py", line 8, in <module>
    from keras.api import activations
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/api/activations/__init__.py", line 7, in <module>
    from keras.src.activations import deserialize
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/__init__.py", line 1, in <module>
    from keras.src import activations
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/activations/__init__.py", line 33, in <module>
    from keras.src.saving import object_registration
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/saving/__init__.py", line 7, in <module>
    from keras.src.saving.saving_api import load_model
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/saving/saving_api.py", line 7, in <module>
    from keras.src.legacy.saving import legacy_h5_format
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/legacy/saving/legacy_h5_format.py", line 13, in <module>
    from keras.src.legacy.saving import saving_utils
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/legacy/saving/saving_utils.py", line 10, in <module>
    from keras.src import models
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/models/__init__.py", line 1, in <module>
    from keras.src.models.functional import Functional
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/models/functional.py", line 16, in <module>
    from keras.src.models.model import Model
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/models/model.py", line 12, in <module>
    from keras.src.trainers import trainer as base_trainer
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/trainers/trainer.py", line 14, in <module>
    from keras.src.trainers.data_adapters import data_adapter_utils
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/trainers/data_adapters/__init__.py", line 4, in <module>
    from keras.src.trainers.data_adapters import array_data_adapter
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/trainers/data_adapters/array_data_adapter.py", line 7, in <module>
    from keras.src.trainers.data_adapters import array_slicing
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/keras/src/trainers/data_adapters/array_slicing.py", line 12, in <module>
    import pandas
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/__init__.py", line 22, in <module>
    from pandas.compat import is_numpy_dev as _is_numpy_dev
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/compat/__init__.py", line 15, in <module>
    from pandas.compat.numpy import (
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/compat/numpy/__init__.py", line 4, in <module>
    from pandas.util.version import Version
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/util/__init__.py", line 1, in <module>
    from pandas.util._decorators import (  # noqa:F401
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/util/_decorators.py", line 14, in <module>
    from pandas._libs.properties import cache_readonly  # noqa:F401
  File "/home/xin/miniconda3/envs/tensorcrypt/lib/python3.10/site-packages/pandas/_libs/__init__.py", line 13, in <module>
    from pandas._libs.interval import Interval
  File "pandas/_libs/interval.pyx", line 1, in init pandas._libs.interval
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

Fixing by downgrade numpy version:
```
numpy==1.22.3
```