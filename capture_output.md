# Naming Conventions

Below are some **naming suggestions** (and the rationale behind them) for the **function** that creates these tracked (“deferred” or “lazy”) objects. Since you already have an `execute_pipeline` and a pipeline model, it’s most consistent if the name:

1. **Implies** we’re capturing a function + parameters for later use (rather than immediately computing).
2. **Fits** naturally into an existing pipeline codebase: “execute_pipeline,” “compile_pipeline,” etc.
3. **Is** short and memorable, because it will be called frequently, e.g., `start_date = ???(...)`.

Below are several **patterns** and **candidate names** you could choose from:

---

## 1) Names Highlighting “Capture” or “Defer”

- **`defer(...)`**  
  - Conveys that we’re *not* computing the value now; we’re deferring until pipeline execution or code-generation time.  
  - Example usage:  
    ```python
    start_date = defer(func=get_current_date, params={}, return_type=str)
    pipeline_steps = [{"func": some_step, "kwargs": {"date": start_date}}]
    ```

- **`capture(...)`**  
  - Conveys that we’re “capturing” the function and its arguments for later replay.  
  - Example usage:  
    ```python
    start_date = capture(get_current_date, foo_type=str)
    ```

- **`capture_value(...)`** or **`capture_output(...)`**  
  - A bit more verbose, but very explicit about capturing the *output* from `get_current_date`.  

---

## 2) Names Highlighting “Bind” or “Parameter”

- **`bind(...)`**  
  - Suggests we’re “binding” a function to parameters in some sort of partial application.  
  - Example usage:
    ```python
    start_date = bind(get_current_date, foo_type=str)
    ```

- **`parameter(...)`** / **`pipeline_parameter(...)`**  
  - If in your pipeline domain, these “tracked values” act like *parameters* (read once at compile time, or updated at runtime), calling them “parameters” might map well conceptually.  
  - Example usage:
    ```python
    start_date = parameter(get_current_date, foo_type=str)
    pipeline_steps.append({"name": "raw", "func": execute_query, "kwargs": {"start_date": start_date}})
    ```

- **`make_parameter(...)`** or **`create_parameter(...)`**  
  - A more verbose version of `parameter(...)`.  

---

## 3) Names Highlighting “Tracked” or “Tagged”

- **`track(...)`** or **`tracked(...)`**  
  - If your library docs refer to these as “tracked objects,” you might keep consistent with a name like `track(func=..., params=..., return_type=...)`.  
  - Example usage:
    ```python
    start_date = track(get_current_date, foo_type=str)
    ```

- **`tag_value(...)`**  
  - If your library uses “tagging” as a metaphor for attaching metadata, this could work. But “track” is usually clearer.  

---

## 4) Short vs. Descriptive

- If your team/pipeline system is comfortable with short, direct names, something like `capture(...)` or `defer(...)` is neat.  
- If you prefer explicit clarity, you might choose `create_pipeline_parameter(...)` or `create_tracked_value(...)`.

---

## 5) Typical Patterns in Other Tools

- **Dagster** uses a concept of “op” or “resource” or “IO manager.”  
- **Prefect** calls them “tasks” and “Parameter().”  
- **Airflow** is often “BashOperator(...), PythonOperator(...), etc.”  
- **Kubeflow Pipelines** typically references `@component` functions.

You could mirror these naming conventions or keep something more domain-specific. 

---

## Recommendation

- Pick **one** short verb that resonates with your code’s style. 
- For user clarity, you might name the function `capture(...)` or `defer(...)`, and in the docstring say something like:

  > *Capture* a function and its parameters for lazy evaluation within the pipeline. Returns a special object that acts like the original data type but retains its function metadata.

Example final usage:

```python
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

start_date = capture(get_current_date, foo_type=str)

pipeline_steps = [
    {"name": "raw", "func": execute_query, "kwargs": {"file": "queries/segmentation.sql", "start_date": start_date}}
]
```

This reads quite cleanly, keeps the code succinct, and signals the intention: “this is not a normal function call; we’re capturing for later.”

# Implementation

```python
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# We'll define a union for the six "primitives":
TrackedPrimitive = Union[dict, str, float, list, tuple, bool]

# We define a TypeVar constrained to the 6 types
T = TypeVar("T", dict, str, float, list, bool, tuple)

#
# 1) A Mixin that stores creation metadata
#


class BoundBase:
    """
    A mixin that stores metadata about how this object was created:
    the function used (creation_func) and the parameters passed (params).
    """

    def __init__(self, creation_func: Callable[..., Any], params: Dict[str, Any]):
        self.creation_func = creation_func
        self.params = params

    def __repr__(self) -> str:
        """Show both the normal representation plus creation metadata."""
        return (
            f"<{self.__class__.__name__} value={super().__repr__()} "
            f"creation_func={self.creation_func.__name__} params={self.params}>"
        )

    def function_definition(self) -> str:
        """Returns the function call that originated the base value

        Returns
        -------
        str
            function definition
        """

        params_definition = str(self.params)[1:-1].replace(":", "=")
        function_definition = f"{self.creation_func.__name__}({params_definition})"
        return function_definition


#
# 2) IMMUTABLE types: define __new__ + a dummy __init__
#


class BoundStr(str, BoundBase):
    def __new__(
        cls, value: str, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        # 1) create the real string
        obj = super().__new__(cls, value)
        # 2) then attach the metadata
        BoundBase.__init__(obj, creation_func, params)
        return obj

    def __init__(
        self, value: str, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        """
        Dummy init. The real initialization logic is in __new__.
        We define this so Python doesn't try to forward arguments
        to str.__init__ or cause confusion in the MRO.
        """
        pass


class BoundFloat(float, BoundBase):
    def __new__(
        cls, value: float, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        obj = super().__new__(cls, value)
        BoundBase.__init__(obj, creation_func, params)
        return obj

    def __init__(
        self, value: float, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        pass


class BoundInt(int, BoundBase):
    def __new__(
        cls, value: int, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        obj = super().__new__(cls, value)
        BoundBase.__init__(obj, creation_func, params)
        return obj

    def __init__(
        self, value: int, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        pass


class BoundTuple(tuple, BoundBase):
    def __new__(
        cls, iterable, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        obj = super().__new__(cls, iterable)
        BoundBase.__init__(obj, creation_func, params)
        return obj

    def __init__(
        self, iterable, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        pass


#
# 3) MUTABLE types: we can do everything in __init__
#


class BoundDict(dict, BoundBase):
    def __init__(
        self, *args, creation_func: Callable[..., Any], params: Dict[str, Any], **kwargs
    ):
        dict.__init__(self, *args, **kwargs)
        BoundBase.__init__(self, creation_func, params)


class BoundList(list, BoundBase):
    def __init__(
        self, *args, creation_func: Callable[..., Any], params: Dict[str, Any]
    ):
        list.__init__(self, *args)
        BoundBase.__init__(self, creation_func, params)


#
# 4) BOOL stand-in: can't subclass bool, so we wrap it.
#


# class TrackedBool(TrackedBase):
#     """
#     Since bool cannot be subclassed, we make a wrapper that
#     behaves like a bool but isn't a subclass.
#     """

#     def __init__(
#         self, value: bool, creation_func: Callable[..., Any], params: Dict[str, Any]
#     ):
#         super().__init__(creation_func, params)
#         self._value = bool(value)

#     def __bool__(self) -> bool:
#         return self._value

#     def __eq__(self, other: Any) -> bool:
#         return self._value == other

#     def __repr__(self) -> str:
#         return self._value.__repr__()

#     def __str__(self) -> str:
#         return self._value.__str__()


#
# 5) A function to dispatch to the correct "tracked" type
#


def _wrap_value(
    value: T,
    creation_func: Callable[..., Any],
    params: Dict[str, Any],
    declared_type: Optional[type] = None,
) -> T:
    """
    Given a plain value and metadata, wrap it in the appropriate tracked subclass.
    If declared_type is given, prefer that. Otherwise dispatch by type(value).
    """
    # If user provided foo_type, we can trust or verify it:
    if declared_type is not None:
        if declared_type is not type(value):
            function_definition = (
                f"{creation_func.__name__}({str(params)[1:-1].replace(':','=')})"
            )
            raise TypeError(
                f"Declared type does not match output of {function_definition}"
            )
    final_type = declared_type or type(value)

    method_map = {
        str: BoundStr,
        dict: BoundDict,
        float: BoundFloat,
        int: BoundInt,
        list: BoundList,
        tuple: BoundTuple,
        # bool: TrackedBool,
    }

    if final_type not in method_map:
        raise TypeError(f"Type {final_type!r} is not supported by the tracking system.")

    track_subtype = method_map[final_type]

    return track_subtype(value, creation_func=creation_func, params=params)  # type: ignore


def capture(
    creation_func: Callable[..., T],
    params: Optional[Dict[str, Any]] = None,
    declared_type: Optional[type] = None,
) -> T:
    """Capture a function and its parameters for defered evaluation within the pipeline. Returns a special object that acts like the original data type but retains its function metadata.

    A factory function that calls `creation_func(**params)`,
    then wraps the result in the appropriate 'Tracked' class.

    Parameters
    ----------
    creation_func : Callable[..., T]
        _description_
    params : Dict[str, Any]
        _description_
    declared_type : Optional[type], optional
        _description_, by default None

    Returns
    -------
    T
        _description_
    """
    params = params or {}
    # 1) call the user’s function
    value = creation_func(**params)

    # 2) wrap in the correct tracked object
    tracked_value = _wrap_value(value, creation_func, params, declared_type)
    return tracked_value


### Testing

from datetime import datetime


def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_bool() -> bool:
    return True


def get_float() -> float:
    return 3.145


def get_int() -> int:
    return 3


def get_list() -> list[int]:
    return [1, 2, 43, 4]


def get_tuple() -> tuple:
    return (1, 2, 3, 5)


tracked_str = capture(creation_func=get_current_date, params={}, declared_type=str)
# tracked_bool = capture(creation_func=get_bool, params={}, declared_type=bool)
tracked_float = capture(creation_func=get_float, params={}, declared_type=float)
tracked_int = capture(creation_func=get_int, params={}, declared_type=int)
tracked_list = capture(creation_func=get_list, params={}, declared_type=list)
tracked_tuple = capture(creation_func=get_tuple, params={}, declared_type=tuple)

print(tracked_str)

assert isinstance(tracked_str, str)
assert tracked_str == get_current_date()

```

# Testing files

```python
import copy
import json
from datetime import datetime

import pytest

from obj_tracker_2 import (  # adjust to actual module name
    BoundDict,
    BoundFloat,
    BoundInt,
    BoundList,
    BoundStr,
    BoundTuple,
    capture,
)


# -- Test functions --
def get_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_float() -> float:
    return 3.145


def get_int() -> int:
    return 42


def get_list() -> list[int]:
    return [1, 2, 3]


def get_tuple() -> tuple:
    return (1, 2, 3)


def get_dict() -> dict:
    return {"a": 1, "b": 2}


# -- Parametrized type check --
@pytest.mark.parametrize(
    "func, declared_type, base_type",
    [
        (get_str, str, BoundStr),
        (get_float, float, BoundFloat),
        (get_int, int, BoundInt),
        (get_list, list, BoundList),
        (get_tuple, tuple, BoundTuple),
        (get_dict, dict, BoundDict),
    ],
)
def test_type_and_value_equivalence(func, declared_type, base_type):
    captured = capture(func, declared_type=declared_type)
    raw = func()

    assert isinstance(captured, declared_type)
    assert isinstance(captured, base_type)
    assert captured == raw


def test_f_string_behavior():
    value = capture(get_str, declared_type=str)
    formatted = f"Date: {value}"
    assert formatted == f"Date: {value}"


def test_json_serialization():
    val = capture(get_dict, declared_type=dict)
    s = json.dumps(val)
    assert isinstance(s, str)
    assert json.loads(s) == get_dict()


def test_copy_and_deepcopy():
    val = capture(get_list, declared_type=list)
    shallow = copy.copy(val)
    deep = copy.deepcopy(val)

    assert shallow == val
    assert deep == val
    assert isinstance(shallow, list)
    assert isinstance(deep, list)


def test_dict_unpacking():
    val = capture(get_dict, declared_type=dict)
    new_dict = {**val}
    assert new_dict == get_dict()


def test_list_indexing_and_iteration():
    val = capture(get_list, declared_type=list)
    assert val[0] == 1
    assert [x for x in val] == get_list()


def test_tuple_unpacking():
    val = capture(get_tuple, declared_type=tuple)
    a, b, c = val
    assert (a, b, c) == (1, 2, 3)


def test_use_as_function_argument():
    def expect_str(x: str):
        assert isinstance(x, str)
        assert x == get_str()

    val = capture(get_str, declared_type=str)
    expect_str(val)


# -- Error handling --
def test_type_mismatch():
    with pytest.raises(TypeError):
        capture(get_int, declared_type=str)


def test_unsupported_type():
    def get_set():
        return {1, 2, 3}

    with pytest.raises(TypeError):
        capture(get_set, declared_type=set)

```

# Issue Add tracked value system (Bound objects) to support deferred pipeline parameters

```markdown
### Summary

Introduce a `Bound` object system to support deferred/lazy evaluation of pipeline parameters. This allows pipeline inputs (e.g., `start_date`) to retain knowledge of how they were generated (function + parameters), enabling transparent execution in both local and compiled modes (e.g., Kubeflow Pipelines).

### Motivation

In current pipelines, early evaluation of inputs (e.g., `start_date = datetime.now().strftime(...)`) loses information about how the value was generated. When compiling the pipeline (e.g., to Kubeflow), it's no longer possible to rehydrate or recompile the original function call.

We need a system that:
- Retains original function + arguments,
- Behaves **identically** to native Python types (`str`, `float`, etc.),
- Supports f-strings, serialization, and general usage without surprises,
- Is type-safe and transparent to users.

### Proposed Feature

Introduce `Bound<T>` objects (e.g., `BoundStr`, `BoundInt`, `BoundDict`, etc.) that:
- Are drop-in replacements for base Python types,
- Store creation metadata (`creation_func`, `params`),
- Provide `function_definition()` to reconstruct origin call,
- Are created using a `capture(...)` factory function.

### Affected Areas

- Pipeline construction logic
- Future pipeline compiler integrations (e.g., `to_kfp`)

### Related Features

- `execute_pipeline(...)`
- `compile_pipeline(...)`

### Tasks

- [x] Define `BoundBase` and tracked subclasses
- [x] Create `capture()` factory
- [x] Add type validation and dispatching
- [x] Build test suite ensuring behavioral transparency
- [ ] Add documentation/examples

```

PR: feat: Add Bound objects for deferred pipeline parameters

```markdown
### Summary

This PR introduces a new system of `Bound` objects to track how pipeline inputs were generated, while behaving like standard Python primitives. This supports deferred evaluation of parameters and future compiler output (e.g., generating code for Kubeflow Pipelines).

### Highlights

- `BoundBase` class stores `creation_func` and `params`
- Tracked subclasses:
  - `BoundStr`
  - `BoundFloat`
  - `BoundInt`
  - `BoundList`
  - `BoundTuple`
  - `BoundDict`
- `capture()` function: wraps a value in its tracked type
- Strict type checking to ensure declared type matches function output
- Booleans intentionally excluded due to Python identity constraints (`is True` fails)
- Full `pytest` test suite to ensure:
  - `isinstance` matches original types
  - `==` behaves as expected
  - f-strings, json, copy, unpacking, and function arg compatibility all pass

### Example Usage

```python
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

start_date = capture(get_current_date, declared_type=str)

pipeline_steps = [
    {"name": "raw", "func": execute_query, "kwargs": {"start_date": start_date}}
]

```
