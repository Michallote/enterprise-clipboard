Certainly! Here is the updated implementation of your configuration management module, along with a comprehensive `README.md` file explaining how to use the library.

---

```python
# config.py

import os
from functools import wraps
from typing import Any, Dict, Optional

import toml
from dotenv import load_dotenv

# Load .env variables if .env file exists
load_dotenv()

# Define obligatory and optional variables
OBLIGATORY_VARS = {'API_KEY', 'API_SECRET'}
OPTIONAL_VARS = {'OPTIONAL_SETTING'}

# Global configuration dictionary
_global_config: Dict[str, Any] = {}

# Load configurations from pyproject.toml if it exists
def _load_pyproject_config():
    try:
        with open('pyproject.toml', 'r') as f:
            config = toml.load(f)
            return config.get('tool', {}).get('your_package_name', {})
    except FileNotFoundError:
        return {}

# Load configurations from environment variables
def _load_env_config():
    return {key: os.getenv(key) for key in OBLIGATORY_VARS.union(OPTIONAL_VARS)}

# Initialize global configuration
def _initialize_global_config():
    global _global_config
    _global_config.update(_load_pyproject_config())
    _global_config.update(_load_env_config())

_initialize_global_config()

def set_config(config: Optional[dict] = None, **kwargs):
    """Set global configuration variables."""
    if config:
        _global_config.update(config)
    if kwargs:
        _global_config.update(kwargs)

def access_env_variable(key: str):
    """Indicate that the configuration variable should be retrieved from SecretsManager."""
    _global_config[key] = "auto"

def _load_secret_from_manager(key: str) -> Any:
    """Retrieve the secret value for the given key from SecretsManager."""
    try:
        from src.credentials.secret_manager import SecretsManager
        secrets_manager = SecretsManager()
        return secrets_manager.get_secret(key)
    except ImportError:
        raise ImportError("SecretsManager is not available.")
    except Exception as e:
        raise Exception(f"Error retrieving '{key}' from SecretsManager: {e}")

def get_config(key: str) -> Any:
    """Get a configuration variable. Raise an exception if an obligatory variable is missing."""
    value = _global_config.get(key)
    if value == "auto":
        value = _load_secret_from_manager(key)
        _global_config[key] = value  # Cache the value
    if value is not None:
        return value
    elif key in OBLIGATORY_VARS:
        raise ValueError(
            f"Obligatory configuration '{key}' is not set. Please set it via pyproject.toml, "
            ".env file, environment variable, or set_config() method."
        )
    elif key in OPTIONAL_VARS:
        raise ValueError(
            f"Optional configuration '{key}' is not set. This may affect certain functionalities."
        )
    else:
        raise KeyError(f"Configuration key '{key}' is not recognized.")

def ephemeral_config(config: Optional[dict] = None, **kwargs):
    """
    Context manager for ephemeral configuration settings.
    Usage:
        with ephemeral_config({'API_KEY': 'temp_key'}):
            # Code that uses the temporary API_KEY
    """
    class EphemeralConfig:
        def __enter__(self):
            self.original_config = _global_config.copy()
            if config:
                _global_config.update(config)
            if kwargs:
                _global_config.update(kwargs)
    
        def __exit__(self, exc_type, exc_val, exc_tb):
            global _global_config
            _global_config = self.original_config
    
    return EphemeralConfig()

def requires_optional_vars(func):
    """Decorator to ensure optional variables are set before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        missing_vars = [var for var in OPTIONAL_VARS if not _global_config.get(var)]
        if missing_vars:
            raise ValueError(
                f"The following optional configurations are missing: {', '.join(missing_vars)}. "
                "Please set them via pyproject.toml, .env file, environment variable, or set_config() method."
            )
        return func(*args, **kwargs)
    return wrapper
```

---

## README.md

# Your Library Name

A Python library that provides a flexible and composable way to manage configuration variables for your modules. It supports multiple configuration methods, handles obligatory and optional variables appropriately, and integrates with your existing `SecretsManager` class.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
  - [Obligatory Variables](#obligatory-variables)
  - [Optional Variables](#optional-variables)
- [Configuration Methods](#configuration-methods)
  - [Pyproject.toml](#pyprojecttoml)
  - [.env File](#env-file)
  - [Environment Variables](#environment-variables)
  - [Global Method Call (`set_config`)](#global-method-call-set_config)
  - [Ephemeral Configuration](#ephemeral-configuration)
  - [Automatic Retrieval from SecretsManager](#automatic-retrieval-from-secretsmanager)
- [Accessing Configuration Variables](#accessing-configuration-variables)
- [Using Optional Variables](#using-optional-variables)
- [Configuration Precedence](#configuration-precedence)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [License](#license)

## Installation

Install the library using pip:

```bash
pip install your-library-name
```

## Configuration

### Obligatory Variables

The following variables are obligatory and must be set before using the library:

- `API_KEY`
- `API_SECRET`

If these variables are not set, an exception will be raised with instructions on how to set them.

### Optional Variables

The following variables are optional and are required only for certain functionalities:

- `OPTIONAL_SETTING`

If these variables are not set when required, an exception will be raised when you attempt to use a feature that depends on them.

## Configuration Methods

The library supports multiple methods for configuring variables. You can use any combination of these methods.

### Pyproject.toml

You can set configuration variables in your `pyproject.toml` file under the `[tool.your_package_name]` section.

Example:

```toml
[tool.your_package_name]
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
OPTIONAL_SETTING = "some_value"
```

### .env File

You can set configuration variables in a `.env` file in your project directory.

Example:

```
API_KEY=your_api_key
API_SECRET=your_api_secret
OPTIONAL_SETTING=some_value
```

### Environment Variables

You can set environment variables in your operating system.

Example (Linux/Mac):

```bash
export API_KEY=your_api_key
export API_SECRET=your_api_secret
```

Example (Windows Command Prompt):

```cmd
set API_KEY=your_api_key
set API_SECRET=your_api_secret
```

### Global Method Call (`set_config`)

You can set configuration variables programmatically using the `set_config` method.

Example:

```python
from your_package_name.config import set_config

set_config(API_KEY='your_api_key', API_SECRET='your_api_secret')

# Or pass a dictionary
config_dict = {
    'API_KEY': 'your_api_key',
    'API_SECRET': 'your_api_secret'
}
set_config(config=config_dict)
```

### Ephemeral Configuration

You can set configuration variables temporarily within a context manager using the `ephemeral_config` method.

Example:

```python
from your_package_name.config import ephemeral_config

with ephemeral_config(API_KEY='temporary_key'):
    # Code that uses the temporary API_KEY
    pass
# After the block, the original configuration is restored
```

You can also pass a dictionary:

```python
temp_config = {'API_KEY': 'temporary_key'}
with ephemeral_config(config=temp_config):
    # Code that uses the temporary API_KEY
    pass
```

### Automatic Retrieval from SecretsManager

If you have sensitive configuration variables that you want to retrieve from your `SecretsManager`, you can use the `access_env_variable` method to indicate that the variable should be retrieved automatically.

Example:

```python
from your_package_name.config import access_env_variable

# Indicate that API_KEY should be retrieved from SecretsManager
access_env_variable('API_KEY')
```

When you access the variable using `get_config`, it will be retrieved from the `SecretsManager` and cached.

**Note:** This requires that `src.credentials.secret_manager` is available and has a `SecretsManager` class with a `get_secret(key)` method.

## Accessing Configuration Variables

To access a configuration variable, use the `get_config` method.

Example:

```python
from your_package_name.config import get_config

api_key = get_config('API_KEY')
```

If the variable is not set and is obligatory, a `ValueError` will be raised with instructions on how to set it.

## Using Optional Variables

If you have functions that require optional variables, you can use the `@requires_optional_vars` decorator to ensure they are set before the function is executed.

Example:

```python
from your_package_name.config import get_config, requires_optional_vars

@requires_optional_vars
def optional_feature():
    optional_setting = get_config('OPTIONAL_SETTING')
    # Function logic
```

If the optional variables are not set, a `ValueError` will be raised when the function is called.

## Configuration Precedence

Configuration variables can be set using multiple methods. The precedence, from highest to lowest, is as follows:

1. **Ephemeral Configuration (`ephemeral_config`)**
2. **Global Method Calls (`set_config`)**
3. **Environment Variables and `.env` File**
4. **`pyproject.toml`**
5. **SecretsManager via `access_env_variable`**

Variables set via higher precedence methods will override those set via lower precedence methods.

## Error Handling

- **Obligatory Variables Not Set**: Raises a `ValueError` with instructions on how to set the variable.
- **Optional Variables Not Set When Required**: Raises a `ValueError` when a function decorated with `@requires_optional_vars` is called without the necessary optional configurations.
- **Unknown Configuration Key**: Raises a `KeyError` if you try to access an unrecognized configuration key.
- **SecretsManager Not Available**: Raises an `ImportError` if `SecretsManager` is not available when attempting to retrieve a variable set to `auto`.

## Examples

### Example 1: Basic Usage

```python
from your_package_name.config import get_config, set_config

# Set configurations
set_config(API_KEY='your_api_key', API_SECRET='your_api_secret')

# Access configurations
api_key = get_config('API_KEY')
api_secret = get_config('API_SECRET')
```

### Example 2: Using Ephemeral Configuration

```python
from your_package_name.config import get_config, ephemeral_config

# Set global configuration
set_config(API_KEY='global_api_key')

print(get_config('API_KEY'))  # Output: global_api_key

with ephemeral_config(API_KEY='temporary_key'):
    print(get_config('API_KEY'))  # Output: temporary_key

print(get_config('API_KEY'))  # Output: global_api_key
```

### Example 3: Retrieving Variables from SecretsManager

```python
from your_package_name.config import get_config, access_env_variable

# Indicate that API_SECRET should be retrieved from SecretsManager
access_env_variable('API_SECRET')

# When you access API_SECRET, it will be retrieved from SecretsManager
api_secret = get_config('API_SECRET')
```

### Example 4: Function that Requires Optional Variables

```python
from your_package_name.config import get_config, requires_optional_vars

@requires_optional_vars
def perform_optional_task():
    optional_setting = get_config('OPTIONAL_SETTING')
    # Use optional_setting in your function

# Set the optional setting
set_config(OPTIONAL_SETTING='some_value')

# Now you can call the function
perform_optional_task()
```

## License

[MIT License](LICENSE)

---

This `README.md` provides users with detailed instructions on how to configure and use your library, along with examples. It explains how to set obligatory and optional variables, the different methods of configuration, how to retrieve variables from the `SecretsManager`, and how to handle errors.

### Notes on the Implementation

1. **`set_config` Function**: Modified to accept a `config` dictionary as an argument, allowing users to pass configurations as a dictionary or keyword arguments.

    ```python
    def set_config(config: Optional[dict] = None, **kwargs):
        """Set global configuration variables."""
        if config:
            _global_config.update(config)
        if kwargs:
            _global_config.update(kwargs)
    ```

2. **`access_env_variable` Method**: Implemented to allow users to specify which variables should be retrieved from the `SecretsManager`.

    ```python
    def access_env_variable(key: str):
        """Indicate that the configuration variable should be retrieved from SecretsManager."""
        _global_config[key] = "auto"
    ```

3. **SecretsManager Integration**: The `_load_secret_from_manager` function is called within `get_config` only for variables that have been set to `"auto"` using `access_env_variable`. This ensures that secrets are retrieved from the `SecretsManager` only when explicitly requested by the user.

    ```python
    def get_config(key: str) -> Any:
        """Get a configuration variable. Raise an exception if an obligatory variable is missing."""
        value = _global_config.get(key)
        if value == "auto":
            value = _load_secret_from_manager(key)
            _global_config[key] = value  # Cache the value
        # Rest of the function...
    ```

4. **Error Handling**: Improved error messages and exception handling, especially when interacting with the `SecretsManager`.

    ```python
    except Exception as e:
        raise Exception(f"Error retrieving '{key}' from SecretsManager: {e}")
    ```

### Instructions for Users

- **Setting Obligatory Variables**: You must set `API_KEY` and `API_SECRET` using one of the supported methods before using the library.
- **Using SecretsManager**: To retrieve sensitive information from the `SecretsManager`, use the `access_env_variable` method to specify which variables should be fetched automatically.
- **Configuration Precedence**: Be aware of the configuration precedence to understand which values will be used when multiple configuration methods are employed.

### Conclusion

This implementation provides a flexible and composable configuration management system for your library. It allows users to configure variables using various methods, handles obligatory and optional variables appropriately, and provides clear documentation and examples through the `README.md` file.

Feel free to adjust the variable names, package names, and paths to match your specific project structure.
