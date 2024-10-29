```python
import subprocess
import sys
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional

from jinja2 import Template

from src.databases.load_gs_data import gs_read_auto, gs_store_auto
from src.pipeline.development_pipeline import parse_dataset_type

GLOBAL_IMPORTS = """
from typing import Any, NamedTuple
import kfp.dsl as dsl
from collections import namedtuple
from kfp.dsl import Input, Output, Dataset, Model, component, pipeline
from consts import MLFLOW_IMAGE

PIPELINE_JSON = "{pipeline_json}"
PIPELINE_ROOT = "{pipeline_root}"
PIPELINE_NAME = "{pipeline_name}"

REGION = "us-east4"
VPC_NETWORK = "projects/12856960411/global/networks/vpcnet-private-svc-access-use4"
SERVICE_ACCOUNT = "svc-mx-dl-iaml-ds-hs@wmt-mx-dl-iaml-dev.iam.gserviceaccount.com"
"""

MAIN_BLOCK = """
if __name__ == "__main__":
    # Imports for vertex pipelines
    from google.cloud import aiplatform
    from kfp import compiler

    # aiplatform.Model
    kfp_compiler = compiler.Compiler()
    kfp_compiler.compile(
        pipeline_func=training_pipeline, package_path=PIPELINE_JSON
    )

    pipeline_job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=PIPELINE_JSON,
        location=REGION,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=dict(),
        enable_caching=False,
    )

    pipeline_job.submit(service_account=SERVICE_ACCOUNT, network=VPC_NETWORK)
"""

DECLARATIVE_FUNCTION_TEMPLATE = Template(
    """
{{ decorator }}
def {{ function_name }}({{ parameters|join(', ') }}):
    # Imports
    {{ imports|join('\n')|indent(4) }}
    {{ body|indent(4) }}

"""
)

MINIMAL_FUNCTION_TEMPLATE = Template(
    """
{{ decorator }}
def {{ function_name }}({{ parameters|join(', ') }}):
    {{ body|indent(4) }}

"""
)


def make_input(params: Iterable) -> list[str]:
    return list(map(lambda x: f"{x}: Input[Dataset]", params))


def make_output(params: Iterable) -> list[str]:
    return list(map(lambda x: f"{x}: Output[Dataset]", params))


def create_import_statement(function: Callable) -> str:
    return f"from {function.__module__} import {function.__name__}"


def create_function_code(
    func_name: str,
    params: list[str],
    body: str,
    imports: list[str],
    decorator: Optional[str] = None,
) -> str:

    if decorator is None:
        decorator = ""

    code = DECLARATIVE_FUNCTION_TEMPLATE.render(
        function_name=func_name,
        parameters=params,
        imports=imports,
        body=body,
        decorator=decorator,
    )

    return code


def create_function_call(
    function_name: str, args: list[str], kwargs: dict[str, Any], outputs: list[str]
) -> str:

    function_arguments = args + [f"{k}={repr(v)}" for k, v in kwargs.items()]

    if outputs:
        function_outputs = ", ".join(outputs) + " = "
    else:
        function_outputs = ""

    return f"{function_outputs}{function_name}({', '.join(function_arguments)})"


def create_function_from_step(
    component_name: str,
    function: Callable,
    inputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    kwargs: Optional[dict] = None,
    decorator: Optional[str] = None,
    load_fn: Optional[Callable] = None,
    store_fn: Optional[Callable] = None,
) -> tuple[str, dict]:

    assert function.__name__ != "<lambda>", "Functions can not be anonymous!"

    load_fn = gs_read_auto if load_fn is None else load_fn
    store_fn = gs_store_auto if store_fn is None else store_fn
    inputs = [] if inputs is None else inputs
    outputs = [] if outputs is None else outputs
    kwargs = {} if kwargs is None else kwargs

    imports = []

    if inputs:
        imports.append(create_import_statement(load_fn))

    if outputs:
        imports.append(create_import_statement(store_fn))
        imports.append(create_import_statement(parse_dataset_type))

    imports.append(create_import_statement(function))

    input_params = list(map(lambda x: f"{x}_idst", inputs))
    loading_lines = [
        f"{inpt} = {load_fn.__name__}({param}.path)"
        for param, inpt in zip(input_params, inputs)
    ]

    loading_statement = "\n".join(loading_lines)

    function_call = create_function_call(
        function_name=function.__name__, args=inputs, kwargs=kwargs, outputs=outputs
    )

    output_params = list(map(lambda x: f"{x}_odst", outputs))

    output_parse_lines = [
        f"{outpt}_path, {outpt} = parse_dataset_type({param}.path, {outpt})"
        for outpt, param in zip(outputs, output_params)
    ]

    output_lines = [f"{store_fn.__name__}({outpt}, {outpt}_path)" for outpt in outputs]

    intercalated_output_lines = [
        item for pair in zip(output_parse_lines, output_lines) for item in pair
    ]

    output_statement = "\n".join(intercalated_output_lines)

    body = "\n".join(
        [
            "# Inputs Loading",
            loading_statement,
            "# Function Call",
            function_call,
            "# Outputs Packing",
            output_statement,
        ]
    )

    component_params = make_input(input_params) + make_output(output_params)

    component_code = create_function_code(
        component_name,
        component_params,
        body,
        imports,
        decorator=decorator,
    )

    pipeline_info = {
        "inputs": input_params,
        "outputs": output_params,
        "component_name": component_name,
    }

    print(pipeline_info)

    return component_code, pipeline_info


def parse_names_safe(name: str):
    return name.split("/")[-1].replace("-", "_")


def kfp_from_pipeline(
    pipeline_steps: list[dict],
    load_function: Callable,
    store_function: Callable,
    pipeline_name: str,
    pipeline_root: str,
    pipeline_json: str,
) -> str:

    code = GLOBAL_IMPORTS.format(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        pipeline_json=pipeline_json,
    )

    components_code, compiler_params = create_components_code(
        pipeline_steps, load_function, store_function
    )

    code += components_code

    code += create_compile_function(compiler_params)

    code += MAIN_BLOCK

    return code


def create_components_code(
    pipeline_steps: list[dict], load_function: Callable, store_function: Callable
) -> tuple[str, list[dict]]:
    compiler_params = []
    components_code = []

    for i, step in enumerate(pipeline_steps):
        name = parse_names_safe(step["name"])
        function: Callable = step["function"]
        inputs: list = step.get("inputs", [])
        outputs: list = step.get("outputs", [])
        kwargs: dict = step.get("kwargs", {})

        inputs = list(map(parse_names_safe, inputs))
        outputs = list(map(parse_names_safe, outputs))

        component_name = "_".join(([f"s{i}_{name}", function.__name__]))

        print({"inputs": inputs, "outputs": outputs})

        function_code, component_info = create_function_from_step(
            component_name,
            function,
            inputs,
            outputs,
            kwargs,
            decorator="@component(base_image=MLFLOW_IMAGE)",
            load_fn=load_function,
            store_fn=store_function,
        )

        components_code.append(function_code)
        compiler_params.append(component_info)

    body = "\n".join(components_code)
    return body, compiler_params


def create_compile_function(compiler_info: list[dict]) -> str:

    refs_dict: dict = defaultdict(dict)

    lines = []

    for component in compiler_info:

        component_name: str = component["component_name"]
        n_outputs = len(component["outputs"])
        pseudo_output = component_name.split("_", maxsplit=1)[0] + "_output"

        for var_name in component["outputs"]:

            refs_dict[var_name]["component_name"] = component_name
            refs_dict[var_name]["pseudo_output"] = pseudo_output

            if n_outputs == 1:
                output_cmd = var_name + ".output"
            else:
                output_cmd = f"{pseudo_output}.outputs['{var_name}']"

            refs_dict[var_name]["output_get"] = output_cmd

        outputs = component["outputs"]
        inputs = component["inputs"]
        inputs_args = list(
            map(
                lambda x: refs_dict[x.replace("idst", "odst")]["output_get"],
                inputs,
            )
        )

        inputs_keyword = list(map(lambda x, y: f"{x}={y}", inputs, inputs_args))

        outputs_assignment = outputs if len(outputs) == 1 else [pseudo_output]

        line = create_function_call(
            function_name=component["component_name"],
            args=inputs_keyword,
            kwargs={},
            outputs=outputs_assignment,
        )

        lines.append(line)

        body = "\n".join(lines)

    code = MINIMAL_FUNCTION_TEMPLATE.render(
        function_name="training_pipeline",
        parameters=[],
        body=body,
        decorator="@pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME)",
    )

    return code


def create_kfp_compiler_file(
    training_steps: list[dict],
    file: str,
    pipeline_name: str,
    pipeline_root: str,
    pipeline_json: str,
):

    contents = kfp_from_pipeline(
        training_steps,
        load_function=gs_read_auto,
        store_function=gs_store_auto,
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        pipeline_json=pipeline_json,
    )

    with open(file, mode="w", encoding="utf-8") as f:
        f.write(contents)

    subprocess.run(["uvx", "black", file], check=False)
    subprocess.run(["uvx", "isort", file], check=False)
    subprocess.run(["uvx", "ruff", "check", file, "--fix"], check=False)


if __name__ == "__main__":
    pass

```


```python
def make_input(params: Iterable) -> list[str]:
    """Generates a list of input parameter declarations for Kubeflow Pipelines (KFP) components.

    Parameters
    ----------
    - params : Iterable
        A list of parameter names.

    Returns
    -------
    list[str]
        A list of formatted strings for input parameters, in the format 'param: Input[Dataset]'.
    """
    return list(map(lambda x: f"{x}: Input[Dataset]", params))


def make_output(params: Iterable) -> list[str]:
    """Generates a list of output parameter declarations for KFP components.

    Parameters
    ----------
    - params : Iterable
        A list of parameter names.

    Returns
    -------
    list[str]
        A list of formatted strings for output parameters, in the format 'param: Output[Dataset]'.
    """
    return list(map(lambda x: f"{x}: Output[Dataset]", params))


def create_import_statement(function: Callable) -> str:
    """Creates an import statement for a given function.

    Parameters
    ----------
    - function : Callable
        The function to generate an import statement for.

    Returns
    -------
    str
        Import statement for the specified function.
    """
    return f"from {function.__module__} import {function.__name__}"


def create_function_code(
    func_name: str,
    params: list[str],
    body: str,
    imports: list[str],
    decorator: Optional[str] = None,
) -> str:
    """Generates the code for a function, including its imports, parameters, and body.

    Parameters
    ----------
    - func_name : str
        The name of the function to generate.
    - params : list[str]
        A list of function parameter names.
    - body : str
        The code to include in the function body.
    - imports : list[str]
        A list of imports to include in the function.
    - decorator : Optional[str], optional
        An optional decorator to apply to the function.

    Returns
    -------
    str
        The full code for the function as a formatted string.
    """
    ...


def create_function_call(
    function_name: str, args: list[str], kwargs: dict[str, Any], outputs: list[str]
) -> str:
    """Creates a function call statement with specified arguments and keyword arguments.

    Parameters
    ----------
    - function_name : str
        The name of the function to call.
    - args : list[str]
        A list of positional arguments for the function.
    - kwargs : dict[str, Any]
        A dictionary of keyword arguments for the function.
    - outputs : list[str]
        A list of output variables for capturing function return values.

    Returns
    -------
    str
        The formatted function call as a string.
    """
    ...


def create_function_from_step(
    component_name: str,
    function: Callable,
    inputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    kwargs: Optional[dict] = None,
    decorator: Optional[str] = None,
    load_fn: Optional[Callable] = None,
    store_fn: Optional[Callable] = None,
) -> tuple[str, dict]:
    """Creates the code for a component function from a pipeline step, including inputs and outputs handling.

    Parameters
    ----------
    - component_name : str
        The name of the component function.
    - function : Callable
        The function to wrap as a component.
    - inputs : Optional[list[str]], optional
        List of input parameters.
    - outputs : Optional[list[str]], optional
        List of output parameters.
    - kwargs : Optional[dict], optional
        Dictionary of keyword arguments.
    - decorator : Optional[str], optional
        Optional decorator for the component.
    - load_fn : Optional[Callable], optional
        Function to load input datasets.
    - store_fn : Optional[Callable], optional
        Function to store output datasets.

    Returns
    -------
    tuple[str, dict]
        A tuple containing the component code as a string and a dictionary of pipeline metadata.
    """
    ...


def parse_names_safe(name: str):
    """Safely parses a string to create a valid identifier by removing specific characters.

    Parameters
    ----------
    - name : str
        The string to parse.

    Returns
    -------
    str
        A formatted string safe for use as a variable name.
    """
    ...


def kfp_from_pipeline(
    pipeline_steps: list[dict],
    load_function: Callable,
    store_function: Callable,
    pipeline_name: str,
    pipeline_root: str,
    pipeline_json: str,
) -> str:
    """Generates the Kubeflow Pipeline (KFP) code for a sequence of pipeline steps.

    Parameters
    ----------
    - pipeline_steps : list[dict]
        List of dictionaries describing each step in the pipeline.
    - load_function : Callable
        Function to load data for each step.
    - store_function : Callable
        Function to store data from each step.
    - pipeline_name : str
        Name of the pipeline.
    - pipeline_root : str
        Root path for pipeline artifacts.
    - pipeline_json : str
        Path to save the compiled pipeline JSON.

    Returns
    -------
    str
        Full generated pipeline code as a string.
    """
    ...


def create_components_code(
    pipeline_steps: list[dict], load_function: Callable, store_function: Callable
) -> tuple[str, list[dict]]:
    """Creates the code for each component in the pipeline.

    Parameters
    ----------
    - pipeline_steps : list[dict]
        A list of steps, where each step is a dictionary with function details.
    - load_function : Callable
        Function to load input data.
    - store_function : Callable
        Function to save output data.

    Returns
    -------
    tuple[str, list[dict]]
        A tuple containing the component code as a string and a list of parameters for the compiler.
    """
    ...


def create_compile_function(compiler_info: list[dict]) -> str:
    """Generates the compilation function for the pipeline with specified compiler information.

    Parameters
    ----------
    - compiler_info : list[dict]
        A list of dictionaries with information for each pipeline component.

    Returns
    -------
    str
        The generated compilation function code.
    """
    ...


def create_kfp_compiler_file(
    training_steps: list[dict],
    file: str,
    pipeline_name: str,
    pipeline_root: str,
    pipeline_json: str,
):
    """Creates a file that compiles the Kubeflow Pipeline (KFP) with the given training steps.

    Parameters
    ----------
    - training_steps : list[dict]
        List of dictionaries specifying each step in the training pipeline.
    - file : str
        Path to the output file to save the compiled pipeline code.
    - pipeline_name : str
        Name of the pipeline.
    - pipeline_root : str
        Root path for pipeline artifacts.
    - pipeline_json : str
        Path to save the compiled pipeline JSON.
    """
    ...

```
