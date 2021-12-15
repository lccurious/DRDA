# A Machine Learning Template Project

This is a machine learning template project for generic training, validation and testing.

## Project Structure

```shell
machine-learning-template
├─data
├─experiments
├─lib
├─notebooks
├─tests
└─tools
```

- Project is mainly considered to be structured, where the core functional about config, models, core, loss etc. are stored in `example_lib` directory.

- And the `tests` directory are mainly consists of some test case of functional components. Generate the random inputs and test if there is any bug.

- The `notebooks` directory consists of some visulization results and early developement of some functional modules.

- The `experiments` directory consists of custom defined `.yaml` file for ablation studies.

- The `tools` directory consists of custom training and testing scripts.

- The `data` directory consists of datasets for training, some subdirectory can be created by symbol link: `ln -s /path/to/target/dataset dataset_name`


## IDE Tips

If use VS Code, put following snippets in your `.vscode/setting.json`:

```json
{
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/*"
    ],
    "python.analysis.extraPaths": [
        "${workspaceFolder}/*"
    ]
}
```

If use Pycharm, just mark related directories as source folders.


## Create Requirements

Install the `pipreqs`, and use this tools to scan your project, and generate the `requirements.txt`.
