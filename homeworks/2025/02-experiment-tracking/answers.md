# Answers

1. 2.22.0
2. 4
3. 2
4. -default-artifact-root
5. 5.335
6. 5.567

# Solution

1.

```{bash}
uv add mlflow
uv run mlflow --version
```

2.

```{bash}
uv run src/preprocess_data.py --raw_data_path ./data --dest_path ./output
```

3.

```{bash}
uv run src/train.py
```

4.

```{bash}
uv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts
```

5.

```{bash}
uv run src/hpo.py
```

6.

```{bash}
uv run src/register_model.py 
```
