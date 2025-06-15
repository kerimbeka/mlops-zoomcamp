# Answers

1. 6.24
2. 66M
3. jupyter nbconvert --to script starter.ipynb
4. "sha256:0650e730afb87402baa88afbf31c07b84c98272622aaba002559b614600ca691"
5. 14.29
6. 0.19

# Solution

1. ```{bash}

uv run src/main.py --year 2023 --month 3

```
5. ```{bash}

uv run src/main.py --year 2023 --month 4

```
6. ```{bash}

docker build -t prediction_model .

docker run prediction_model --year 2023 --month 3

```

