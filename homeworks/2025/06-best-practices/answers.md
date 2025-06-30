# Answers

1. if __name__ == "__main__"
2. __init__.py
3. 2
4. --endpoint-url
5. 3620
6. 36.28


# Solution

4.
```bash
uv run aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration
```
5.
```bash
aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/
```
6.
```bash
uv run src/batch.py --year 2023 --month 1
```
