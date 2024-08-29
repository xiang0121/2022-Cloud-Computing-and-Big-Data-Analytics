## Dataset
- Unzip 311511056.zip to `./311511056`
    ```sh
    unzip 311511056.zip -d ./311511056
    ```
- Folder structure
    ```
    .
    ├── eval.py
    ├── hw1.py
    ├── Readme.md
    └── requirements.txt
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python hw1.py
```

## Make Prediction
```sh
python eval.py
```
The prediction file is `submission.csv`.