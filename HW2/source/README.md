## Dataset
- Unzip 311511056.zip to `./311511056`
    ```sh
    unzip 311511056.zip -d ./311511056
    ```
- Folder structure
    ```
    .
    ├── data
    │   ├── test/
    │   └── unlabeled/
    ├── 311511056     
    │   ├── embed.py
    │   └── hw2.py
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

## Make Embedding
```sh
python embed.py
```
The embedding file is `311511056.npy`.