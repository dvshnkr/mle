# sources
# https://technowhisp.com/kaggle-api-python-documentation/
# https://github.com/Kaggle/kaggle-api

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

_ = api.dataset_list(search="demographics", license_name="cc", file_type="csv")

_ = api.dataset_list_files("avenn98/world-of-warcraft-demographics").files

# Download all files of a dataset
# Signature: dataset_download_files(
#     dataset,
#     path=None,
#     force=False,
#     quiet=True,
#     unzip=False
# )
# api.dataset_download_files('avenn98/world-of-warcraft-demographics')

# download single file
# Signature: dataset_download_file(
#     dataset,
#     file_name,
#     path=None,
#     force=False,
#     quiet=True
# )
api.dataset_download_file(
    "thorgodofthunder/tvradionewspaperadvertising", "Advertising.csv"
)
