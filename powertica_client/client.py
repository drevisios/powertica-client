import os
from linecache import cache
from typing import Optional
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from azure.identity import InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml import MLClient
import uuid
import hashlib
from sklearn.pipeline import Pipeline
import pandas as pd
import cloudpickle


class PowerticaClient:
    def __init__(self, tenant_id: str, use_cli_creds:bool=False):
        self.tenant_id = tenant_id
        self.cred = AzureCliCredential() if use_cli_creds else InteractiveBrowserCredential(tenant_id=self.tenant_id)
        self.client = MLClient.from_config(credential=self.cred)

    def _download_artifact(self, uri: str, cache_key: Optional[str] = None) -> str:
        cache_key = cache_key or uuid.uuid4().hex
        destination = f".artifacts/{cache_key}"

        if os.path.exists(destination):
            return destination
        else:
            os.makedirs(destination)

        artifact_utils.download_artifact_from_aml_uri(uri,
                                                      destination=destination,
                                                      datastore_operation=self.client.datastores
                                                      )
        return destination

    def download_dataset(self, name: str, version: str, pattern: str) -> pd.DataFrame:
        dataset = self.client.data.get(name=name, version=version)
        uri = dataset.path + pattern
        cache_key = hashlib.sha256(uri.encode('utf-8')).hexdigest()
        files_dir = self._download_artifact(uri=uri, cache_key=cache_key)
        return pd.read_parquet(files_dir)

    def download_model(self, name: str, version: str) -> Pipeline:
        model = self.client.models.get(name=name, version=version)
        uri = model.path
        cache_key = hashlib.sha256(uri.encode('utf-8')).hexdigest()
        files_dir = self._download_artifact(uri=uri, cache_key=cache_key)
        first_file_path = os.path.join(files_dir, os.listdir(files_dir)[0])
        pipeline = cloudpickle.load(open(first_file_path, "rb"))
        return pipeline
