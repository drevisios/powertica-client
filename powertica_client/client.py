import os
from typing import Optional
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
import uuid
import hashlib
import pandas as pd


class PowerticaClient:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.cred = InteractiveBrowserCredential(tenant_id=self.tenant_id)
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
        dir = self._download_artifact(uri=uri, cache_key=cache_key)
        return pd.read_parquet(dir)
