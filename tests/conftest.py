# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import MagicMock

# Setup dummy environment variables for tests
os.environ["GOOGLE_CLOUD_PROJECT"] = "mock-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["INTEGRATION_TEST"] = "TRUE"

# Mock google.auth.default to prevent DefaultCredentialsError
try:
    import google.auth
    from google.auth.credentials import Credentials
    
    class MockCredentials(Credentials):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.token = "mock-token"
        def refresh(self, request):
            self.token = "mock-token"
            
    mock_creds = MockCredentials()
    google.auth.default = lambda *args, **kwargs: (mock_creds, "mock-project-id")
except ImportError:
    pass

# Mock google.cloud.logging.Client to prevent GoogleAuthError in tests
try:
    import google.cloud.logging
    mock_logging_client = MagicMock()
    mock_logger = MagicMock()
    mock_logging_client.logger.return_value = mock_logger
    google.cloud.logging.Client = MagicMock(return_value=mock_logging_client)
except ImportError:
    pass
