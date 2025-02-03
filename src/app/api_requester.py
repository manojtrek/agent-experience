import requests

class APIRequester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def get(self, url, params=None):
        return requests.get(url, params=params)

    def post(self, url, json=None):
        return requests.post(url, json=json)

    def put(self, url, json=None):
        return requests.put(url, json=json)

    def patch(self, url, json=None):
        return requests.patch(url, json=json)

    def delete(self, url, json=None):
        return requests.delete(url, json=json)
