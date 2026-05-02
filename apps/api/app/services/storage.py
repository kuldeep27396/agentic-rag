import httpx


class BlobStorage:
    async def download(self, blob_url: str) -> bytes:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(blob_url)
            response.raise_for_status()
            return response.content


storage = BlobStorage()

