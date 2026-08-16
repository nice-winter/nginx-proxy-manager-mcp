"""NPM API client with authentication handling."""

import logging
import os
import re
from typing import Optional, List
import httpx
from .models import (
    ProxyHost, Certificate, AccessList, RedirectionHost,
    Stream, DeadHost, User, Setting, AuditLogEntry, NPMConfig,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF = 1.0  # seconds


class NPMClientError(Exception):
    pass


class NPMConfigError(NPMClientError):
    pass


class NPMAuthenticationError(NPMClientError):
    pass


class NPMNetworkError(NPMClientError):
    pass


def _validate_int_id(value, name: str) -> int:
    """Cast to int and validate positive. Prevents path traversal via string IDs."""
    try:
        int_val = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got: {type(value).__name__}")
    if int_val < 1:
        raise ValueError(f"{name} must be >= 1, got: {int_val}")
    return int_val


_SETTING_ID_PATTERN = re.compile(r"^[a-z0-9\-]{1,64}$")


def _validate_setting_id(value) -> str:
    """Validate setting_id is a safe alphanumeric-dash string."""
    s = str(value)
    if not _SETTING_ID_PATTERN.match(s):
        raise ValueError(f"setting_id must match [a-z0-9-]{{1,64}}, got: {s!r}")
    return s


class NPMClient:
    def __init__(self, config: NPMConfig):
        self.config = config
        self.base_url = config.url.rstrip("/") if config.url else ""
        self._token: Optional[str] = None
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=False)
        self._validate_config()

    def _validate_config(self):
        if not self.config.url:
            raise NPMConfigError("NPM_URL is required but not set")
        if not self.config.email:
            raise NPMConfigError("NPM_EMAIL is required but not set")
        if not self.config.password.get_secret_value():
            raise NPMConfigError("NPM_PASSWORD is required but not set")

    async def _get_token(self) -> str:
        if self._token:
            return self._token

        try:
            response = await self._client.post(
                f"{self.base_url}/api/tokens",
                json={
                    "identity": self.config.email,
                    "secret": self.config.password.get_secret_value(),
                },
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["token"]
            logger.info("NPM auth token obtained (len=%d)", len(self._token))
            return self._token
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise NPMAuthenticationError(
                    "Authentication failed: invalid NPM email or password"
                ) from e
            body = e.response.text[:500] if e.response.content else "(empty)"
            raise NPMNetworkError(
                f"HTTP {e.response.status_code} on token auth: {body}"
            ) from e
        except httpx.ConnectError as e:
            raise NPMNetworkError("Cannot connect to NPM. Check NPM_URL and that NPM is running.") from e
        except httpx.TimeoutException as e:
            raise NPMNetworkError("Request to NPM timed out") from e
        except NPMClientError:
            raise
        except Exception as e:
            raise NPMNetworkError(f"Unexpected error during auth: {type(e).__name__}: {e}") from e

    async def _request(
        self,
        method: str,
        path: str,
        retry_auth: bool = True,
        _attempt: int = 0,
        **kwargs,
    ) -> httpx.Response:
        try:
            token = await self._get_token()
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {token}"

            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                **kwargs,
            )

            logger.debug("NPM API %s %s -> %d", method, path, response.status_code)

            if response.status_code == 401 and retry_auth:
                logger.info("Token expired, clearing and retrying")
                self._token = None
                return await self._request(method, path, retry_auth=False, **kwargs)

            response.raise_for_status()
            return response
        except NPMClientError:
            raise
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response.content else "(empty)"
            raise NPMNetworkError(
                f"HTTP {e.response.status_code} error on {method} {path}: {body}"
            ) from e
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.WriteError) as e:
            if _attempt < MAX_RETRIES:
                logger.warning("NPM API %s %s failed (%s), retrying (%d/%d)",
                              method, path, type(e).__name__, _attempt + 1, MAX_RETRIES)
                import asyncio
                await asyncio.sleep(RETRY_BACKOFF * (_attempt + 1))
                return await self._request(method, path, retry_auth=retry_auth, _attempt=_attempt + 1, **kwargs)
            raise NPMNetworkError(
                f"Network error on {method} {path} after {MAX_RETRIES} retries: {type(e).__name__}: {e}"
            ) from e

    async def list_proxy_hosts(self) -> List[ProxyHost]:
        response = await self._request("GET", "/api/nginx/proxy-hosts")
        return [ProxyHost(**host) for host in response.json()]

    async def get_proxy_host(self, host_id: int) -> ProxyHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request("GET", f"/api/nginx/proxy-hosts/{host_id}")
        return ProxyHost(**response.json())

    async def create_proxy_host(self, host: ProxyHost) -> ProxyHost:
        response = await self._request(
            "POST",
            "/api/nginx/proxy-hosts",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "locations", "enabled",
                "allow_websocket_upgrade", "http2_support",
                "hsts_enabled", "hsts_subdomains",
            }),
        )
        return ProxyHost(**response.json())

    async def update_proxy_host(self, host_id: int, host: ProxyHost) -> ProxyHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request(
            "PUT",
            f"/api/nginx/proxy-hosts/{host_id}",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on", "owner_user_id",
                "meta", "locations", "enabled",
                "allow_websocket_upgrade", "http2_support",
                "hsts_enabled", "hsts_subdomains",
            }),
        )
        return ProxyHost(**response.json())

    async def delete_proxy_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("DELETE", f"/api/nginx/proxy-hosts/{host_id}")

    async def enable_proxy_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/proxy-hosts/{host_id}/enable")

    async def disable_proxy_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/proxy-hosts/{host_id}/disable")

    async def list_certificates(self) -> List[Certificate]:
        response = await self._request("GET", "/api/nginx/certificates")
        return [Certificate(**cert) for cert in response.json()]

    async def request_certificate(self, cert: Certificate) -> Certificate:
        response = await self._request(
            "POST",
            "/api/nginx/certificates",
            json=cert.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "expires_on",
            }),
        )
        return Certificate(**response.json())

    async def get_certificate(self, cert_id: int) -> Certificate:
        cert_id = _validate_int_id(cert_id, "certificate_id")
        response = await self._request("GET", f"/api/nginx/certificates/{cert_id}")
        return Certificate(**response.json())

    async def delete_certificate(self, cert_id: int) -> None:
        cert_id = _validate_int_id(cert_id, "certificate_id")
        await self._request("DELETE", f"/api/nginx/certificates/{cert_id}")

    async def renew_certificate(self, cert_id: int) -> Certificate:
        cert_id = _validate_int_id(cert_id, "certificate_id")
        response = await self._request("POST", f"/api/nginx/certificates/{cert_id}/renew")
        return Certificate(**response.json())

    async def list_dns_providers(self) -> list[dict]:
        response = await self._request("GET", "/api/nginx/certificates/dns-providers")
        return response.json()

    async def test_http_challenge(self, domains: List[str]) -> dict:
        response = await self._request(
            "POST", "/api/nginx/certificates/test-http", json={"domains": domains}
        )
        return response.json()

    async def list_access_lists(self) -> List[AccessList]:
        response = await self._request("GET", "/api/nginx/access-lists")
        return [AccessList(**al) for al in response.json()]

    async def create_access_list(self, access_list: AccessList) -> AccessList:
        response = await self._request(
            "POST",
            "/api/nginx/access-lists",
            json=access_list.model_dump(exclude_none=True, exclude={"id", "created_on", "modified_on"}),
        )
        return AccessList(**response.json())

    async def update_access_list(self, access_list_id: int, access_list: AccessList) -> AccessList:
        access_list_id = _validate_int_id(access_list_id, "access_list_id")
        response = await self._request(
            "PUT",
            f"/api/nginx/access-lists/{access_list_id}",
            json=access_list.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "items",
            }),
        )
        return AccessList(**response.json())

    async def get_access_list(self, access_list_id: int) -> AccessList:
        access_list_id = _validate_int_id(access_list_id, "access_list_id")
        response = await self._request("GET", f"/api/nginx/access-lists/{access_list_id}")
        return AccessList(**response.json())

    async def delete_access_list(self, access_list_id: int) -> None:
        access_list_id = _validate_int_id(access_list_id, "access_list_id")
        await self._request("DELETE", f"/api/nginx/access-lists/{access_list_id}")

    async def list_redirection_hosts(self) -> List[RedirectionHost]:
        response = await self._request("GET", "/api/nginx/redirection-hosts")
        return [RedirectionHost(**h) for h in response.json()]

    async def get_redirection_host(self, host_id: int) -> RedirectionHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request("GET", f"/api/nginx/redirection-hosts/{host_id}")
        return RedirectionHost(**response.json())

    async def create_redirection_host(self, host: RedirectionHost) -> RedirectionHost:
        response = await self._request(
            "POST", "/api/nginx/redirection-hosts",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "enabled",
                "hsts_enabled", "hsts_subdomains",
                "http2_support",
            }),
        )
        return RedirectionHost(**response.json())

    async def update_redirection_host(self, host_id: int, host: RedirectionHost) -> RedirectionHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request(
            "PUT",
            f"/api/nginx/redirection-hosts/{host_id}",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on", "owner_user_id",
                "meta", "enabled",
                "hsts_enabled", "hsts_subdomains",
                "http2_support",
            }),
        )
        return RedirectionHost(**response.json())

    async def delete_redirection_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("DELETE", f"/api/nginx/redirection-hosts/{host_id}")

    async def enable_redirection_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/redirection-hosts/{host_id}/enable")

    async def disable_redirection_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/redirection-hosts/{host_id}/disable")

    async def list_streams(self) -> List[Stream]:
        response = await self._request("GET", "/api/nginx/streams")
        return [Stream(**s) for s in response.json()]

    async def get_stream(self, stream_id: int) -> Stream:
        stream_id = _validate_int_id(stream_id, "stream_id")
        response = await self._request("GET", f"/api/nginx/streams/{stream_id}")
        return Stream(**response.json())

    async def create_stream(self, stream: Stream) -> Stream:
        response = await self._request(
            "POST", "/api/nginx/streams",
            json=stream.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "enabled",
            }),
        )
        return Stream(**response.json())

    async def update_stream(self, stream_id: int, stream: Stream) -> Stream:
        stream_id = _validate_int_id(stream_id, "stream_id")
        response = await self._request(
            "PUT",
            f"/api/nginx/streams/{stream_id}",
            json=stream.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on", "owner_user_id",
                "meta", "enabled",
            }),
        )
        return Stream(**response.json())

    async def delete_stream(self, stream_id: int) -> None:
        stream_id = _validate_int_id(stream_id, "stream_id")
        await self._request("DELETE", f"/api/nginx/streams/{stream_id}")

    async def enable_stream(self, stream_id: int) -> None:
        stream_id = _validate_int_id(stream_id, "stream_id")
        await self._request("POST", f"/api/nginx/streams/{stream_id}/enable")

    async def disable_stream(self, stream_id: int) -> None:
        stream_id = _validate_int_id(stream_id, "stream_id")
        await self._request("POST", f"/api/nginx/streams/{stream_id}/disable")

    async def list_dead_hosts(self) -> List[DeadHost]:
        response = await self._request("GET", "/api/nginx/dead-hosts")
        return [DeadHost(**h) for h in response.json()]

    async def get_dead_host(self, host_id: int) -> DeadHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request("GET", f"/api/nginx/dead-hosts/{host_id}")
        return DeadHost(**response.json())

    async def create_dead_host(self, host: DeadHost) -> DeadHost:
        response = await self._request(
            "POST", "/api/nginx/dead-hosts",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "enabled",
            }),
        )
        return DeadHost(**response.json())

    async def update_dead_host(self, host_id: int, host: DeadHost) -> DeadHost:
        host_id = _validate_int_id(host_id, "host_id")
        response = await self._request(
            "PUT",
            f"/api/nginx/dead-hosts/{host_id}",
            json=host.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on", "owner_user_id",
                "meta", "enabled",
            }),
        )
        return DeadHost(**response.json())

    async def delete_dead_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("DELETE", f"/api/nginx/dead-hosts/{host_id}")

    async def enable_dead_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/dead-hosts/{host_id}/enable")

    async def disable_dead_host(self, host_id: int) -> None:
        host_id = _validate_int_id(host_id, "host_id")
        await self._request("POST", f"/api/nginx/dead-hosts/{host_id}/disable")

    async def list_users(self) -> List[User]:
        response = await self._request("GET", "/api/users")
        return [User(**u) for u in response.json()]

    async def get_user(self, user_id: int) -> User:
        user_id = _validate_int_id(user_id, "user_id")
        response = await self._request("GET", f"/api/users/{user_id}")
        return User(**response.json())

    async def create_user(self, user: User) -> User:
        response = await self._request(
            "POST", "/api/users",
            json=user.model_dump(exclude_none=True, exclude={"id", "created_on", "modified_on", "meta", "avatar"}),
        )
        return User(**response.json())

    async def update_user(self, user_id: int, user: User) -> User:
        user_id = _validate_int_id(user_id, "user_id")
        response = await self._request(
            "PUT", f"/api/users/{user_id}",
            json=user.model_dump(exclude_none=True, exclude={
                "id", "created_on", "modified_on",
                "meta", "avatar",
            }),
        )
        return User(**response.json())

    async def delete_user(self, user_id: int) -> None:
        user_id = _validate_int_id(user_id, "user_id")
        await self._request("DELETE", f"/api/users/{user_id}")

    async def list_settings(self) -> List[Setting]:
        response = await self._request("GET", "/api/settings")
        return [Setting(**s) for s in response.json()]

    async def get_setting(self, setting_id: str) -> Setting:
        setting_id = _validate_setting_id(setting_id)
        response = await self._request("GET", f"/api/settings/{setting_id}")
        return Setting(**response.json())

    async def update_setting(self, setting_id: str, setting: Setting) -> Setting:
        setting_id = _validate_setting_id(setting_id)
        payload = {"value": setting.value}
        if setting.meta:
            payload["meta"] = setting.meta
        response = await self._request(
            "PUT", f"/api/settings/{setting_id}",
            json=payload,
        )
        return Setting(**response.json())

    async def list_audit_log(self) -> List[AuditLogEntry]:
        response = await self._request("GET", "/api/audit-log")
        return [AuditLogEntry(**e) for e in response.json()]

    async def get_host_report(self) -> dict:
        response = await self._request("GET", "/api/reports/hosts")
        return response.json()

    async def close(self):
        await self._client.aclose()


def create_client_from_env() -> NPMClient:
    config = NPMConfig(
        url=os.getenv("NPM_URL", ""),
        email=os.getenv("NPM_EMAIL", ""),
        password=os.getenv("NPM_PASSWORD", ""),
    )
    return NPMClient(config)
