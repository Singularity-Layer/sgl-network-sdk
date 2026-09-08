"""ProcessorsClient credential-handling tests.

These pin the audit findings rather than the happy path. Every one of them is a bug that
existed: the key was sent to any https origin, IPv6 loopback was refused, `catalogue()` was
authenticated and so returned your own processors instead of the catalogue, and an explicit
:443 or :0 confused the official-host check.
"""

import pytest

from singularity_grid import ProcessorsClient, PROCESSORS_BASE_URL

KEY = "x402c_test"


def test_official_host_accepts_the_key():
    c = ProcessorsClient(api_key=KEY, base_url=PROCESSORS_BASE_URL)
    assert c._client.headers.get("X-API-Key") == KEY


def test_explicit_default_port_is_still_the_official_host():
    # `p.port or default` used to fold None and 443 together correctly but rejected the
    # explicit form, pushing callers into allow_key_on_custom_host for the real host.
    ProcessorsClient(api_key=KEY, base_url="https://processors.x402compute.cc:443")


def test_port_zero_is_not_the_official_host():
    # `or` treats 0 as falsy, which authorized https://host:0 as official.
    with pytest.raises(ValueError):
        ProcessorsClient(api_key=KEY, base_url="https://processors.x402compute.cc:0")


@pytest.mark.parametrize("url", ["http://localhost:8787", "http://127.0.0.1:8787", "http://[::1]:8787"])
def test_loopback_is_allowed_over_plain_http(url):
    c = ProcessorsClient(api_key=KEY, base_url=url)
    # trust_env must be OFF here, or HTTP_PROXY would carry the plaintext key to a proxy.
    assert c._client.trust_env is False


def test_proxy_support_is_kept_for_the_official_host():
    assert ProcessorsClient(api_key=KEY)._client.trust_env is True


def test_key_is_refused_on_a_foreign_https_host():
    # https alone was the original bug: an attacker-influenced env var pointing at any valid
    # TLS origin was enough to harvest a long-lived full-control credential.
    with pytest.raises(ValueError):
        ProcessorsClient(api_key=KEY, base_url="https://attacker.example")


def test_foreign_host_is_fine_without_a_key():
    ProcessorsClient(base_url="https://attacker.example")


def test_foreign_host_allowed_with_explicit_opt_in():
    c = ProcessorsClient(api_key=KEY, base_url="https://attacker.example", allow_key_on_custom_host=True)
    assert c._client.headers.get("X-API-Key") == KEY


@pytest.mark.parametrize("url", ["http://evil.tld", "ftp://x", "not a url"])
def test_unsafe_schemes_are_refused(url):
    with pytest.raises(ValueError):
        ProcessorsClient(api_key=KEY, base_url=url)
