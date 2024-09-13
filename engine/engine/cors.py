from typing import Iterable


def _add_cors_headers(response, methods: Iterable[str]) -> None:
    allow_methods = list(set(methods))
    if "OPTIONS" not in allow_methods:
        allow_methods.append("OPTIONS")
    headers = {
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Headers": "*",
    }
    response.headers.extend(headers)


def add_cors_headers(request, response):
    _add_cors_headers(response, list(request.route.methods))
