from prometheus_client import start_http_server

start_http_server(8001)  # Metrics exposed at /metrics on port 8001
