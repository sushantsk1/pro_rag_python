from scripts.seed_demo import seed


def test_demo_seed_is_idempotent_and_makes_no_generation_calls(client):
    # TestClient supports the same request interface as the CLI's httpx client.
    class PrefixedClient:
        def request(self, method, route, **kwargs):
            return client.request(method, "/api/v1" + route, **kwargs)

    first = seed(PrefixedClient())
    second = seed(PrefixedClient())
    assert first["id"] == second["id"]
    service = client.app.state.service
    assert len(service.store.all("document", first["id"])) == 3
    assert len(service.store.all("eval_case", first["id"])) == 3
    assert service.store.all("query") == []
