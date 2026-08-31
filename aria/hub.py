from dataclasses import dataclass, field

@dataclass
class GPGLDataHub:
    region: object
    adapters: dict = field(default_factory=dict)

    def register(self, adapter):
        self.adapters[adapter.name] = adapter
        return self

    def fetch(self, sources, start, end, source_kwargs=None):
        source_kwargs = source_kwargs or {}
        results, errors = {}, {}
        for name in sources:
            try:
                results[name] = self.adapters[name].fetch(
                    start, end, **source_kwargs.get(name, {})
                )
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        return results, errors
