"""Resolve explicit launch paths to context-local attention provider choices."""

from uniserve.nn.attention import Attention
from uniserve_worker.errors import invalid_descriptor


def resolve_overrides(model, declarations):
    """Resolve every declared path once and reject misspelled paths."""
    result = {}
    for path, provider in declarations:
        try:
            module = model.get_submodule(path)
        except AttributeError as error:
            raise invalid_descriptor(
                f"attention override names unknown module {path!r}"
            ) from error
        if not isinstance(module, Attention):
            raise invalid_descriptor(
                f"attention override {path!r} does not name an Attention layer"
            )
        if module in result and result[module] != provider:
            raise invalid_descriptor(
                f"attention override {path!r} conflicts with an alias "
                "of the same layer"
            )
        result[module] = provider
    return result


def select_overrides(module, overrides):
    """Select the already-resolved declarations borrowed by one component."""
    return {
        child: overrides[child]
        for child in module.modules()
        if child in overrides
    }
