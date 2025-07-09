def freeze_modules(modules):
    """Freezes the parameters of the given module(s)."""
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

def unfreeze_modules(modules):
    """Unfreezes the parameters of the given module(s)."""
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True