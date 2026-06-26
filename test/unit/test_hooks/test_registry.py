import importlib

import pytest

import tgm.hooks.registry as registry_module
from tgm.hooks import DGHook


@pytest.fixture(autouse=True)
def fresh_registry():
    importlib.reload(registry_module)
    yield registry_module


def test_hook_returns_the_class_unchanged(fresh_registry):
    class FooHook(DGHook):
        pass

    assert fresh_registry.hook(FooHook) is FooHook


def test_hook_registers_single_class(fresh_registry):
    class FooHook(DGHook):
        pass

    fresh_registry.hook(FooHook)
    assert FooHook in fresh_registry.list_hooks()


def test_hook_registers_multiple_classes(fresh_registry):
    class FooHook(DGHook):
        pass

    class WoofHook(DGHook):
        pass

    fresh_registry.hook(FooHook)
    fresh_registry.hook(WoofHook)

    hooks = fresh_registry.list_hooks()
    assert FooHook in hooks
    assert WoofHook in hooks


def test_hook_preserves_registration_order(fresh_registry):
    class First(DGHook):
        pass

    class Second(DGHook):
        pass

    class Third(DGHook):
        pass

    fresh_registry.hook(First)
    fresh_registry.hook(Second)
    fresh_registry.hook(Third)

    assert fresh_registry.list_hooks() == [First, Second, Third]


def test_hook_can_be_used_as_decorator_syntax(fresh_registry):
    @fresh_registry.hook
    class FooHook(DGHook):
        pass

    assert FooHook in fresh_registry.list_hooks()


def test_hook_does_not_alter_class_attributes(fresh_registry):
    class FooHook(DGHook):
        value = 42

        def method(self):
            return 'hello'

    fresh_registry.hook(FooHook)
    assert FooHook.value == 42
    assert FooHook().method() == 'hello'


def test_registering_same_class_twice_adds_it_twice(fresh_registry):
    class FooHook(DGHook):
        pass

    fresh_registry.hook(FooHook)
    fresh_registry.hook(FooHook)

    assert fresh_registry.list_hooks().count(FooHook) == 2


def test_empty_registry_returns_empty_list(fresh_registry):
    assert fresh_registry.list_hooks() == []


def test_list_hooks_returns_a_list(fresh_registry):
    assert isinstance(fresh_registry.list_hooks(), list)


def test_list_hooks_contains_only_registered_classes(fresh_registry):
    class Registered(DGHook):
        pass

    class NotRegistered(DGHook):
        pass

    fresh_registry.hook(Registered)
    assert Registered in fresh_registry.list_hooks()
    assert NotRegistered not in fresh_registry.list_hooks()
