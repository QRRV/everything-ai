# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# The fixture expect for the tests in this folder: pytest reads this file first, and gives the
# fixture to every test function with an argument named expect. The Everything AI course put it here.
# De fixture expect voor de tests in deze map: pytest leest dit bestand eerst, en geeft de fixture
# aan elke testfunctie met een argument met de naam expect. De Everything AI-course zette het hier.
import pytest

from ops.serving import load


@pytest.fixture
def expect():
    # what your packed model in ops/model expects for a table of hours
    # wat je verpakte model in ops/model verwacht voor een tabel met uren
    return load()
