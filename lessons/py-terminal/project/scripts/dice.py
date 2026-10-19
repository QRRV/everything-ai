# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0

# A long job: roll a die ten million times, in five parts, and count the sixes.
import random

random.seed(7)
sixes = 0
for part in range(5):
    for i in range(2000000):
        if random.randint(1, 6) == 6:
            sixes = sixes + 1
    print(f"part {part + 1} of 5 done")
print(f"{sixes} sixes in ten million rolls")
