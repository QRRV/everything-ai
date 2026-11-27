# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# One episode of an agent in a grid world: the Everything AI course put this loop here.
# Eén episode van een agent in een grid world: de Everything AI-course heeft deze loop hier gezet.
# It is your file: your curves.py lets your agents play their episodes with it.
# Het is jouw bestand: je curves.py speelt er de episodes van je agents mee.


def episode(agent, world, limit):
    # the agent and the world take turns until the goal, or until `limit` steps; gives back the return
    # de agent en de wereld zijn om de beurt aan zet tot het doel, of tot `limit` stappen; geeft de return terug
    state = world.reset()
    action = agent.act(state)
    total = 0
    for t in range(limit):
        nxt, reward, done = world.step(action)
        # the next action first: SARSA learns from it
        # eerst de volgende action: SARSA leert ervan
        nxt_action = agent.act(nxt)
        agent.learn(state, action, reward, nxt, nxt_action)
        total = total + reward
        if done:
            break
        state = nxt
        action = nxt_action
    # Monte Carlo learns now, from the whole episode
    # Monte Carlo leert nu, van de hele episode
    agent.end()
    return total
