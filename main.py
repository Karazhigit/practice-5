GRID = [
    ["A", "B", "C", "G"],
    ["D", "#", "E", "X"],
    ["S", "F", "H", "I"],
]
POSITIONS = {
    cell: (row, col)
    for row, line in enumerate(GRID)
    for col, cell in enumerate(line)
    if cell != "#"
}
STATE_AT = {position: state for state, position in POSITIONS.items()}
TERMINALS = {"G", "X"}
STATES = [s for s in POSITIONS if s not in TERMINALS]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0),
         "LEFT": (0, -1), "RIGHT": (0, 1)}
SIDES = {"UP": ("LEFT", "RIGHT"), "DOWN": ("RIGHT", "LEFT"),
         "LEFT": ("DOWN", "UP"), "RIGHT": ("UP", "DOWN")}


def move(state, direction):
    """Қабырғаға немесе шекараға тірелсе, робот орнында қалады."""
    row, col = POSITIONS[state]
    dr, dc = MOVES[direction]
    return STATE_AT.get((row + dr, col + dc), state)


def transitions(state, action, p_intended=0.8,
                danger_reward=-10.0, living_reward=-0.1):
    """Нәтиже: [(ықтималдық, келесі күй, сыйақы), ...]."""
    # Терминалға кіргенде сыйақы бір рет беріледі.
    # Терминалдан кейін жаңа сыйақы жиналмайды.
    if state in TERMINALS:
        return [(1.0, state, 0.0)]
    if not 0.0 <= p_intended <= 1.0:
        raise ValueError("Қозғалыс ықтималдығы 0 мен 1 арасында болсын.")

    p_side = (1.0 - p_intended) / 2.0
    left, right = SIDES[action]
    outcomes = [(action, p_intended), (left, p_side), (right, p_side)]
    probabilities = {}
    for direction, probability in outcomes:
        next_state = move(state, direction)
        probabilities[next_state] = (
            probabilities.get(next_state, 0.0) + probability
        )

    result = []
    for next_state, probability in probabilities.items():
        if next_state == "G":
            reward = 10.0
        elif next_state == "X":
            reward = danger_reward
        else:
            reward = living_reward
        result.append((probability, next_state, reward))
    return result


def q_value(state, action, values, gamma=0.9, **model):
    """Әр ықтимал нәтиженің үлесін қосып, Q мәнін есептеу."""
    q = 0.0
    for probability, next_state, reward in transitions(state, action, **model):
        next_value = 0.0 if next_state in TERMINALS else values[next_state]
        q += probability * (reward + gamma * next_value)
    return q


def value_iteration(gamma=0.9, theta=0.001, **model):
    """V, саясат, барлық итерациялар және соңғы delta мәнін қайтарады."""
    if not 0.0 <= gamma < 1.0 or theta <= 0.0:
        raise ValueError("0 <= gamma < 1 және theta > 0 болуы қажет.")
    values = {state: 0.0 for state in STATES}
    history = [values.copy()]

    for _ in range(10000):
        new_values = {}
        for state in STATES:
            # Барлық есептеулерде алдыңғы итерацияның values мәндері қолданылады.
            new_values[state] = max(
                q_value(state, action, values, gamma, **model)
                for action in ACTIONS
            )
        delta = max(abs(new_values[s] - values[s]) for s in STATES)
        values = new_values
        history.append(values.copy())
        if delta < theta:
            break
    else:
        raise RuntimeError("Итерациялар шегіне жетті.")

    policy = {
        state: max(ACTIONS,
                   key=lambda action: q_value(state, action, values, gamma, **model))
        for state in STATES
    }
    return values, policy, history, delta


def print_policy(policy):
    for row in GRID:
        print(" | ".join(f"{cell}:{policy.get(cell, 'END' if cell in TERMINALS else '#'):<5}"
                         for cell in row))


def main():
    values, policy, history, delta = value_iteration()
    print("2-ТАПСЫРМА: АУЫСУЛАР")
    for state in ["F", "E", "C"]:
        rows = [(round(p, 3), s, r) for p, s, r in transitions(state, "RIGHT")]
        print(state, "RIGHT:", rows)

    print("\n3-ТАПСЫРМА: ДИСКОНТТАУ")
    for rewards, gamma in [([2, 4, 6], 0.5), ([2, 4, 6], 0.9), ([0, 0, 10], 0.5)]:
        utility = sum(gamma ** t * reward for t, reward in enumerate(rewards))
        print(rewards, "gamma =", gamma, "U =", round(utility, 6))

    print("\n5-ТАПСЫРМА: V0, V1, V2")
    for state in ["C", "E", "H", "I"]:
        print(state, [round(history[k][state], 6) for k in range(3)])

    print("\n6-ТАПСЫРМА: СОҢҒЫ МӘНДЕР МЕН ӘРЕКЕТТЕР")
    print("Итерациялар:", len(history) - 1, "delta =", round(delta, 9))
    for state in STATES:
        scores = {a: round(q_value(state, a, values), 6) for a in ACTIONS}
        print(f"{state}: V = {values[state]:.6f}; Q = {scores}; best = {policy[state]}")
    print("G: V = 0; X: V = 0 (терминалдан кейінгі мәндер)")
    print_policy(policy)

    experiments = [
        ("gamma = 0.2", {"gamma": 0.2}),
        ("p = 0.6; бүйірлер = 0.2 / 0.2", {"p_intended": 0.6}),
        ("X сыйақысы = -30", {"danger_reward": -30.0}),
        ("Қадам сыйақысы = -1", {"living_reward": -1.0}),
    ]
    print("\n7-ТАПСЫРМА: ЭКСПЕРИМЕНТТЕР")
    # Әр эксперимент бастапқы параметрлерден бөлек орындалады.
    for label, parameters in experiments:
        exp_values, exp_policy, exp_history, exp_delta = value_iteration(**parameters)
        changes = [f"{s}: {policy[s]} -> {exp_policy[s]}"
                   for s in STATES if exp_policy[s] != policy[s]]
        print("\n", label)
        print("Итерациялар:", len(exp_history) - 1, "delta:", round(exp_delta, 9))
        print("Саясат өзгерістері:", "; ".join(changes) if changes else "жоқ")
        for state in STATES:
            print(f"{state}: V = {exp_values[state]:.6f}; action = {exp_policy[state]}")
        print_policy(exp_policy)


if __name__ == "__main__":
    main()
