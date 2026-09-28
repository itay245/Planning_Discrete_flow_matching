from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus

def create_blocksworld_problem(config):
    """Build a problem from blocks, initial_state, and goal_state.

    `blocks` is a nonempty list of unique block names. Each state maps
    fact tuples to booleans, e.g. {("on", "A", "B"): True,
    ("handempty",): False}. Supported predicates are on, ontable, clear,
    holding, and handempty. Omitted initial facts are false; omitted goal
    facts are unconstrained. Include clear and handempty explicitly when
    needed; they are not inferred from block positions.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    for key in ("blocks", "initial_state", "goal_state"):
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    names = config["blocks"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("blocks must be a nonempty list of unique, nonempty strings")
    # ---------------------------------------------------------
    # Types and objects
    # ---------------------------------------------------------

    Block = UserType("Block")

    blocks = {name: Object(name, Block) for name in names}

    problem = Problem(f"blocksworld_{len(blocks)}_blocks")
    problem.add_objects(list(blocks.values()))

    # ---------------------------------------------------------
    # Fluents
    # ---------------------------------------------------------

    # on(x, y): block x is directly on block y
    on = Fluent(
        "on",
        BoolType(),
        x=Block,
        y=Block,
    )

    # ontable(x): block x is on the table
    ontable = Fluent(
        "ontable",
        BoolType(),
        x=Block,
    )

    # clear(x): nothing is on top of x
    clear = Fluent(
        "clear",
        BoolType(),
        x=Block,
    )

    # holding(x): the hand is holding x
    holding = Fluent(
        "holding",
        BoolType(),
        x=Block,
    )

    # handempty: the single hand is empty
    handempty = Fluent(
        "handempty",
        BoolType(),
    )

    # Anything not explicitly true is false.
    problem.add_fluent(on, default_initial_value=False)
    problem.add_fluent(ontable, default_initial_value=False)
    problem.add_fluent(clear, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)

    # ---------------------------------------------------------
    # Action: pick_up(x)
    #
    # Pick a clear block up from the table.
    # ---------------------------------------------------------

    pick_up = InstantaneousAction(
        "pick_up",
        x=Block,
    )

    x = pick_up.parameter("x")

    pick_up.add_precondition(ontable(x))
    pick_up.add_precondition(clear(x))
    pick_up.add_precondition(handempty)

    pick_up.add_effect(ontable(x), False)
    pick_up.add_effect(clear(x), False)
    pick_up.add_effect(handempty, False)
    pick_up.add_effect(holding(x), True)

    problem.add_action(pick_up)

    # ---------------------------------------------------------
    # Action: put_down(x)
    #
    # Put a held block onto the table.
    # ---------------------------------------------------------

    put_down = InstantaneousAction(
        "put_down",
        x=Block,
    )

    x = put_down.parameter("x")

    put_down.add_precondition(holding(x))

    put_down.add_effect(holding(x), False)
    put_down.add_effect(ontable(x), True)
    put_down.add_effect(clear(x), True)
    put_down.add_effect(handempty, True)

    problem.add_action(put_down)

    # ---------------------------------------------------------
    # Action: stack(x, y)
    #
    # Put held block x on top of clear block y.
    # ---------------------------------------------------------

    stack = InstantaneousAction(
        "stack",
        x=Block,
        y=Block,
    )

    x = stack.parameter("x")
    y = stack.parameter("y")

    stack.add_precondition(holding(x))
    stack.add_precondition(clear(y))

    # Prevent nonsensical stack(A, A)
    stack.add_precondition(Not(Equals(x, y)))

    stack.add_effect(holding(x), False)
    stack.add_effect(clear(y), False)
    stack.add_effect(clear(x), True)
    stack.add_effect(handempty, True)
    stack.add_effect(on(x, y), True)

    problem.add_action(stack)

    # ---------------------------------------------------------
    # Action: unstack(x, y)
    #
    # Remove clear block x from block y.
    # ---------------------------------------------------------

    unstack = InstantaneousAction(
        "unstack",
        x=Block,
        y=Block,
    )

    x = unstack.parameter("x")
    y = unstack.parameter("y")

    unstack.add_precondition(on(x, y))
    unstack.add_precondition(clear(x))
    unstack.add_precondition(handempty)

    unstack.add_precondition(Not(Equals(x, y)))

    unstack.add_effect(on(x, y), False)
    unstack.add_effect(clear(y), True)
    unstack.add_effect(clear(x), False)
    unstack.add_effect(handempty, False)
    unstack.add_effect(holding(x), True)

    problem.add_action(unstack)

    predicates = {
        "on": (on, 2),
        "ontable": (ontable, 1),
        "clear": (clear, 1),
        "holding": (holding, 1),
        "handempty": (handempty, 0),
    }

    for state_name in ("initial_state", "goal_state"):
        state = config[state_name]
        if not isinstance(state, dict):
            raise ValueError(f"{state_name} must be a dictionary of fact tuples to booleans")
        for fact, value in state.items():
            if (not isinstance(fact, tuple) or not fact
                    or not isinstance(fact[0], str) or fact[0] not in predicates):
                raise ValueError(f"Invalid fact in {state_name}: {fact!r}")
            fluent, arity = predicates[fact[0]]
            if len(fact) != arity + 1:
                raise ValueError(f"{fact[0]} expects {arity} block arguments")
            if any(not isinstance(name, str) or name not in blocks for name in fact[1:]):
                raise ValueError(f"Unknown block in {state_name}: {fact!r}")
            if not isinstance(value, bool):
                raise ValueError(f"Fact values must be booleans: {fact!r}")
            expression = fluent(*(blocks[name] for name in fact[1:]))
            if state_name == "initial_state":
                problem.set_initial_value(expression, value)
            else:
                problem.add_goal(expression if value else Not(expression))

    return problem


# Run with: python problem_generator.py -v
# These tests use the built-in simulator; no external planner is required.
import unittest
from copy import deepcopy


class TestBlocksworldProblem(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "blocks": ["A", "B", "C"],
            "initial_state": {
                **{("ontable", b): True for b in ("A", "B", "C")},
                **{("clear", b): True for b in ("A", "B", "C")},
                ("handempty",): True,
            },
            "goal_state": {("on", "A", "B"): True, ("on", "B", "C"): True},
        }

    @staticmethod
    def fact(problem, predicate, *names):
        return problem.fluent(predicate)(*(problem.object(name) for name in names))

    def test_objects_defaults_and_explicit_initial_values(self):
        config = self.config()
        config["initial_state"][("clear", "C")] = False
        problem = create_blocksworld_problem(config)
        self.assertEqual({obj.name for obj in problem.all_objects}, {"A", "B", "C"})
        for fact, value in config["initial_state"].items():
            self.assertEqual(problem.initial_value(self.fact(problem, *fact)).bool_constant_value(), value)
        self.assertTrue(problem.initial_value(self.fact(problem, "holding", "A")).is_false())
        self.assertTrue(problem.initial_value(self.fact(problem, "on", "A", "B")).is_false())
        self.assertEqual(set(problem.goals), {self.fact(problem, *f) for f in config["goal_state"]})

    def test_custom_blocks_and_empty_states(self):
        for names in (["solo"], ["red", "blue"], ["a", "b", "c", "d"]):
            with self.subTest(names=names):
                problem = create_blocksworld_problem({"blocks": names, "initial_state": {}, "goal_state": {}})
                self.assertEqual({obj.name for obj in problem.all_objects}, set(names))
                self.assertEqual(len(problem.goals), 0)
                self.assertTrue(all(value.is_false() for value in problem.initial_values.values()))
                self.assertEqual({a.name for a in problem.actions}, {"pick_up", "put_down", "stack", "unstack"})

    def test_input_is_not_mutated_and_calls_are_independent(self):
        config = self.config()
        original = deepcopy(config)
        first = create_blocksworld_problem(config)
        config["initial_state"][("handempty",)] = False
        second = create_blocksworld_problem(config)
        self.assertTrue(first.initial_value(self.fact(first, "handempty")).is_true())
        self.assertTrue(second.initial_value(self.fact(second, "handempty")).is_false())
        config["initial_state"][("handempty",)] = True
        self.assertEqual(config, original)
        self.assertIsNot(first, second)

    def test_invalid_configuration(self):
        for value in (None, [], "config", 42):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "config must"):
                create_blocksworld_problem(value)
        for key in ("blocks", "initial_state", "goal_state"):
            config = self.config()
            del config[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "Missing required key"):
                create_blocksworld_problem(config)
        for names in (None, [], "A", ("A",), ["A", "A"], [""], [1], [[]]):
            config = self.config()
            config["blocks"] = names
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "blocks must"):
                create_blocksworld_problem(config)

    def test_invalid_states_and_facts(self):
        bad_facts = [
            ({"clear": True}, "Invalid fact"),
            ({(): True}, "Invalid fact"),
            ({(1,): True}, "Invalid fact"),
            ({("unknown", "A"): True}, "Invalid fact"),
            ({("on", "A"): True}, "expects 2"),
            ({("clear", "A", "B"): True}, "expects 1"),
            ({("handempty", "A"): True}, "expects 0"),
            ({("clear", "missing"): True}, "Unknown block"),
            ({("clear", 1): True}, "Unknown block"),
        ]
        for value in (0, 1, None, "True", []):
            bad_facts.append(({("clear", "A"): value}, "must be booleans"))
        for state_name in ("initial_state", "goal_state"):
            for state, message in [(None, "must be a dictionary"), ([], "must be a dictionary")] + bad_facts:
                config = self.config()
                config[state_name] = state
                with self.subTest(state_name=state_name, state=state):
                    with self.assertRaisesRegex(ValueError, message):
                        create_blocksworld_problem(config)

    def test_all_predicates_support_positive_and_negative_goals(self):
        facts = [("on", "A", "B"), ("ontable", "A"), ("clear", "A"), ("holding", "A"), ("handempty",)]
        for fact in facts:
            for value in (True, False):
                with self.subTest(fact=fact, value=value):
                    config = self.config()
                    config["goal_state"] = {fact: value}
                    problem = create_blocksworld_problem(config)
                    expression = self.fact(problem, *fact)
                    self.assertEqual(problem.goals, [expression if value else Not(expression)])

    def test_build_tower_then_unstack_and_put_down(self):
        problem = create_blocksworld_problem(self.config())
        with SequentialSimulator(problem) as simulator:
            state = simulator.get_initial_state()
            self.assertFalse(simulator.is_goal(state))
            steps = [
                ("pick_up", ("B",), {("holding", "B"): True, ("handempty",): False, ("ontable", "B"): False, ("clear", "B"): False}),
                ("stack", ("B", "C"), {("on", "B", "C"): True, ("clear", "C"): False, ("clear", "B"): True, ("holding", "B"): False, ("handempty",): True}),
                ("pick_up", ("A",), {("holding", "A"): True}),
                ("stack", ("A", "B"), {("on", "A", "B"): True}),
                ("unstack", ("A", "B"), {("on", "A", "B"): False, ("clear", "B"): True, ("clear", "A"): False, ("holding", "A"): True, ("handempty",): False}),
                ("put_down", ("A",), {("holding", "A"): False, ("ontable", "A"): True, ("clear", "A"): True, ("handempty",): True}),
            ]
            for index, (name, args, expected) in enumerate(steps):
                with self.subTest(action=name, args=args):
                    action = problem.action(name)
                    parameters = [problem.object(arg) for arg in args]
                    self.assertTrue(simulator.is_applicable(state, action, parameters))
                    state = simulator.apply(state, action, parameters)
                    self.assertIsNotNone(state)
                    for fact, value in expected.items():
                        self.assertEqual(state.get_value(self.fact(problem, *fact)).bool_constant_value(), value)
                    self.assertEqual(simulator.is_goal(state), index == 3)

    def test_inapplicable_actions(self):
        problem = create_blocksworld_problem(self.config())
        with SequentialSimulator(problem) as simulator:
            state = simulator.get_initial_state()
            for name, args in [("put_down", ["A"]), ("stack", ["A", "B"]), ("unstack", ["A", "B"])]:
                with self.subTest(action=name):
                    self.assertFalse(simulator.is_applicable(state, problem.action(name), [problem.object(a) for a in args]))
            state = simulator.apply(state, problem.action("pick_up"), [problem.object("A")])
            self.assertFalse(simulator.is_applicable(state, problem.action("pick_up"), [problem.object("B")]))
            self.assertFalse(simulator.is_applicable(state, problem.action("stack"), [problem.object("A"), problem.object("A")]))

    def test_initial_stack_and_negative_goal(self):
        config = self.config()
        config["initial_state"] = {("on", "A", "B"): True, ("ontable", "B"): True, ("clear", "A"): True, ("handempty",): True}
        config["goal_state"] = {("on", "A", "B"): False, ("holding", "A"): True, ("handempty",): False}
        problem = create_blocksworld_problem(config)
        with SequentialSimulator(problem) as simulator:
            state = simulator.get_initial_state()
            self.assertFalse(simulator.is_goal(state))
            self.assertFalse(simulator.is_applicable(state, problem.action("pick_up"), [problem.object("B")]))
            state = simulator.apply(state, problem.action("unstack"), [problem.object("A"), problem.object("B")])
            self.assertIsNotNone(state)
            self.assertTrue(simulator.is_goal(state))


if __name__ == "__main__":
    unittest.main()
