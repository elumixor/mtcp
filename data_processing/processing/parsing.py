# This file is a crime against humanity.
# It required huge mental effort from GPT4 and me as well. And a lot of wasted hours.
# It can probably break at any moment so be careful if you are using it.
# On the other hand, it may just work fine.
import re


def count_parentheses(expr):
    return expr.count('(') - expr.count(')')


def is_enclosed_in_parentheses(expr):
    open_parentheses = 0
    for ch in expr:
        if ch == '(':
            open_parentheses += 1
        elif ch == ')':
            open_parentheses -= 1
        if open_parentheses == 0 and ch != expr[-1]:
            return False
    return open_parentheses == 0


def remove_outer_parentheses(expr):
    if expr.startswith('(') and expr.endswith(')') and is_enclosed_in_parentheses(expr):
        return remove_outer_parentheses(expr[1:-1])
    return expr


def find_main_operator(expr):
    precedence = {
        "||": 1,
        "&&": 2,
        "==": 3,
        "!=": 3,
        "<=": 3,
        ">=": 3,
        "<": 3,
        ">": 3,
        "!": 4,
        "+": 5,
        "-": 5,
        "*": 6,
        "/": 6
    }
    min_precedence = float("inf")
    main_operator = None
    open_parentheses = 0

    if (expr.startswith("abs$") and expr.endswith("$")) or (expr.startswith("fabs$") and expr.endswith("$")):
        return None

    for i in range(len(expr)):
        if expr[i] == '(':
            open_parentheses += 1
        elif expr[i] == ')':
            open_parentheses -= 1
        elif open_parentheses == 0:
            for op in precedence.keys():
                if expr.startswith(op, i):
                    if precedence[op] < min_precedence:
                        min_precedence = precedence[op]
                        main_operator = op
    return main_operator


def split_expr(expr, operator):
    operator_indices = [i for i in range(len(expr)) if expr.startswith(operator, i)]
    if not operator_indices:
        return expr, ""
    split_index = min([i for i in operator_indices if count_parentheses(expr[:i]) == 0])
    part1 = expr[:split_index].strip()
    part2 = expr[split_index + len(operator):].strip()
    return part1, part2


def print_expr(expr, indentation=""):
    expr = remove_outer_parentheses(expr.strip())

    # Before the printing, replace abs$XXX$ with abs(XXX) and fabs$XXX$ with fabs(XXX)
    expr_p = re.sub(r"abs\$(.*?)\$", r"abs(\1)", expr)
    expr_p = re.sub(r"fabs\$(.*?)\$", r"fabs(\1)", expr_p)

    operator = find_main_operator(expr)
    if operator is None:
        return expr_p

    part1, part2 = split_expr(expr, operator)

    if operator != "!":
        part1 = print_expr(part1, indentation + "  ")

    part2 = print_expr(part2, indentation + "  ")

    operator = "+" if operator == "||" else "*" if operator == "&&" else "~" if operator == "!" else operator

    if operator == "~":
        return f"({operator}({part2}))"

    return f"(({part1}) {operator} ({part2}))"


def fix_syntax(expr):
    # Replace all abs(XXX) with abs$XXX$ and fabs(XXX) with fabs$XXX$ to avoid confusion
    expr = re.sub(r"abs\((.*?)\)", r"abs$\1$", expr)
    return print_expr(expr)
