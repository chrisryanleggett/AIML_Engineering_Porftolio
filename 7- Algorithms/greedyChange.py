"""
Finds the minimum number of coins and bills needed for change using a greedy algorithm, using all real US denominations.
This portfolio example demonstrates practical I/O handling and a classic greedy strategy for optimization.
Other developers can use this as a reference for implementing classic greedy algorithms or for similar situations where an optimal local choice leads to a global optimum—such as resource allocation or scheduling challenges.
"""

# The locally optimal greedy choice: always take as many of the largest denomination as possible at each step.
def change(money):
    # List of denominations the algorithm will use for making change, as (float, string) pairs.
    denominations = [
        (100.00, "$100 bill"),
        (50.00, "$50 bill"),
        (20.00, "$20 bill"),
        (10.00, "$10 bill"),
        (5.00, "$5 bill"),
        (1.00, "$1 bill"),
        (0.25, "quarter"),
        (0.10, "dime"),
        (0.05, "nickel"),
        (0.01, "penny")
    ]
    change_counter = 0
    # amount_left tracks how much remains where change needs to be issued
    amount_left = round(money, 2)  # round to the nearest cent to match real-world currency floats
    denomination_counts = []
    for coin, name in denominations:
        count = int(amount_left // coin)
        change_counter += count
        if count > 0:
            denomination_counts.append((name, count))
        amount_left = round(amount_left - count * coin, 2)
    # Printing denomination breakdown
    for name, count in denomination_counts:
        unit = name if count == 1 else (name + "s" if not name.endswith("y") else name[:-1] + "ies")
        print(f"{count} {unit}")
    return change_counter

# Handles user input/output : prompts user, parses input, prints results or error.
if __name__ == '__main__':
    try:
        user_input = input("Enter an amount in dollars and cents (e.g., 18.36): ")
        m = float(user_input)
        if m < 0:
            print("Amount cannot be negative.")
        else:
            print("Minimum number of coins and bills needed:", change(m))
    except ValueError:
        print("Invalid input. Please enter a valid numeric amount.")
