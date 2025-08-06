try:
    from core.actions import Actions
except ImportError:
    print("Importing Actions from core.actions failed, trying relative import...")
    from .core.actions import Actions

import time

def test_elixir_counting():
    actions = Actions()
    
    print("Testing elixir counting with ADB...")
    while True:
        count = actions.count_elixir()
        print(f"Current elixir count: {count}")
        time.sleep(1)

if __name__ == "__main__":
    test_elixir_counting()
