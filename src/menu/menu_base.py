"""
Base menu functionality for Takenoko
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging


class MenuItem:
    """Represents a single menu item"""

    def __init__(self, key: str, description: str, action, requires_confirm: bool = False):
        self.key = key
        self.description = description
        self.action = action
        self.requires_confirm = requires_confirm

    def execute(self, *args, **kwargs) -> bool:
        """Execute the menu item action"""
        try:
            # Check if the action is a bound method (doesn't need arguments)
            import inspect
            sig = inspect.signature(self.action)
            if len(sig.parameters) == 0:
                return self.action()
            else:
                return self.action(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error executing menu item '{self.description}': {e}")
            return False


class BaseMenu(ABC):
    """Base class for creating menus"""

    def __init__(self, title: str):
        self.title = title
        self.items: Dict[str, MenuItem] = {}
        self.logger = logging.getLogger(__name__)

    def add_item(self, key: str, description: str, action, requires_confirm: bool = False):
        """Add a menu item"""
        self.items[key] = MenuItem(key, description, action, requires_confirm)

    def display(self) -> str:
        """Display the menu and return user choice"""
        print(f"\n{'=' * 50}")
        print(f"{self.title}")
        print(f"{'=' * 50}")

        for key in sorted(self.items.keys()):
            item = self.items[key]
            print(f"{key}. {item.description}")

        print(f"{'=' * 50}")

        while True:
            choice = input(f"Enter your choice ({', '.join(sorted(self.items.keys()))}): ").strip()
            if choice in self.items:
                return choice
            else:
                print(f"Invalid choice. Please enter {', '.join(sorted(self.items.keys()))}.")

    def run(self, *args, **kwargs):
        """Run the menu loop"""
        while True:
            try:
                choice = self.display()
                item = self.items[choice]

                if item.requires_confirm:
                    confirm = input(f"Are you sure you want to '{item.description}'? (y/N): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        continue

                result = item.execute(*args, **kwargs)

                if not result:
                    self.logger.error(f"Menu item '{item.description}' failed")
                    input("Press Enter to continue...")

            except KeyboardInterrupt:
                print("\nExiting menu...")
                break
            except Exception as e:
                self.logger.exception(f"Unexpected error in menu: {e}")
                input("Press Enter to continue...")