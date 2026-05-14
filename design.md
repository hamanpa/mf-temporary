Collection of design choices

**naming convention**
consistent variable names (e.g., use `rate` over `nu/activity`, `params` over `pars`).



# Design choice
- **Modularity:** Modules should be easily swappable (e.g., changing the MF model should not require changing the rest of the workflow).
- **Separation of Concerns (Plotting):** No computations inside plotting modules. Results generation, analysis, and plotting must be strictly separate.
- **Data Handling:** The runner/controller should not store data internally; data must be explicitly returned to or saved for the user.
- **Consistent User-Facing Units:** Units that the user interacts with (network params, results) should always be identical. Any necessary conversions must happen internally within modules.
- **Naming Conventions:** Use underscores instead of hyphens for attribute names (e.g., `my_attribute`, not `my-attribute`).
- **Code Style Standard:** Use standard Python conventions (PEP 8) and a consistent docstring style (e.g., Google or NumPy style) across the codebase.
- **Testing Framework:** Use a standard testing framework (e.g., `pytest`) to ensure updates do not break existing functionality.


YAGNI (You Aren't Gonna Need It) principle


- **The Open/Closed Principle (OCP)** 
states that software entities (classes, modules, functions) should be open for extension (you can add new behavior) but closed for modification (you don't have to alter existing code to add that behavior).
If a PhD student in your lab writes a custom simulator for a new type of hardware, with the match statement, they must edit your WorkflowRunner.py to add case "custom_hardware":. If you use an interface, they can just pass their custom class to your runner, and your runner will execute it blindly because it trusts the abstract class structure. Your core code remains untouched.


Single Responsibility Principle (SRP)
Single Responsibility Principle (they hold data and validate it, leaving the simulation logic to other classes).

Avoid "anti-patterns"
In software engineering, relying on the name of a variable to determine its physical properties is an anti-pattern known as "Stringly Typed Programming."

In software design, there is a golden rule: "Code should be open for extension, but closed for modification."
