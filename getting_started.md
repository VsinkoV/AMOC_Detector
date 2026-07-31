# Getting Started

These notes are a short description of why i have set up the repo in this way. 


------------------------------------------------------------------------

# Why is this code packaged?

Research code often starts life as a collection of scripts. That works well at first, but it quickly becomes difficult to organise as a project grows.

Packaging a project gives the code a clear structure. Instead of copying files between notebooks or editing Python's search path, you can simply import the functionality you need.

For example:

``` python
from changepoint_package import Detector
```

rather than importing individual files directly.

A package also makes it much easier for multiple people to work on the same project while keeping everyone's environment consistent.

------------------------------------------------------------------------

# Why create a virtual environment?

Different projects often require different versions of Python libraries.

A **virtual environment** is an isolated Python installation for one
project. Installing packages inside it does not affect other projects on
your computer.

It is good practice to create a new environment for every research
repository.

------------------------------------------------------------------------

# Creating an environment (macOS)

Open **Terminal**, navigate to the repository, and create a virtual
environment:

``` bash
python3 -m venv .venv
```

This creates a directory called `.venv` containing an isolated Python
installation.

Activate it using:

``` bash
source .venv/bin/activate
```

When activated, your terminal prompt will usually begin with `(.venv)`.

Whenever you return to the project, activate the environment before
doing any work.

To leave the environment, simply run:

``` bash
deactivate
```

------------------------------------------------------------------------

# Installing the project

Once the environment is active, install everything listed in `requirements.txt`:

type the following in a terminal, after activating your environment

``` bash
pip install -r requirements.txt
```

This installs both the external libraries (such as NumPy and Matplotlib) and the package itself.

------------------------------------------------------------------------

# What does "editable mode" mean?

You may notice that the first line of `requirements.txt` is

``` text
-e src/.
```

The `-e` stands for **editable**.

Normally, when you install a package, Python copies it into your environment. If you later change the source code, you must reinstall the package before Python sees those changes.

Editable mode behaves differently.

Instead of copying the code, Python creates a link to the source directory. That means:

-   you edit the code in the repository;
-   Python immediately uses the updated version;
-   there is no need to reinstall after every change.

This is especially useful for research projects, where the code changes frequently as ideas are tested.

------------------------------------------------------------------------

# Why do research repositories use this approach?

A packaged repository with a virtual environment has several advantages:
-   Imports are far easier
-   everyone imports the code in the same way;
-   changes are immediately available when using editable mode;
-   dependencies are recorded in one place (`requirements.txt`);
-   collaborators can reproduce the same software environment more
    easily.

Overall, it helps keep research code organised, reusable, and easier to maintain.

