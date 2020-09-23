import os


def setup_workspace_environment_variable():

    print(
        ""
        "######################################\n"
        "### WORKSPACE ENVIRONMENT VARIABLE ###\n"
        "######################################\n\n"
        ""
        "An environment variable is a variable stored in your command line interface (e.g. where you just ran the command "
        "`python3 welcome.py`). Environment variables are used by Python and installed packages to inform them where to "
        "look for certain files.\n\n"
        ""
        "PyAutoLens uses an environment variable called WORKSPACE to know where the `autolens_workspace` folder is located. "
        "This is used to locate config files and output results.\n"
        ""
    )

    workspace_path = "{}".format(os.path.dirname(os.path.realpath(__file__)))

    if "WORKSPACE" not in os.environ:

        print(
            "PyAutoLens has detected that your workspace is at the following location on you computer:\n"
        )
        print(f"WORKSPACE PATH: {workspace_path}\n")
        input(
            "Please press Enter to confirm this is the correct path. If it is not, please exit this script and set the "
            "WORKSPACE path manually following the instructions at the readthedocs page:\n\n"
            "https://pyautolens.readthedocs.io/en/latest/general/installation.html\n\n"
            ""
            "[Press Enter to continue]"
        )

        os.environ["WORKSPACE"] = workspace_path

    else:

        workspace_path = os.environ["WORKSPACE"]

        print(
            "PyAutoLens has detected an existing WORKSPACE environment variable at the following location on you computer:\n"
        )
        print(f"WORKSPACE PATH: {workspace_path}\n")
        input(
            "Please press Enter to confirm this is the correct path. If it is not, please exit this script and set the "
            "WORKSPACE path manually following the instructions at the readthedocs page:\n\n"
            "https://pyautolens.readthedocs.io/en/latest/general/installation.html\n\n"
            ""
            "[Press Enter to continue]"
        )


def setup_pythonpath_environment_variable():

    workspace_path = os.environ["WORKSPACE"]

    print(
        "\n"
        "#######################################\n"
        "### PYTHONPATH ENVIRONMENT VARIABLE ###\n"
        "#######################################\n\n"
        ""
        "A second environment variable, PYTHONPATH, is used by Python to tell it where to import modules "
        "and installed software packages.\n\n"
        ""
        "Throughout the autolens_workspace we will import modules from it, by using the command:\n\n"
        ""
        "`from autolens_workspace import *`\n\n"
        ""
        "We therefore need to add the path to the autolens_workspace to your PYTHONPATH.\n\n"
        "NOTE: The PYTHONPATH must include all folders UP TO the `autolens_workspace` folder, but not the "
        "`autolens_workspace` folder itself!\n"
    )

    pythonpath_path = workspace_path.rsplit("/", 1)[0]
    pythonpath = os.environ["PYTHONPATH"]

    if pythonpath not in pythonpath:

        print("PyAutoLens will now add the following path to your PYTHONPATH:\n\n")
        print(f"PYTHONPATH: {pythonpath_path}\n\n")
        input(
            "Please press Enter to confirm this is the correct path. If it is not, please exit this script and set the "
            "PYTHONPATH path manually following the instructions at the readthedocs page:\n\n"
            "https://pyautolens.readthedocs.io/en/latest/general/installation.html\n\n"
            ""
            "[Press Enter to continue]"
        )

    else:

        input(
            "PyAutoLens has detected your PYTHONPATH is already set up correctly.\n\n"
            ""
            "[Press Enter to continue]"
        )

    pythonpath = f"{pythonpath}:{pythonpath_path}"
    os.environ["PYTHONPATH"] = pythonpath

    print(
        "\n"
        "##################################\n"
        "### SETTING UP THE ENVIRONMENT ###\n"
        "##################################\n\n"
        ""
        "Environment variables are removed every time you close your command line window, meaning they may need to be set"
        "up again manually every time you wish to your PyAutoLens. You can set up both variables using the following "
        "command in your command line: \n\n"
        ""
        "export WORKSPACE=/path/to/autolens_workspace/"
        "export PYTHONPATH=$PYTHONPATH:/path/to\n\n"
        ""
        "Where `/path/to` should include the path up to the workspace location\n\n"
        "Alternatively, you can run the `setup_environment.py` script in your autolens_workspace.\n\n"
        "All example scripts will print the WORKSPACE environment variable when they are run. If this raises an error it "
        "means you need to reset your environment variables.\n\n"
        "You can set environment variables permanently by adding the following commands to your .bashrc file:\n\n"
        ""
        "export WORKSPACE=/path/to/autolens_workspace/\n"
        "export PYTHONPATH=$PYTHONPATH:/path/to\n\n"
        ""
        "Where `/path/to` should include the path up to the workspace location.\n"
    )


setup_workspace_environment_variable()
setup_pythonpath_environment_variable()
