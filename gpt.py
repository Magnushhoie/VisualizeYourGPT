import logging
import os
import re
import subprocess
import time
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

import numpy as np
import pandas as pd
from chatgpt_wrapper import ChatGPT
from chatgpt_wrapper.core.config import Config

from your_prompts import (code_for_script, prompt_introduction,
                          prompts_option_dict, responses_simulated_dict)

logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

# Enable plotting in terminal with itermplot (MacOS only)
os.environ["MPLBACKEND"] = "module://itermplot"
os.environ["ITERMPLOT"] = "rv"


def cmdline_args():
    # Make parser object
    usage = """\
    python gpt.py --csv_file <FILE> --model <MODEL>

    Examples:

    # Run VisualizeYourGPT on pre-processed dataset 'data.csv'
    python gpt.py --csv_file data/data.csv

    # Output notebook, insights and script stored in output/
    """
    p = ArgumentParser(
        description="Connect dataset and executable Python script to ChatGPT",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    p.add_argument(
        "--csv_file",
        default="data/data.csv",
        help="Input, pre-processed dataset (CSV file)",
    )
    p.add_argument(
        "--model",
        default="default",
        help="ChatGPT model to use (default = gpt-3.5 turbo), gpt-4, gpt4-32k",
    )

    p.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulate ChatGPT responses",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    return p.parse_args()


def _get_bot_response(bot, input_str, simulated_flag=False):
    """Get the bot response to the input."""

    # Print the input
    print(dedent(f"--------------------\nInput GPT\n--------------------\n{input_str}"))

    # Get the bot response
    if simulated_flag:
        time.sleep(0.2)
        choice = np.random.choice(list(responses_simulated_dict.keys()))
        success, response, message = True, responses_simulated_dict[choice], "Simulated"
    else:
        success, response, message = bot.ask(input_str)

    if success:
        print(
            dedent(
                f"--------------------\nOutput GPT\n--------------------\n{response}"
            )
        )
    else:
        log.error("ERROR: GPT server error: ", success, message)
        raise Exception

    return response


def pick_prompt_options(dataset_desc_str):
    """Enables user to pick prompt from list, or input their own"""

    # Print prompts (imported from from your_prompts.py)
    OPTIONS_LIST = [f"{k}: {v.splitlines()[0]}" for k, v in prompts_option_dict.items()]
    print("\n".join(OPTIONS_LIST))

    # User selects option or provides own prompt
    choice = input("\nPick a choice or enter your own prompt:\n")

    if choice not in prompts_option_dict:
        input_str = choice
    else:
        input_str = prompts_option_dict[choice]

    # Check for special actions (only replacing "USER_INPUT" for now)
    if "USER_INPUT" in input_str:
        user_input = input("Enter your own prompt:\n")
        input_str = input_str.replace("USER_INPUT", user_input)

    if "DATASET_DESC_INPUT" in input_str:
        input_str = input_str.replace("DATASET_DESC_INPUT", dataset_desc_str)

    return input_str


def record_model_output(bot, input_str, simulated=False):
    """Record the model output to a file."""

    # 1. Record input
    write_markdown_file("output/notebook.md", mode="append", string=input_str)

    # 2. Get bot response
    response = _get_bot_response(bot, input_str, simulated)
    text_block, code_block = extract_text_and_code_block(response)

    # (If empty, the bot failed and we should try again (once))
    if not text_block and not code_block:
        log.info("==== Invalid code block found, asking GPT to try again ====")
        input_str = "Please return only a single ```python code block, less than ~350 words, and nothing outside of the code block!"
        response = _get_bot_response(bot, input_str, simulated)
        text_block, code_block = extract_text_and_code_block(response)

    # Save results
    write_markdown_file("output/insights.txt", mode="append", string=text_block)
    write_markdown_file("output/notebook.md", mode="append", string=text_block)
    write_markdown_file(
        "output/notebook.md", mode="append", string=code_block, code_flag=True
    )

    # 3. Run code block if found
    if code_block:
        # Only write script to run once, in case needs to be manually edited
        write_python_file("output/script.py", mode="write", string=code_for_script)

        # Loop in case need to manually edit
        while True:
            # Run script
            write_python_file("output/script.py", mode="append", string=code_block)
            code_block = ""
            output_str = run_python_file("output/script.py")

            if len(output_str) > 1:
                input_str = input(
                    "==== Enter to send output to ChatGPT. 1. Manually edit and compute again or 0. Cancel ==== "
                )

                # Enter: Send output to GPT
                if input_str == "":
                    input_str = f"{output_str}\nSummarize the main findings in a list."
                    response = _get_bot_response(bot, input_str, simulated)
                    text_block, code_block = extract_text_and_code_block(response)
                    code_block = ""

                    # Save results
                    write_markdown_file(
                        "output/insights.txt", mode="append", string=text_block
                    )
                    write_markdown_file(
                        "output/notebook.md", mode="append", string=text_block
                    )
                    write_markdown_file(
                        "output/notebook.md",
                        mode="append",
                        string=code_block,
                        code_flag=True,
                    )
                    break

                # 1. Manually edit and compute again
                elif input_str == "1":
                    continue

                # 0. Cancel or continue
                else:
                    input_str = ""
                    break


def write_output(string, outfile="output/output.py"):
    """Write the output to a file."""

    with open(outfile, "w") as file:
        file.write(string)


def extract_text_and_code_block(input_string):
    """First extracts Python code block, then leaves text block as everything else"""

    def check_started_code_block(code):
        """Simple check for starting code block"""
        return code.count(r"```") >= 1

    def check_completed_code_block(code):
        """Simple check for completed 1x block"""
        return code.count(r"```") == 2

    pattern = r"```(?:\w+\n)?(.*?)\n```"
    code_match = re.search(pattern, input_string, re.DOTALL)
    if code_match:
        code_block = re.sub(r"```python\s*|\s*```", "", code_match.group(0)).strip()
        text_block = input_string.replace(code_match.group(0), "")
        log.info(f"==== Found Python code block ({len(code_block)}) ==== ")
    else:
        code_block = ""
        text_block = input_string

    # Check for non-ending code block
    if check_started_code_block(text_block) and not check_completed_code_block(
        text_block
    ):
        text_block = ""
        code_block = ""

    return text_block.strip(), code_block


def write_python_file(filename, mode, string=""):
    """Write a multi-line string to a file"""

    # Create new script and notebook files
    if mode == "write":
        # Start new script file, with imported libraries
        with open(filename, "w") as file:
            file.write(string)

    # Append to script and notebook files
    if mode == "append":
        string = f"\n{string}\n"

        # Write to script file
        with open(filename, "a") as file:
            file.write(string)


def write_markdown_file(filename, mode, string="", code_flag=False):
    """Write a multi-line string to a file"""

    # Create new script and notebook files
    if mode == "write":
        # Write to script file
        with open(filename, "w") as file:
            file.write(string)

    # Append to script and notebook files
    if mode == "append":
        if code_flag:
            string = f"```python\n{string}\n```"
        else:
            string = f"\n{string}\n"

        # Write to script file
        with open(filename, "a") as file:
            file.write(string)


def run_python_file(file):
    """Run a Python file."""

    user_input = input(
        f"==== WARNING: Press enter to run above code in {file} or 0. Cancel ==== "
    )
    if user_input == "0":
        return ""

    # Run the Python script and capture the output
    result = subprocess.run(
        ["python", file], stdout=subprocess.PIPE, env=dict(os.environ, DISPLAY=":0.0")
    )

    # Store output in string
    output_str = result.stdout.decode("utf-8")
    print(f"\n--------------------\nProgram output\n--------------------\n{output_str}")

    # Format nicely
    output_str = f"Program output:\n{output_str}"

    # Remove "figure" lines plotted by iTermplot
    output_str = remove_fig_lines(output_str)

    return output_str


def remove_fig_lines(string):
    """Removes iTerm2 plotted "figure" plot lines from a string."""
    pattern = r"]1337;File=name=c"
    return "\n".join(
        [line for line in string.split("\n") if not line.startswith(pattern)]
    )


def prepare_dataset_description(df_proc):
    """Prepare a description of the dataset."""

    # Add some formatting
    dataset_desc_str = dedent(
        f"""
Dataset (df_proc) description:

```text
df_proc.head():
{df_proc.head()}

df_proc.describe():
{df_proc.describe()}
```
"""
    )

    return dataset_desc_str


def main(args):
    """Main function for the GPT-4 chatbot."""

    # Simulate or not
    if args.simulate:
        simulate_flag = True
        bot = True
    else:
        simulate_flag = False
        config = Config()
        """
        OPENAPI_CHAT_RENDER_MODELS = {
            "default": "gpt-3.5-turbo",
            "turbo": "gpt-3.5-turbo",
            "turbo-0301": "gpt-3.5-turbo-0301",
            "gpt4": "gpt-4",
            "gpt4-0314": "gpt-4-0314",
            "gpt4-32k": "gpt-4-32k",
            "gpt4-32k-0314": "gpt-4-32k-0314",
        }
        """
        config.set("chat.model", args.model)
        bot = ChatGPT(config)

    # Reset files by writing to them
    os.makedirs("output/", exist_ok=True)
    write_python_file("output/script.py", mode="write", string=code_for_script)
    write_markdown_file(
        "output/notebook.md",
        mode="write",
        string=f"```python\n{code_for_script}\n```",
    )
    write_markdown_file("output/insights.txt", mode="write", string="# insights.txt")

    # Data is loaded fresh everytime in script.py, but loaded here for description (.head() and .describe())
    log.info(f"Reading dataset from {args.csv_file}")
    df_proc = pd.read_csv(args.csv_file)

    # First task gives data description and introduction from your_prompts.py
    dataset_desc_str = prepare_dataset_description(df_proc)
    input_str = f"\n{dataset_desc_str}\n{prompt_introduction}"

    # Loop over ChatGPT input -> code -> executino -> output
    # Nb: GPT-code written and executed from script.py and new output written to notebook.md
    stage_int = 1
    while True:
        log.info(f"\n\n==== Loop {stage_int} ====")

        if not input_str:
            input_str = pick_prompt_options(dataset_desc_str=dataset_desc_str)

        # Option for quitting
        if input_str.lower() == "quit":
            break

        # Send to ChatGPT, execute code if available and record output
        else:
            record_model_output(bot, input_str, simulate_flag)
            input_str = ""
            stage_int += 1

    # Finally, convert conversation from notebook.md to Jupyter notebook
    log.info("==== Done! Converting notebook.md to notebook.ipynb ====")
    os.system("jupytext --to notebook output/notebook.md output/notebook.ipynb")


if __name__ == "__main__":
    log.info("Running VisualizeGPT")

    args = cmdline_args()
    main(args)
