import os
import re
import subprocess
import sys

import tiktoken

from doc_comments_ai.constants import Language


def get_programming_language(file_extension: str) -> Language:
    """
    Returns the programming language based on the provided file extension.

    Args:
        file_extension (str): The file extension to determine the programming language of.

    Returns:
        Language: The programming language corresponding to the file extension. If the file extension is not found
        in the language mapping, returns Language.UNKNOWN.
    """
    language_mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".cjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".rs": Language.RUST,
        ".go": Language.GO,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".cs": Language.C_SHARP,
        ".hs": Language.HASKELL,
    }
    return language_mapping.get(file_extension, Language.UNKNOWN)


def get_file_extension(file_name: str) -> str:
    """
    Returns the extension of a file from its given name.

    Parameters:
        file_name (str): The name of the file.

    Returns:
        str: The extension of the file.

    """
    return os.path.splitext(file_name)[-1]


def write_code_snippet_to_file(
    file_path: str, original_code: str, modified_code: str
) -> None:
    """
    Writes a modified version of a code snippet to a file.

    Args:
        file_path (str): The path of the file to write to.
        original_code (str): The original code snippet to be replaced.
        modified_code (str): The modified code snippet.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
        start_pos = file_content.find(original_code)
        if start_pos != -1:
            end_pos = start_pos + len(original_code)
            indentation = file_content[:start_pos].split("\n")[-1]
            modeified_lines = modified_code.split("\n")
            first_line = modeified_lines.pop(0)
            indented_modified_lines = [indentation + line for line in modeified_lines]
            indented_modified_code = (
                first_line + "\n" + "\n".join(indented_modified_lines)
            )
            modified_content = (
                file_content[:start_pos]
                + indented_modified_code
                + file_content[end_pos:]
            )
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(modified_content)


def extract_content_from_markdown_code_block(markdown_code_block) -> str:
    """
    Extracts the content from a markdown code block.

    :param markdown_code_block: A string containing a markdown code block.
    :return: The content within the code block, with leading and trailing whitespace removed if found,
             or the original markdown code block if no match is found.
    """
    pattern = r"```(?:[a-zA-Z0-9]+)?\n(.*?)```"
    match = re.search(pattern, markdown_code_block, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return markdown_code_block.strip()


def get_bold_text(text):
    """
    Returns the provided text formatted to appear as bold when printed in the console.

    Args:
        text (str): The text to be formatted as bold.

    Returns:
        str: The formatted text, wrapped in ANSI escape sequences to achieve bold appearance.
    """
    return f"\033[01m{text}\033[0m"


def has_unstaged_changes(file):
    """
    Checks if the given file has any unstaged changes in the Git repository.

    Args:
        file (str): The name of the file to check for unstaged changes.

    Returns:
        bool: Returns True if the file has unstaged changes, otherwise returns False.
    """
    try:
        subprocess.check_output(["git", "diff", "--quiet", file])
        return False
    except subprocess.CalledProcessError:
        return True



