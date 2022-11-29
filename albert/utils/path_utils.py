from __future__ import absolute_import

import errno
import os
import re
import sys
from collections import MutableMapping
from typing import Any


def _isiterable(x: Any) -> bool:
    # checks if an object is iterable or not
    try:
        iter(x)
        return True
    except TypeError:
        return False
    except Exception as e:  # pylint: disable=unused-variable
        raise


def is_pathname_valid(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    Refer to
    https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    for further details.
    Basically, simply checking against the regex ([a-zA-Z0-9_.]+?/)+[a-zA-Z0-9_.]+ is not enough, especially
    if we want to support windows paths
    """
    # pylint: disable=line-too-long
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    ERROR_INVALID_NAME = 123
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = (
            os.environ.get("HOMEDRIVE", "C:")
            if sys.platform == "win32"
            else os.path.sep
        )
        assert os.path.isdir(root_dirname)  # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, "winerror"):
                    if exc.winerror == ERROR_INVALID_NAME:  # pylint: disable=no-member
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NULL character" indicating an invalid pathname.
    except TypeError as exc:  # pylint: disable=unused-variable
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.


def append_expdir_to_paths(x: Any, dirname: str) -> Any:
    if isinstance(x, str):
        if "/" not in x or re.match(r"</.*>", x):
            # this is a hack, but the only way
            # I can differentiate between paths
            # and general parameters. Note that
            # this fails for windows. ¯\_(ツ)_/¯
            return x
        elif is_pathname_valid(x):
            # paths starting with / are treated as absolute
            # paths. Hence we have to strip out the leading
            # /
            return os.path.join(dirname, x.lstrip("/"))
        # a general string. Return
        return x
    elif isinstance(x, list):
        # modify the list in place
        for ix in range(len(x)):  # pylint: disable=consider-using-enumerate
            x[ix] = append_expdir_to_paths(x[ix], dirname)
    elif isinstance(x, tuple):
        # tuples are immutable. The only option is to create
        # a new one
        return (append_expdir_to_paths(y, dirname) for y in x)
    elif isinstance(x, MutableMapping):
        # modify the dict, Params, OrderedDict etc. in place
        for key in x:
            x[key] = append_expdir_to_paths(x[key], dirname)
    elif _isiterable(x):
        # No idea how to handle this iterable. Raise Error
        raise RuntimeError("{0} found, but not handled".format(type(x)))
    # this is a constant, or a modified list/ dictionary. Return as is
    return x
