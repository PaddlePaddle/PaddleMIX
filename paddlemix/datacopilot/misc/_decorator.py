# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import time
import warnings

__all__ = ["deprecated", "retry"]


def deprecated(message=""):
    def decorator(func):
        """
        This is a decorator which can be used to mark functions as deprecated.
        It will result in a warning being emitted when the function is used.

        :param message: extra message to submit.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(
                f"Call to deprecated function `{func.__name__}`. {message}", category=DeprecationWarning, stacklevel=2
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry(
    max_trials: int = 3,
    delay: float = 0.1,
    verbose: bool = True,
    suppress_exceptions: bool = True,
):
    """
    A decorator for retrying a function call with a specified delay in case of an exception.
    If the maximum number of attempts is reached, it will return None or the default return value if specified.
    It can also print verbose messages during the process based on the verbose parameter.

    :param max_trials: Maximum number of attempts before giving up.
    :param delay: Delay between attempts in seconds.
    :param verbose: If True, print the details of each attempt.
    :param suppress_exceptions: If True, suppress all exceptions and return None instead of raising.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            exceptions = []
            while attempts < max_trials:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    exceptions.append(e)
                    if verbose:
                        print(f"[{attempts}/{max_trials}] <{e}>. Retrying after {delay} seconds.")
                    time.sleep(delay)

            if suppress_exceptions:
                if verbose:
                    print(f"Function {func.__name__} failed after {max_trials} attempts with {exceptions}.")
                return None
            else:
                raise Exception(f"Function {func.__name__} failed after {max_trials} attempts with {exceptions}.")

        return wrapper

    return decorator
