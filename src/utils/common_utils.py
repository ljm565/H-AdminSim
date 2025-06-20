from utils import Information



def padded_int(n: int, total_digit_l: int = 3) -> str:
    """
    Convert an integer to a zero-padded string of length 2.

    Args:
        n (int): The integer to convert.
        total_digit_l (int): The total number of digits in the output string. Default is 3.

    Returns:
        str: The zero-padded string representation of the integer.
    """
    if n < 0:
        raise ValueError("Negative integers are not supported")
    if total_digit_l <= 0:
        raise ValueError("Total digit length must be a positive integer")
    
    return str(n).zfill(total_digit_l)



def to_dict(obj) -> dict:
    """
    Convert an object to a dictionary representation.

    Args:
        obj: The object to convert.

    Returns:
        dict: A dictionary representation of the object.
    """
    if isinstance(obj, Information):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    else:
        return obj
