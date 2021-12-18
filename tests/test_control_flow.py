import jax
import jax.numpy as jnp

import jqdm


def test_scan_decorator():
    """
    Test of the use of decorator for scan.
    """

    n = 10000

    @jqdm.jqdm_scan(total=n, desc="Scan-decorator")
    def body_func(carry, x):
        carry += 1  # counter
        y = jnp.sin(x)
        return carry, y

    total, ys = jax.lax.scan(body_func, 0, jnp.arange(n))
    return total, ys


def test_scan_func1():
    """
    Test of the use of a custom replacement of scan.
    """

    n = 10000

    def body_func(carry, x):
        carry += 1  # counter
        y = jnp.sin(x)
        return carry, y

    total, ys = jqdm.jscan(body_func, 0, jnp.arange(n), desc="scan-func1")
    return total, ys


def test_fori_decorator():
    """
    Test of the use of decorator for fori.
    """

    n = 10000

    @jqdm.jqdm_fori(total=n, desc="Fori-decorator")
    def body_func(iter_num, carry):
        carry = jnp.sin(carry)  # counter
        return carry

    ys = jax.lax.fori_loop(0, n, body_func, init_val=0)
    return ys


if __name__ == "__main__":
    test_scan_decorator()
    test_scan_func1()
    test_fori_decorator()
