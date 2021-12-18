from functools import partial
from tqdm.auto import tqdm
import jax
from jax.experimental import host_callback


def jqdm(total, num_intervals=100, fori=False, **kwargs):
    "Decorate a jax scan body function to include a TQDM progressbar."

    rate = total // num_intervals if total > num_intervals else 1
    remainder = total % rate

    bar = tqdm(range(total), **kwargs)

    def _update(arg, transform):
        bar.update(arg)

    def update_jqdm(iter_num):
        _ = jax.lax.cond(
            (iter_num % rate == 0),
            lambda _: host_callback.id_tap(_update, rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            iter_num == total - remainder,
            lambda _: host_callback.id_tap(_update, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == total - 1,
            lambda _: host_callback.id_tap(lambda *_: bar.close(), None, result=result),
            lambda _: result,
            operand=None,
        )

    def _jqdm_scan(func):
        def wrapper_body_fun(carry, x):
            try:
                iter_num, *_ = x
            except:
                iter_num = x
            update_jqdm(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_body_fun

    def _jqdm_fori(func):
        def wrapper_body_fun(iter_num, carry):
            update_jqdm(iter_num)
            result = func(iter_num, carry)
            return close_tqdm(result, iter_num)

        return wrapper_body_fun

    if fori:
        return _jqdm_fori

    return _jqdm_scan


jqdm_scan = partial(jqdm, fori=False)
jqdm_fori = partial(jqdm, fori=True)


def jscan(
    f, init, xs, length=None, reverse=False, unroll=1, num_intervals=100, **kwargs
):
    total = length if length else len(xs)
    fake_xs = jax.numpy.stack((jax.numpy.arange(total), xs), axis=1)
    fake_f = jqdm_scan(total, num_intervals=num_intervals, **kwargs)(f)

    return jax.lax.scan(
        fake_f, init, fake_xs, length=length, reverse=reverse, unroll=unroll
    )
