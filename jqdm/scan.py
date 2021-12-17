from tqdm.auto import tqdm
import jax
from jax.experimental import host_callback


def jqdm_scan(total, desc=None, update_steps=10000):
    "Decorate a jax scan to include a TQDM progressbar."
    tqdm_bars = tqdm(range(total))

    if total > update_steps:
        print_rate = int(total / update_steps)
    else:
        print_rate = 1
    remainder = total % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars.set_description(desc, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars.update(arg)

    def _update(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != total - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == total - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars.close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == total - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _jqdm_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            # TODO(alonfnt): fix this so that the user has to do nothing
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _jqdm_scan
