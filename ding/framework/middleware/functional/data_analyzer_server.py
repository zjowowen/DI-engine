from typing import TYPE_CHECKING, Callable
import time
import atexit
from easydict import EasyDict
from ding.framework import task
from ding.framework.parallel import Parallel
from ding.utils import data_analyzer
if TYPE_CHECKING:
    from ding.framework import Context


def data_analyzer_server(cfg: EasyDict, online_analyse: bool = False) -> Callable:
    """
    Overview:
        Middleware for data analyser server as a master node that is both effective in local or remote mode. \
    Arguments:
        - cfg (:obj:`EasyDict`): Task configuration dictionary.
        - online_analyse (:obj:`bool`): Whether to enable online analysis. 
    Returns:
        - _data_analyzer_server_main (:obj:`Callable`): The main function for data analyzer server.
    """

    file_path = "./" + str(cfg.exp_name) + "/data_analyzer/log.txt"

    data_analyzer.config(
        file_path=file_path,
        online=online_analyse,
        register_default_fn=online_analyse,
        router=Parallel(),
        is_writer=True
    )

    atexit.register(data_analyzer.close)

    def _data_analyzer_server_main(ctx: "Context") -> None:
        """
        Overview:
            Make online analysis if needed. \
                Listen and record data and save it offline. 
        Arguments:
            - ctx (:obj:`Context`): Context of task object.
        """

        time.sleep(0.001)
        if online_analyse:
            result = data_analyzer.analyse()
            #do something if needed for result.
            #print tabule
            #logging
            #...

        return

    return _data_analyzer_server_main
