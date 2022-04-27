#from ast import Call
from typing import TYPE_CHECKING, Callable, Union, List
import time
import os
#import math
import json
#import sys
import logging
import ding.utils
import pandas as pd
from pandas import DataFrame
from tensorboardX import SummaryWriter

if TYPE_CHECKING:
    from ding.framework import Parallel


# To do
class ProcessFunctions(object):

    def general_process_fun(data_frame):

        fn_list = ["", "mean", "max", "min", "std"]

        def _general_process_fun(variables_name: str):
            overall_results = []
            numeric_data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
            for process_fn_name in fn_list:
                results = {}
                if variables_name:
                    for var_name in variables_name:
                        if process_fn_name:
                            process_fn = getattr(numeric_data_frame[var_name], process_fn_name)
                            results[var_name] = process_fn()
                        else:
                            #get last valid value of this variable
                            last_valid_index = numeric_data_frame[var_name].last_valid_index()
                            if last_valid_index:
                                results[var_name] = numeric_data_frame[var_name].iloc[last_valid_index]
                            else:
                                pass
                    results_in_data_frame = pd.DataFrame.from_dict([results]).rename(index={0: process_fn_name})

                    overall_results.append(results_in_data_frame)

            return pd.concat(overall_results)

        return _general_process_fun


class DataAnalyzer(object):

    def config(
            self,
            file_path: str = None,
            online: bool = False,
            tensorboard_path: str = None,
            register_default_fn: bool = False,
            router: "Parallel" = None
    ) -> "DataAnalyzer":
        """
        Overview:
            Enable and change the configuration of a DataAnalyzer instance. 
        Arguments:
            - file_path (:obj:`str`): File path that offline data are to be saved at. \
                In local mode, it must be set sucessfully for at least once. \
                In remote mode, no need to set it for worker, but is necessary for master. \
            - online (:obj:`bool`): To enable online analysis and maintain data in memory
            - tensorboard_path (:obj:`str`): Set tensorboard file path , effective when online is enabled.
            - register_default_fn (:obj:`bool`): To register default analysis functions, \
                including [default_latest_value, "mean", "max", "min", "std"]
            - router (:obj:`Parallel`): To enable remote mode.
        """

        if not hasattr(self, "_file") or self._file is None:
            self._file = None
            self._has_file_writer = False
            self._file_path = None
            if file_path:
                try:
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    self._file = open(file_path, "a+")
                    self._has_file_writer = True
                    self._file_path = file_path
                except IOError as error:
                    logging.error("Invalid file path.")
                    raise

        if not hasattr(self, "_router") or self._router is None:
            self._in_parallel = False
            self._router = None
            if router and router.is_active:
                self._in_parallel = True
                self._router = router
                if _has_file_writer:
                    router.on("_DataAnalyzer_", self._on_data_center)

        if not hasattr(self, "_has_data") or self._has_data is False:
            self._has_data = False
            if online:
                self._has_data = True
                self._data = []

        if not hasattr(self, "_tb_writer") or self._tb_writer is None:
            self._tb_writer = None
            self._has_tb_writer = False
            self._tensorboard_path = None
            if online and tensorboard_path:
                try:
                    self._tb_writer = SummaryWriter(tensorboard_path)
                    self._has_tb_writer = True
                except IOError:
                    logging.error("Invalid tensorboard file path.")

        if not hasattr(self, "_analysis_functions") or len(self._analysis_functions) == 0:
            self._analysis_functions = []
            if register_default_fn:
                self.register_analysis_function(ProcessFunctions.general_process_fun)

        return self

    def record(self, info: dict) -> None:
        """
        Overview:
            Record information message of a dictionary form, and may save it in local file, or emit it to master, \
                or save it in memory for online analysis  
        Arguments:
            - info (:obj:`dict`): information message of dictionary form
        """
        dict_to_record = info
        if not "__time" in dict_to_record:
            dict_to_record.update({"__time": time.time()})

        msg = json.dumps(dict_to_record)
        if self._has_file_writer and self._file:
            self._file.write(msg + "\r\n")

        if self._in_parallel and not self._has_file_writer:
            self._router.emit("_DataAnalyzer_", dict_to_record)

        if self._has_data:
            self._data.append(dict_to_record)

    def register_analysis_function(self, fn: Callable, variable_name: Union[str, List[str]] = None) -> None:
        """
        Overview:
            Register an analysis function and provide the variable name of the input. \
                Those funtions would be called for every analyse steps.  
        Arguments:
            - fn (:obj:`Callable`): An analysis function that return n output variables. \
                The input of the analysis function is from the current data in the form of dataframe from pandas.
            - variable_name (:obj:` Union[str, List[str]]`): Provide the variable name of the input. \
                If is None, the variables in the whole dataframe will be given 
        """

        variables_name = []
        if variable_name:
            if type(variable_name) is str:
                variable_name.append(variable_name)
            else:
                variables_name.extend(variable_name)

        self._analysis_functions.append((fn, variables_name))

    def analyse(
            self,
            feature=None,
            tensorboard_step_key: Union[str, List[str]] = None,
            show_result: bool = True
    ) -> "DataFrame":
        """
        Overview:
            Analyse the current data in momery for every analysis function registered. \
                If no output, the return type is None. 
        Arguments:
            - feature (:obj:`str`): #To do# Adjust and limit the range of data that to be analysed.
            - tensorboard_step_key (:obj:`Union[str, List[str]]`): To record results with respect to every tb_step provided.
            - show_result (:obj:`bool`): Whether to log and show results. 
        """

        results_data_frame = None
        if self._has_data:
            if not self._data:
                return results_data_frame

            results = []

            current_data_frame = pd.DataFrame.from_dict(self._data)
            for fn, variables_name in self._analysis_functions:
                # fn(current_data_frame)(variables_name) -> data_frame
                if not variables_name:  # empty list means to deal with all variables
                    for key in current_data_frame.keys():
                        if key != '__time':
                            variables_name.append(key)

                results.append(fn(current_data_frame)(variables_name))

            if results:
                results_data_frame = pd.concat(results)

        if results_data_frame is not None and show_result:
            logging.info(results_data_frame.to_string())

        if results_data_frame is not None and tensorboard_step_key:
            tensorboard_step_key_list = []
            if type(tensorboard_step_key) is str:
                tensorboard_step_key_list.append(tensorboard_step_key)
            else:
                tensorboard_step_key_list.extend(tensorboard_step_key)

            if self._has_tb_writer:
                for step_key in tensorboard_step_key_list:
                    if not pd.isna(results_data_frame.loc["", step_key]):
                        for column_label in results_data_frame.columns:
                            for index, row_item in results_data_frame.iterrows():
                                if not pd.isna(row_item[column_label]):
                                    scalar_name = column_label
                                    if index:
                                        scalar_name = scalar_name + "/" + index
                                    self._tb_writer.add_scalar(
                                        scalar_name, row_item[column_label], results_data_frame.loc["", step_key]
                                    )

        return results_data_frame

    def show(self):
        """
        Overview:
            Simply display all data in the current memory.
        """
        if self._has_data:
            current_data_frame = pd.DataFrame.from_dict(self._data)
            logging.info(current_data_frame.to_string())

    def close(self):
        """
        Overview:
            Safely close the module.
        """
        self._has_file_writer = False
        self._in_parallel = False
        if self._router:
            self._router.off("_DataAnalyzer_")
            self._router = None
        if self._file:
            self._file.close()
            self._file = None

    def _on_data_center(self, info: object, *args, **kwargs):
        """
        Overview:
            Listen for RPC from non-master instance.
            *** Private method ***
        """
        self.record(info)

    def __del__(self):
        self.close()


data_analyzer = DataAnalyzer()
