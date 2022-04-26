#from ast import Call
from typing import TYPE_CHECKING, Callable, Union, List
import time
#import math
import json
#import sys
import logging
import ding.utils
import pandas as pd
from tensorboardX import SummaryWriter

if TYPE_CHECKING:
    from ding.framework import Parallel


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


class PostProcessMethod(object):

    def print_in_tabulate(data_dict: dict):
        pass


class DataAnalyzer(object):

    def config(
            self,
            file_path: str = None,
            tensorboard_path: str = None,
            online: bool = True,
            register_default_fn: bool = True
    ) -> "DataAnalyzer":
        self._file = None
        self._has_file_writer = False
        self._file_path = None
        if file_path:
            try:
                self._file = open(file_path, "a+")
                self._has_file_writer = True
                self._file_path = file_path
            except IOError:
                logging.error("Invalid file path.")

        self._tb_writer = None
        self._has_tb_writer = False
        self._tensorboard_path = None
        if tensorboard_path:
            try:
                self._tb_writer = SummaryWriter(tensorboard_path)
                self._has_tb_writer = True
            except IOError:
                logging.error("Invalid tensorboard file path.")

        self._in_parallel = False
        self._router = None
        self._has_data = False

        self._analysis_functions = []

        if online:
            self._has_data = True
            self._data = []

        if register_default_fn:
            self.register_analysis_function(ProcessFunctions.general_process_fun)

        return self

    def record(self, info: dict):
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

    def register_analysis_function(self, fn: Callable, variable_name: Union[str, List[str]] = None):
        variables_name = []
        if variable_name:
            if type(variable_name) is str:
                variable_name.append(variable_name)
            else:
                variables_name.extend(variable_name)

        self._analysis_functions.append((fn, variables_name))

    #process origin data
    def analyse(self, feature=None, tensorboard_step_key: Union[str, List[str]] = None, show_result: bool = True):
        results_data_frame = None
        if self._has_data:
            results = []

            current_data_frame = pd.DataFrame.from_dict(self._data)
            for fn, variables_name in self._analysis_functions:
                # fn(current_data_frame)(variables_name) -> data_frame
                if not variables_name:  # empty list means to deal with all variables
                    for key in current_data_frame.keys():
                        if key != '__time':
                            variables_name.append(key)

                mean_df = fn(current_data_frame)(variables_name)

                results.append(mean_df)

            if results:
                results_data_frame = pd.concat(results)

        if results_data_frame and show_result:
            logging.info(results_data_frame.to_string())

        if results_data_frame and tensorboard_step_key:
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

    #Show all origin data
    def show(self):
        if self._has_data:
            current_data_frame = pd.DataFrame.from_dict(self._data)
            logging.info(current_data_frame.to_string())

    def close(self):
        self._has_file_writer = False
        self._in_parallel = False
        if self._router:
            self._router.off("_DataAnalyzer_")
            self._router = None
        if self._file:
            self._file.close()
            self._file = None

    def parallel_config(self, router: "Parallel" = None, is_writer: bool = False) -> "DataAnalyzer":
        if router and router.is_active:
            self._in_parallel = True
            self._router = router
            self._has_file_writer = is_writer
            if is_writer:
                router.on("_DataAnalyzer_", self._on_data_center)
        return self

    def _on_data_center(self, info: object, *args, **kwargs):
        self.record(info)

    def __del__(self):
        self.close()


data_analyzer = DataAnalyzer()
