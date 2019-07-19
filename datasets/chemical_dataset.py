from typing import AnyStr, Any
import pandas as pd
import os
import json
import time


class ChemicalDataset:
    def __init__(self, chemical_name):
        self.chemical_name = chemical_name
        self.meta_info = {'chemical_name': chemical_name}
        self.meta_info_loaded = dict()
        self._dataframes = dict()
        self._data_dicts = dict()
        self.load_status = None

    def set_meta_info(self, key, value):
        #  type: (Any, Any) -> None
        self.meta_info[key] = value


    def save(self,
             directory,  # type: AnyStr
             ):
        # type (...) -> None
        """
        directory/
        Files:
            contents.json - dictionary filesnames and number of rows
            meta.json - dictionary of metadata
            one file for each key, value pair in dataframes
        :param directory:
        :return:
        """
        save_dir = os.path.join(directory, self.chemical_name)
        os.mkdir(save_dir)
        self.meta_info['save_time'] = time.time()
        with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
            f.write(json.dumps(self.meta_info, indent=1))
        contents = dict()
        contents['dataframes'] = dict()
        contents['datadicts'] = dict()
        for df_name, df in self._dataframes.items():
            contents['dataframes'][df_name] = df.shape
        for name, data in self._data_dicts.items():
            contents['datadicts'] = len(data)
        with open(os.path.join(save_dir, 'contents.json'), 'w') as f:
            f.write(json.dumps(contents, indent=1))
        for df_name, df in self._dataframes.items():
            df.to_csv(os.path.join(save_dir, '{}.csv'.format(df_name)), index=False)
        for name, data in self._data_dicts.items():
            with open(os.path.join(save_dir, '{}.json'.format(name)), 'w') as f:
                f.write(json.dumps(data, indent=1))

    def load(self,
             directory,  # type: AnyStr
             ):
        # type (...) -> None
        """

        :param directory:
        :return: status code
        """

        if len(self._dataframes) > 0:
            return -1
        load_dir = os.path.join(directory, self.chemical_name)
        with open(os.path.join(load_dir, 'contents.json'), 'r') as f:
            contents = json.loads(f.read())
        with open(os.path.join(load_dir, 'meta.json'), 'r') as f:
            meta_info = json.loads(f.read())
        if self.chemical_name != meta_info.get("chemical_name"):
            return -1
        self.meta_info_loaded = meta_info
        self.load_status = dict()
        self.load_status['dataframes'] = dict()
        #for name, shape in contents['dataframes'].items():
        for name, shape in contents.items():
            df = pd.read_csv(os.path.join(load_dir, '{}.csv'.format(name)))
            self.add_dataframe(name, df)
            self.load_status['dataframes'][name] = df.shape == shape

        self.load_status['datadicts'] = dict()
        for name, length in contents['datadicts'].items():
            with open(os.path.join(load_dir, '{}.json'.format(name)), 'r') as f:
                data = json.loads(f.read())
                self.add_dict(name, data)
                self.load_status['datadicts'][name] = len(data) == length
        return 0

    def get_dataframe(self,
                         name,  # type: AnyStr
                         ):
        """

        :param name:
        :return: status code
        """
        return self._dataframes.get(name)

    def add_dataframe(self,
                         name,  # type: AnyStr
                         dataframe,  #type: pd.DataFrame
                      ):
        """

        :param name:
        :param dataframe:
        :return:
        """
        if name in self._dataframes:
            return -1
        self._dataframes[name] = pd.DataFrame(dataframe)
        return 0

    def add_dict(self,
                 name,
                 data):
        if name in self._data_dicts:
            return -1
        self._data_dicts[name] = dict(data)


