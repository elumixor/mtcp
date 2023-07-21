class DotDict(dict):
    """ Dot notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __repr__(self):
        """ Print as a yaml file """
        result = ""
        for key, value in self.items():
            value_str = str(value)
            if isinstance(value, dict):
                value_str = "\n  " + value_str.replace("\n", "\n  ")
            result += f"{key}: {value_str}\n"
        return result.strip()

    def __str__(self):
        return self.__repr__()


    @staticmethod
    def from_dict(data: dict):
        """ Recursively transforms a dictionary into a DotDict """
        if not isinstance(data, dict):
            return data
        else:
            return DotDict({key: DotDict.from_dict(data[key]) for key in data})