import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'

# __file__这是一个特殊的变量，在Python脚本中表示当前文件的路径。它的值是一个字符串，包含从根目录到当前文件的完整路径。
# abspath 函数将相对路径转换为绝对路径。这里它会将 __file__ 的值转换为当前脚本文件的绝对路径。
# dirname 函数返回给定路径的目录部分，这里 __file__ 是 G:\newcv\CGNet\basicsr\models\__init__.py，
# 那么 osp.dirname(osp.abspath(__file__)) 将返回 G:\newcv\CGNet\basicsr\models
model_folder = osp.dirname(osp.abspath(__file__))


# scandir 函数用于扫描指定目录下的文件，并返回一个迭代器，其中包含指定目录下所有以 '_model.py' 结尾的文件名（不包含路径）。
# basename 函数返回路径的最后一部分（即文件名），这里 osp.splitext(v) 将返回一个元组，元组第一个元素是文件名（不含扩展名），第二个元素是扩展名（不含点）。
# 结果：model_filenames = [base_model, image_restoration_model]
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'basicsr.models.{file_name}')
    for file_name in model_filenames
]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:   # _model_modules是一个包含所有可能模型模块的列表
        # 使用 getattr 函数在每个模块中查找与 model_type 对应的类
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')
    #使用找到的模型model_cls，根据opt参数创建模型实例
    model = model_cls(opt)


    logger = get_root_logger()  # 获取根日志记录器
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
