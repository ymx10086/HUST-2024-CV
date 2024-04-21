"""
下载MNIST数据集脚本
"""

import os
from pathlib import Path
import logging

import wget

logging.basicConfig(level=logging.INFO, format="%(message)s")


def download_minst(save_dir: str = None) -> bool:
    """下载MNIST数据集
    输入参数:
        save_dir: MNIST数据集的保存地址. 类型: 字符串.

    返回值:
        全部下载成功返回True, 否则返回False
    """
    
    save_dir = Path(save_dir)
    train_set_imgs_addr = save_dir / "train-images-idx3-ubyte.gz"
    train_set_labels_addr = save_dir / "train-labels-idx1-ubyte.gz"
    test_set_imgs_addr = save_dir / "t10k-images-idx3-ubyte.gz"
    test_set_labels_addr = save_dir / "t10k-labels-idx1-ubyte.gz"

    try:
        if not os.path.exists(train_set_imgs_addr):
            logging.info("下载train-images-idx3-ubyte.gz")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=str(train_set_imgs_addr))
            logging.info("\tdone.")
        else:
            logging.info("train-images-idx3-ubyte.gz已经存在.")

        if not os.path.exists(train_set_labels_addr):
            logging.info("下载train-labels-idx1-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", out=str(train_set_labels_addr))
            logging.info("\tdone.")
        else:
            logging.info("train-labels-idx1-ubyte.gz已经存在.")

        if not os.path.exists(test_set_imgs_addr):
            logging.info("下载t10k-images-idx3-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", out=str(test_set_imgs_addr))
            logging.info("\tdone.")
        else:
            logging.info("t10k-images-idx3-ubyte.gz已经存在.")

        if not os.path.exists(test_set_labels_addr):
            logging.info("下载t10k-labels-idx1-ubyte.gz.")
            filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", out=str(test_set_labels_addr))
            logging.info("\tdone.")
        else:
            logging.info("t10k-labels-idx1-ubyte.gz已经存在.")
        
    except:
        return False
    
    return True


if __name__ == "__main__":
	download_minst(save_dir="./")

