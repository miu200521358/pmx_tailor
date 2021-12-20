# -*- coding: utf-8 -*-
#
from datetime import datetime
from logging import root
import os
import sys
import pathlib
import numpy as np
import math
import glob

# このソースのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + '/../')
sys.path.append(str(current_dir) + '/../src/')

from mmd.PmxReader import PmxReader # noqa
from mmd.VmdReader import VmdReader # noqa
from mmd.VmdWriter import VmdWriter # noqa
from mmd.PmxWriter import PmxWriter # noqa
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, RigidBodyParam, IkLink, Ik # noqa
from mmd.VmdData import VmdMotion, VmdBoneFrame, VmdCameraFrame, VmdInfoIk, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame # noqa
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from module.MParams import BoneLinks # noqa
from utils import MFileUtils # noqa
from utils.MException import SizingException # noqa
from utils.MLogger import MLogger # noqa


MLogger.initialize(level=MLogger.WARNING, is_file=True)
logger = MLogger(__name__, level=MLogger.WARNING)


def exec():
    for n, pmx_path in enumerate(glob.glob("D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Model\\刀剣乱舞\\**\\*.pmx", recursive=True)):
        reader = PmxReader(pmx_path, is_check=False, is_sizing=False)
        try:
            model = reader.read_data()
            if len(model.bones) > 0:
                for bone_name in model.bones:
                    if "袖" in bone_name:
                        logger.warning(f"袖: {bone_name}({model.bones[bone_name].index}), モデル: {pmx_path}")
                        break
        except Exception:
            print(f"エラーpath: {pmx_path}")
        
        if n % 20 == 0:
            logger.warning(f"[{n}] --------------------")

    logger.warning("検出終了")


if __name__ == '__main__':
    exec()
