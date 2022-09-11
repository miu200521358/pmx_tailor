# -*- coding: utf-8 -*-
#
from datetime import datetime
from logging import root
import os
import sys
import pathlib
import numpy as np
import math

# このソースのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + "/../")
sys.path.append(str(current_dir) + "/../src/")

from mmd.PmxReader import PmxReader  # noqa
from mmd.VmdReader import VmdReader  # noqa
from mmd.VmdWriter import VmdWriter  # noqa
from mmd.PmxWriter import PmxWriter  # noqa
from mmd.PmxData import (
    PmxModel,
    Vertex,
    Material,
    Bone,
    Morph,
    DisplaySlot,
    RigidBody,
    Joint,
    Bdef1,
    Bdef2,
    Bdef4,
    RigidBodyParam,
    IkLink,
    Ik,
)  # noqa
from mmd.VmdData import (
    VmdMotion,
    VmdBoneFrame,
    VmdCameraFrame,
    VmdInfoIk,
    VmdLightFrame,
    VmdMorphFrame,
    VmdShadowFrame,
    VmdShowIkFrame,
)  # noqa
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4  # noqa
from module.MParams import BoneLinks  # noqa
from utils import MFileUtils  # noqa
from utils.MException import SizingException  # noqa
from utils.MLogger import MLogger  # noqa


MLogger.initialize(level=MLogger.DEBUG_INFO, is_file=True)
logger = MLogger(__name__, level=MLogger.DEBUG_INFO)


def exec():
    # model = PmxReader("D:\\MMD\\Blender\\スカート\\cloak.pmx", is_check=False, is_sizing=False).read_data()
    # model = PmxReader("D:\\MMD\\Blender\\スカート\\cloak.pmx", is_check=False, is_sizing=False).read_data()
    # model = PmxReader("D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Model\\艦隊これくしょん\\金剛改二1.2 つみだんご\\金剛改二(艤装なし).pmx", is_check=False, is_sizing=False).read_data()
    # model = PmxReader("D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Model\\艦隊これくしょん\\金剛改二1.2 つみだんご\\金剛改二(艤装なし)_左袖分離2_20211107_122553.pmx", is_check=False, is_sizing=False).read_data()
    # model = PmxReader("D:\\MMD\\MikuMikuDance_v926x64\\UserFile\\Model\\刀剣乱舞\\025_一期一振\\一期一振 ひわこ式 ver.2.0\\一期一振(ひわこ式) ver.2.0.pmx", is_check=False, is_sizing=False).read_data()
    # model = PmxReader("D:\\MMD\\Blender\\スカート\\_報告\\niki修正中\\日光一文字ver.1.0Tailorテスト版9_クリア.pmx", is_check=False, is_sizing=False).read_data()
    model = PmxReader(
        "E:\\MMD\\MikuMikuDance_v926x64\\Work\\202101_vroid\\_報告\\ワルサー\\リィン_材質名重複なし.pmx",
        is_check=False,
        is_sizing=False,
    ).read_data()

    new_file_path = os.path.join(
        MFileUtils.get_dir_path(model.path),
        "{0}_{1:%Y%m%d_%H%M%S}{2}".format(os.path.basename(model.path.split(".")[0]), datetime.now(), ".pmx"),
    )

    pmx_writer = PmxWriter()
    pmx_writer.write(model, new_file_path)

    logger.warning(f"出力終了: {new_file_path}")


if __name__ == "__main__":
    exec()
