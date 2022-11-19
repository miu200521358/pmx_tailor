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


OUTPUT_BONE_NAMES = (
    "下半身",
    "上半身",
    "上半身2",
    "上半身3",
    "首",
    "頭",
    "左肩",
    "左腕",
    "左ひじ",
    "左手首",
    "左親指０",
    "左親指１",
    "左親指２",
    "左人指１",
    "左人指２",
    "左人指３",
    "左中指１",
    "左中指２",
    "左中指３",
    "左薬指１",
    "左薬指２",
    "左薬指３",
    "左小指１",
    "左小指２",
    "左小指３",
    "右肩",
    "右腕",
    "右ひじ",
    "右手首",
    "右親指０",
    "右親指１",
    "右親指２",
    "右人指１",
    "右人指２",
    "右人指３",
    "右中指１",
    "右中指２",
    "右中指３",
    "右薬指１",
    "右薬指２",
    "右薬指３",
    "右小指１",
    "右小指２",
    "右小指３",
    "左足",
    "左ひざ",
    "左足首",
    "右足",
    "右ひざ",
    "右足首",
)

# ローカルX軸の取得
def get_tail_position(model, bone_name: str):
    if bone_name not in model.bones:
        return MVector3D()

    bone = model.bones[bone_name]
    to_pos = MVector3D()

    from_pos = model.bones[bone.name].position
    if not bone.getConnectionFlag() and bone.tail_position > MVector3D():
        # 表示先が相対パスの場合、保持
        to_pos = from_pos + bone.tail_position
    elif bone.getConnectionFlag() and bone.tail_index >= 0 and bone.tail_index in model.bone_indexes:
        # 表示先が指定されているの場合、保持
        to_pos = model.bones[model.bone_indexes[bone.tail_index]].position
    else:
        # 表示先がない場合、とりあえず親ボーンからの向きにする
        from_pos = model.bones[model.bone_indexes[bone.parent_index]].position
        to_pos = model.bones[bone.name].position

    return to_pos


def exec():
    model = PmxReader(
        "C:/MMD/mmd-auto-trace-3/data/pmx/trace_check_model.pmx",
        is_check=False,
        is_sizing=False,
    ).read_data()

    WIDTH = 0.1
    NORMAL_VEC = MVector3D(0, 1, 0)

    material = model.materials["ボーン材質"]

    for bone_name in OUTPUT_BONE_NAMES:
        bone = model.bones[bone_name]
        from_pos = bone.position
        tail_pos = get_tail_position(model, bone_name)

        # FROMからTOまで面を生成
        v1 = Vertex(
            index=len(model.vertex_dict),
            position=from_pos,
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v1.index] = v1
        print(str(v1))

        v2 = Vertex(
            index=len(model.vertex_dict),
            position=tail_pos,
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v2.index] = v2
        print(str(v2))

        v3 = Vertex(
            index=len(model.vertex_dict),
            position=from_pos + MVector3D(WIDTH, 0, 0),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v3.index] = v3
        print(str(v3))

        v4 = Vertex(
            index=len(model.vertex_dict),
            position=tail_pos + MVector3D(WIDTH, 0, 0),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v4.index] = v4
        print(str(v4))

        v5 = Vertex(
            index=len(model.vertex_dict),
            position=from_pos + MVector3D(WIDTH, WIDTH, 0),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v5.index] = v5
        print(str(v5))

        v6 = Vertex(
            index=len(model.vertex_dict),
            position=tail_pos + MVector3D(WIDTH, WIDTH, 0),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v6.index] = v6
        print(str(v6))

        v7 = Vertex(
            index=len(model.vertex_dict),
            position=from_pos + MVector3D(0, 0, WIDTH),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v7.index] = v7
        print(str(v7))

        v8 = Vertex(
            index=len(model.vertex_dict),
            position=tail_pos + MVector3D(0, 0, WIDTH),
            normal=NORMAL_VEC,
            uv=MVector2D(),
            extended_uvs=[],
            deform=Bdef1(bone.index),
            edge_factor=0,
        )
        model.vertex_dict[v8.index] = v8
        print(str(v8))

        model.indices[len(model.indices)] = [v1.index, v2.index, v3.index]
        model.indices[len(model.indices)] = [v3.index, v2.index, v4.index]
        model.indices[len(model.indices)] = [v3.index, v4.index, v5.index]
        model.indices[len(model.indices)] = [v5.index, v4.index, v6.index]
        model.indices[len(model.indices)] = [v5.index, v6.index, v7.index]
        model.indices[len(model.indices)] = [v7.index, v6.index, v8.index]
        model.indices[len(model.indices)] = [v7.index, v8.index, v1.index]
        model.indices[len(model.indices)] = [v1.index, v8.index, v2.index]
        material.vertex_count = 3 * len(model.indices)

    new_file_path = os.path.join(
        MFileUtils.get_dir_path(model.path),
        "{0}_{1:%Y%m%d_%H%M%S}{2}".format(os.path.basename(model.path.split(".")[0]), datetime.now(), ".pmx"),
    )

    pmx_writer = PmxWriter()
    pmx_writer.write(model, new_file_path)

    logger.warning(f"出力終了: {new_file_path}")


if __name__ == "__main__":
    exec()
