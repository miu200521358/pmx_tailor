# -*- coding: utf-8 -*-
#
from datetime import datetime
import sys
import pathlib
import numpy as np
import math
from numpy.core.defchararray import center

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
from module.MOptions import MOptionsDataSet  # noqa
from module.MParams import BoneLinks  # noqa
from utils import MBezierUtils, MServiceUtils  # noqa
from utils.MException import SizingException  # noqa
from utils.MLogger import MLogger  # noqa


MLogger.initialize(level=MLogger.DEBUG_INFO, is_file=True)
logger = MLogger(__name__, level=MLogger.DEBUG_INFO)


def exec():
    model = PmxReader(
        "E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル0813/APmiku_normal_BDEF4_purge.pmx",
        is_check=False,
        is_sizing=False,
    ).read_data()
    model.name = f"初音ミク捩りあり"

    for node_name, bone_param in BONE_PAIRS.items():
        parent_index = model.bones[bone_param["parent"]].index

        position = MVector3D()
        bone = Bone(bone_param["name"], node_name, position, parent_index, 0, bone_param["flag"])
        if bone.name in model.bones:
            # 同一名が既にある場合、スルー（尻対策）
            continue

        if parent_index >= 0:
            if "arm_twist_" in node_name:
                factor = 0.25 if node_name[-2] == "1" else 0.75 if node_name[-2] == "3" else 0.5
                position = model.bones[f"{bone_param['name'][0]}腕"].position + (
                    (
                        model.bones[f"{bone_param['name'][0]}ひじ"].position
                        - model.bones[f"{bone_param['name'][0]}腕"].position
                    )
                    * factor
                )
                if node_name[-2] in ["1", "2", "3"]:
                    bone.effect_index = model.bones[f"{bone_param['name'][0]}腕捩"].index
                    bone.effect_factor = factor
            elif "wrist_twist_" in node_name:
                factor = 0.25 if node_name[-2] == "1" else 0.75 if node_name[-2] == "3" else 0.5
                position = model.bones[f"{bone_param['name'][0]}ひじ"].position + (
                    (
                        model.bones[f"{bone_param['name'][0]}手首"].position
                        - model.bones[f"{bone_param['name'][0]}ひじ"].position
                    )
                    * factor
                )
                if node_name[-2] in ["1", "2", "3"]:
                    bone.effect_index = model.bones[f"{bone_param['name'][0]}手捩"].index
                    bone.effect_factor = factor

        bone.position = position
        bone.index = len(model.bones)
        model.bones[bone.name] = bone

    mats = {}
    for from_name, to_name in [
        ("左首", "左肩"),
        ("左肩", "左腕"),
        ("左腕", "左ひじ"),
        ("左ひじ", "左手首"),
        ("右首", "右肩"),
        ("右肩", "右腕"),
        ("右腕", "右ひじ"),
        ("右ひじ", "右手首"),
    ]:
        bone_from_name = "首" if "首" in from_name else from_name
        # ボーン進行方向(x)
        x_direction_pos = (model.bones[to_name].position - model.bones[bone_from_name].position).normalized()
        # ボーン進行方向に対しての横軸(y)
        y_direction_pos = MVector3D(1, 0, 0)
        # ボーン進行方向に対しての縦軸(z)
        z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
        qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)

        mat = MMatrix4x4()
        mat.setToIdentity()
        mat.translate(model.bones[bone_from_name].position)
        mat.rotate(qq)

        mats[from_name] = mat

    # for vertex in model.vertex_dict.values():
    #     if type(vertex.deform) is Bdef1:
    #         deform_bone_indexes = [vertex.deform.index0, 0, 0, 0]
    #         deform_weights = [1, 0, 0, 0]
    #     elif type(vertex.deform) is Bdef2:
    #         deform_bone_indexes = [vertex.deform.index0, vertex.deform.index1, 0, 0]
    #         deform_weights = [vertex.deform.weight0, 1 - vertex.deform.weight0, 0, 0]
    #     elif type(vertex.deform) is Bdef4:
    #         deform_bone_indexes = [
    #             vertex.deform.index0,
    #             vertex.deform.index1,
    #             vertex.deform.index2,
    #             vertex.deform.index3,
    #         ]
    #         deform_weights = [
    #             vertex.deform.weight0,
    #             vertex.deform.weight1,
    #             vertex.deform.weight2,
    #             vertex.deform.weight3,
    #         ]
    #     # まずは0じゃないデータ（何かしら有効なボーンINDEXがあるリスト）
    #     deform_bone_indexes = np.array(deform_bone_indexes)
    #     valiable_joints = np.where(deform_bone_indexes > 0)[0].tolist()
    #     deform_weights = np.array(deform_weights)

    #     for direction in ["右", "左"]:
    #         if not [
    #             bidx
    #             for bidx, w in zip(deform_bone_indexes, deform_weights)
    #             if bidx
    #             in [
    #                 model.bones[f"{direction}肩"].index,
    #                 model.bones[f"{direction}腕"].index,
    #                 model.bones[f"{direction}ひじ"].index,
    #                 model.bones[f"{direction}手首"].index,
    #             ]
    #             and w > 0.1
    #         ]:
    #             continue

    #         for base_from_name, base_to_name in [("首", "肩"), ("肩", "腕"), ("腕", "ひじ"), ("ひじ", "手首")]:
    #             from_bone_name = f"{direction}{base_from_name}" if base_from_name != "首" else base_from_name
    #             from_mat_name = f"{direction}{base_from_name}"
    #             to_bone_name = f"{direction}{base_to_name}"

    #             from_mat = mats[from_mat_name]
    #             from_bone_local_pos = from_mat.inverted() * model.bones[from_bone_name].position
    #             to_bone_local_pos = from_mat.inverted() * model.bones[to_bone_name].position
    #             vertex_local_pos = from_mat.inverted() * vertex.position

    #             if 0 < vertex_local_pos.y() <= to_bone_local_pos.y():
    #                 effector = (vertex_local_pos.y() - from_bone_local_pos.y()) / (
    #                     to_bone_local_pos.y() - from_bone_local_pos.y()
    #                 )

    #                 # 元のINDEXとウェイトをクリア
    #                 deform_weights[deform_bone_indexes == model.bones[from_bone_name].index] = 0
    #                 deform_weights[deform_bone_indexes == model.bones[to_bone_name].index] = 0
    #                 deform_bone_indexes[deform_bone_indexes == model.bones[from_bone_name].index] = 0
    #                 deform_bone_indexes[deform_bone_indexes == model.bones[to_bone_name].index] = 0

    #                 deform_bone_indexes = np.append(
    #                     deform_bone_indexes, [model.bones[from_bone_name].index, model.bones[to_bone_name].index]
    #                 )
    #                 deform_weights = np.append(deform_weights, [1 - effector, effector])

    #                 # 載せ替えた事で、ジョイントが重複している場合があるので、調整する
    #                 joint_weights = {}
    #                 for j, w in zip(deform_bone_indexes, deform_weights):
    #                     if j not in joint_weights:
    #                         joint_weights[j] = 0
    #                     joint_weights[j] += w

    #                 # 対象となるウェイト値
    #                 joint_values = np.array(list(joint_weights.keys()) + [0, 0, 0, 0])
    #                 # 正規化(合計して1になるように)
    #                 total_weights = np.array(list(joint_weights.values()) + [0, 0, 0, 0])
    #                 weight_values = total_weights / total_weights.sum(axis=0, keepdims=1)

    #                 total_joints = joint_values[np.argsort(-weight_values)[:4]]
    #                 total_weights = weight_values[np.argsort(-weight_values)[:4]]
    #                 total_weights = total_weights / total_weights.sum(axis=0, keepdims=1)

    #                 if np.count_nonzero(total_joints) == 1:
    #                     vertex.deform = Bdef1(total_joints[0])
    #                 elif np.count_nonzero(total_joints) == 2:
    #                     vertex.deform = Bdef2(total_joints[0], total_joints[1], total_weights[0])
    #                 else:
    #                     vertex.deform = Bdef4(
    #                         total_joints[0],
    #                         total_joints[1],
    #                         total_joints[2],
    #                         total_joints[3],
    #                         total_weights[0],
    #                         total_weights[1],
    #                         total_weights[2],
    #                         total_weights[3],
    #                     )

    #                 logger.info(
    #                     "[%s] from: %s, to: %s, effector: %s, deform: %s",
    #                     vertex.index,
    #                     from_bone_name,
    #                     to_bone_name,
    #                     effector,
    #                     vertex.deform,
    #                 )

    #                 break

    for vertex in model.vertex_dict.values():
        if type(vertex.deform) is Bdef1:
            deform_bone_indexes = [vertex.deform.index0, 0, 0, 0]
            deform_weights = [1, 0, 0, 0]
        elif type(vertex.deform) is Bdef2:
            deform_bone_indexes = [vertex.deform.index0, vertex.deform.index1, 0, 0]
            deform_weights = [vertex.deform.weight0, 1 - vertex.deform.weight0, 0, 0]
        elif type(vertex.deform) is Bdef4:
            deform_bone_indexes = [
                vertex.deform.index0,
                vertex.deform.index1,
                vertex.deform.index2,
                vertex.deform.index3,
            ]
            deform_weights = [
                vertex.deform.weight0,
                vertex.deform.weight1,
                vertex.deform.weight2,
                vertex.deform.weight3,
            ]
        # まずは0じゃないデータ（何かしら有効なボーンINDEXがあるリスト）
        deform_bone_indexes = np.array(deform_bone_indexes)
        valiable_joints = np.where(deform_bone_indexes > 0)[0].tolist()
        deform_weights = np.array(deform_weights)

        for direction in ["右", "左"]:
            if not [
                bidx
                for bidx, w in zip(deform_bone_indexes, deform_weights)
                if bidx
                in [
                    model.bones[f"{direction}肩"].index,
                    model.bones[f"{direction}腕"].index,
                    model.bones[f"{direction}ひじ"].index,
                    model.bones[f"{direction}手首"].index,
                ]
                and w > 0.1
            ]:
                continue

            for base_from_name, base_to_name, base_twist_name in [("腕", "ひじ", "腕捩"), ("ひじ", "手首", "手捩")]:
                dest_arm_bone_name = f"{direction}{base_from_name}"
                dest_elbow_bone_name = f"{direction}{base_to_name}"
                dest_arm_twist1_bone_name = f"{direction}{base_twist_name}1"
                dest_arm_twist2_bone_name = f"{direction}{base_twist_name}2"
                dest_arm_twist3_bone_name = f"{direction}{base_twist_name}3"

                arm_elbow_distance = -1
                vector_arm_distance = 1

                arm_mat = mats[dest_arm_bone_name]
                dest_arm_bone_local_pos = arm_mat.inverted() * model.bones[dest_arm_bone_name].position
                dest_elbow_bone_local_pos = arm_mat.inverted() * model.bones[dest_elbow_bone_name].position
                verte_local_pos = arm_mat.inverted() * vertex.position

                # 腕捩に分散する
                if (
                    model.bones[dest_arm_bone_name].index in deform_bone_indexes
                    or model.bones[dest_arm_twist1_bone_name].index in deform_bone_indexes
                    or model.bones[dest_arm_twist2_bone_name].index in deform_bone_indexes
                    or model.bones[dest_arm_twist3_bone_name].index in deform_bone_indexes
                ):
                    # 腕に割り当てられているウェイトの場合
                    arm_elbow_distance = dest_elbow_bone_local_pos.y() - dest_arm_bone_local_pos.y()
                    vector_arm_distance = verte_local_pos.y() - dest_arm_bone_local_pos.y()
                    twist_list = [
                        (dest_arm_twist1_bone_name, dest_arm_bone_name),
                        (dest_arm_twist2_bone_name, dest_arm_twist1_bone_name),
                        (dest_arm_twist3_bone_name, dest_arm_twist2_bone_name),
                    ]

                if np.sign(arm_elbow_distance) == np.sign(vector_arm_distance):
                    for dest_to_bone_name, dest_from_bone_name in twist_list:
                        dest_from_bone_local_pos = arm_mat.inverted() * model.bones[dest_from_bone_name].position
                        dest_to_bone_local_pos = arm_mat.inverted() * model.bones[dest_to_bone_name].position

                        # 腕からひじの間の頂点の場合
                        twist_distance = dest_to_bone_local_pos.y() - dest_from_bone_local_pos.y()
                        vector_distance = verte_local_pos.y() - dest_from_bone_local_pos.y()
                        if np.sign(twist_distance) == np.sign(vector_distance):
                            # 腕から腕捩1の間にある頂点の場合
                            arm_twist_factor = vector_distance / twist_distance
                            # 腕が割り当てられているウェイトINDEX
                            arm_twist_weight_joints = np.where(
                                deform_bone_indexes == model.bones[dest_from_bone_name].index
                            )[0]
                            if len(arm_twist_weight_joints) > 0:
                                if arm_twist_factor > 1:
                                    # 範囲より先の場合
                                    deform_bone_indexes[arm_twist_weight_joints] = model.bones[dest_to_bone_name].index
                                else:
                                    # 腕のウェイト値
                                    dest_arm_weight = deform_weights[arm_twist_weight_joints]
                                    # 腕捩のウェイトはウェイト値の指定割合
                                    arm_twist_weights = dest_arm_weight * arm_twist_factor
                                    # 腕のウェイト値は残り
                                    arm_weights = dest_arm_weight * (1 - arm_twist_factor)

                                    # FROMのウェイトを載せ替える
                                    valiable_joints = valiable_joints + [model.bones[dest_from_bone_name].index]
                                    deform_bone_indexes[arm_twist_weight_joints] = model.bones[
                                        dest_from_bone_name
                                    ].index
                                    deform_weights[arm_twist_weight_joints] = arm_weights
                                    # 腕捩のウェイトを追加する
                                    valiable_joints = valiable_joints + [model.bones[dest_to_bone_name].index]
                                    deform_bone_indexes = np.append(
                                        deform_bone_indexes, model.bones[dest_to_bone_name].index
                                    )
                                    deform_weights = np.append(deform_weights, arm_twist_weights)

                                    # 載せ替えた事で、ジョイントが重複している場合があるので、調整する
                                    joint_weights = {}
                                    for j, w in zip(deform_bone_indexes, deform_weights):
                                        if j not in joint_weights:
                                            joint_weights[j] = 0
                                        joint_weights[j] += w

                                    # 対象となるウェイト値
                                    joint_values = np.array(list(joint_weights.keys()) + [0, 0, 0, 0])
                                    # 正規化(合計して1になるように)
                                    total_weights = np.array(list(joint_weights.values()) + [0, 0, 0, 0])
                                    weight_values = total_weights / total_weights.sum(axis=0, keepdims=1)

                                    total_joints = joint_values[np.argsort(-weight_values)[:4]]
                                    total_weights = weight_values[np.argsort(-weight_values)[:4]]
                                    total_weights = total_weights / total_weights.sum(axis=0, keepdims=1)

                                    if np.count_nonzero(total_joints) == 1:
                                        vertex.deform = Bdef1(total_joints[0])
                                    elif np.count_nonzero(total_joints) == 2:
                                        vertex.deform = Bdef2(total_joints[0], total_joints[1], total_weights[0])
                                    else:
                                        vertex.deform = Bdef4(
                                            total_joints[0],
                                            total_joints[1],
                                            total_joints[2],
                                            total_joints[3],
                                            total_weights[0],
                                            total_weights[1],
                                            total_weights[2],
                                            total_weights[3],
                                        )

                                    logger.info(
                                        "[%s] from: %s, to: %s, factor: %s, dest_joints: %s, org_weights: %s, deform: %s",
                                        vertex.index,
                                        dest_from_bone_name,
                                        dest_to_bone_name,
                                        arm_twist_factor,
                                        deform_bone_indexes,
                                        deform_weights,
                                        vertex.deform,
                                    )

    pmx_writer = PmxWriter()
    pmx_writer.write(
        model, "E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル0813/APmiku_normal_BDEF4_purge_twist.pmx"
    )


BONE_PAIRS = {
    "arm_twist_L": {
        "name": "左腕捩",
        "parent": "左腕",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800 | 0x0800,
    },
    "arm_twist_1L": {
        "name": "左腕捩1",
        "parent": "左腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_2L": {
        "name": "左腕捩2",
        "parent": "左腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_3L": {
        "name": "左腕捩3",
        "parent": "左腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_L": {
        "name": "左手捩",
        "parent": "左ひじ",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "wrist_twist_1L": {
        "name": "左手捩1",
        "parent": "左ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_2L": {
        "name": "左手捩2",
        "parent": "左ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_3L": {
        "name": "左手捩3",
        "parent": "左ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_R": {
        "name": "右腕捩",
        "parent": "右腕",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "arm_twist_1R": {
        "name": "右腕捩1",
        "parent": "右腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_2R": {
        "name": "右腕捩2",
        "parent": "右腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_3R": {
        "name": "右腕捩3",
        "parent": "右腕",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_R": {
        "name": "右手捩",
        "parent": "右ひじ",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "wrist_twist_1R": {
        "name": "右手捩1",
        "parent": "右ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_2R": {
        "name": "右手捩2",
        "parent": "右ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_3R": {
        "name": "右手捩3",
        "parent": "右ひじ",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
}

if __name__ == "__main__":
    exec()
