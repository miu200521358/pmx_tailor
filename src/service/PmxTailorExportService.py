# -*- coding: utf-8 -*-
#
import logging
import os
import traceback
import numpy as np
import math
import copy
import bezier
import csv
import random
import string
from itertools import combinations
from collections import Counter
from glob import glob

from module.MOptions import MExportOptions
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
    Sdef,
    RigidBodyParam,
    IkLink,
    Ik,
    BoneMorphData,
)
from mmd.PmxWriter import PmxWriter
from module.MMath import (
    MVector2D,
    MVector3D,
    MVector4D,
    MQuaternion,
    MMatrix4x4,
    MPoint,
    MLine,
    MSegment,
    MSphere,
    MCapsule,
)
from utils.MLogger import MLogger
from utils.MException import SizingException, MKilledException
import utils.MBezierUtils as MBezierUtils

logger = MLogger(__name__, level=MLogger.DEBUG)


class VirtualVertex:
    def __init__(self, key):
        self.key = key
        # 実頂点
        self.real_vertices = []
        # 実頂点の位置リスト
        self.positions = []
        # 実頂点の法線リスト
        self.normals = []
        # 実頂点の面リスト（処理対象）
        self.indexes = []
        # 実頂点の面リスト（処理対象外）
        self.out_indexes = []
        # 実頂点の遷移先仮想頂点リスト
        self.connected_vvs = []
        # 対象頂点に対するウェイト情報
        self.deform = None
        # 対象頂点に対するボーン情報(map単位)
        self.map_bones = {}
        # 対象頂点に対する剛体情報(map単位)
        self.map_rigidbodies = {}
        self.map_balance_rigidbodies = {}

    def append(self, real_vertices: list, connected_vvs: list, indexes: list):
        for rv in real_vertices:
            if rv not in self.real_vertices:
                self.real_vertices.append(rv)
                self.positions.append(rv.position.data())
                self.normals.append(rv.normal.data())

        for lv in connected_vvs:
            if lv not in self.connected_vvs:
                self.connected_vvs.append(lv)

        for i in indexes:
            if i not in self.indexes:
                self.indexes.append(i)

    def vidxs(self):
        return [v.index for v in self.real_vertices]

    def position(self):
        if not self.positions:
            return MVector3D()
        return MVector3D(np.mean(self.positions, axis=0))

    def normal(self):
        if not self.normals:
            return MVector3D()
        return MVector3D(np.mean(self.normals, axis=0))

    def __str__(self):
        return f"v[{','.join([str(v.index) for v in self.real_vertices])}] pos[{self.position().to_log()}] nor[{self.normal().to_log()}], lines[{self.connected_vvs}], indexes[{self.indexes}], out_indexes[{self.out_indexes}]"


class PmxTailorExportService:
    def __init__(self, options: MExportOptions):
        self.options = options

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = f"{logger.transtext('PmxTailor変換処理実行')}\n------------------------\n{logger.transtext('exeバージョン')}: {self.options.version_name}\n"
            service_data_txt = (
                f"{service_data_txt}　{logger.transtext('元モデル')}: {os.path.basename(self.options.pmx_model.path)}\n"
            )

            for pidx, param_option in enumerate(self.options.param_options):
                service_data_txt = f"{service_data_txt}\n　【No.{pidx + 1}】 --------- "
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('材質')}: {param_option['material_name']}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('略称')}: {param_option['abb_name']}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('質量')}: {param_option['mass']}"
                service_data_txt = (
                    f"{service_data_txt}\n　　{logger.transtext('柔らかさ')}: {param_option['air_resistance']}"
                )
                service_data_txt = (
                    f"{service_data_txt}\n　　{logger.transtext('張り')}: {param_option['shape_maintenance']}"
                )

            logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

            model = self.options.pmx_model
            model.comment += f"\r\n\r\n{logger.transtext('物理')}: PmxTailor {self.options.version_name}"

            # 足に繋げるか否か
            if param_option["rigidbody_leg"] and not (
                "下半身" in model.bones and "上半身" in model.bones and "左足" in model.bones and "右足" in model.bones
            ):
                logger.warning(
                    "足に繋げるオプションがONになっていますが、上半身・下半身・左足・右足のいずれかのボーンがないため、処理を終了します", decoration=MLogger.DECORATION_BOX
                )
                return False

            # 既存物理を削除する
            is_overwrite = False
            is_reuse = False
            for pidx, param_option in enumerate(self.options.param_options):
                if param_option["exist_physics_clear"] == logger.transtext("上書き"):
                    is_overwrite = True
                if param_option["exist_physics_clear"] == logger.transtext("再利用"):
                    is_reuse = True

            # FIXME とりあえず再利用時の既存削除処理OFF（2022/08/07）
            if is_overwrite:
                model = self.clear_physics(model, is_overwrite)

                if not model:
                    return False

            all_virtual_vertices = []
            all_vertex_maps = []
            for pidx, param_option in enumerate(self.options.param_options):
                result, virtual_vertices, vertex_maps = self.create_physics(model, param_option)
                if not result:
                    return False
                all_virtual_vertices.append(virtual_vertices)
                all_vertex_maps.append(vertex_maps)

            for pidx, param_option in enumerate(self.options.param_options):
                if param_option.get("physics_parent", 0):
                    # 物理親設定がある場合
                    result = self.change_physics_parent(
                        model, param_option, all_virtual_vertices, all_vertex_maps, pidx
                    )
                    if not result:
                        return False

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            PmxWriter().write(model, self.options.output_path)

            logger.info(
                "出力終了: %s",
                os.path.basename(self.options.output_path),
                decoration=MLogger.DECORATION_BOX,
                title=logger.transtext("成功"),
            )

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("PmxTailor変換処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical(
                "PmxTailor変換処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX
            )
        finally:
            logging.shutdown()

    def clear_physics(self, model: PmxModel, is_overwrite: bool):
        logger.info("既存物理削除", decoration=MLogger.DECORATION_LINE)
        target_vertices = []

        for param_option in self.options.param_options:
            material_name = param_option["material_name"]

            # 物理対象頂点CSVが指定されている場合、対象頂点リスト生成
            if param_option["vertices_csv"]:
                target_vertices.extend(
                    read_vertices_from_file(
                        param_option["vertices_csv"], model, None if param_option["vertices_csv"] else material_name
                    )
                )
            else:
                target_vertices.extend(list(model.material_vertices[material_name]))

            if len(target_vertices) > 0 and len(target_vertices) % 1000 == 0:
                logger.info("-- 頂点確認: %s個目:終了", len(target_vertices))

        # 重複除去
        target_vertices = list(set(target_vertices))

        if not target_vertices:
            logger.warning("削除対象頂点が取得できなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None

        weighted_bone_indecies = []

        # 処理対象全頂点のウェイトボーンを確認
        for n, vidx in enumerate(target_vertices):
            v = model.vertex_dict[vidx]

            if type(v.deform) is Bdef1:
                weighted_bone_indecies.append(v.deform.index0)
            if type(v.deform) is Bdef2 or type(v.deform) is Sdef:
                if 0 < v.deform.weight0 < 1:
                    weighted_bone_indecies.append(v.deform.index0)
                if 0 < (1 - v.deform.weight0) < 1:
                    weighted_bone_indecies.append(v.deform.index1)
            elif type(v.deform) is Bdef4:
                if 0 < v.deform.weight0 < 1:
                    weighted_bone_indecies.append(v.deform.index0)
                if 0 < v.deform.weight1 < 1:
                    weighted_bone_indecies.append(v.deform.index1)
                if 0 < v.deform.weight2 < 1:
                    weighted_bone_indecies.append(v.deform.index2)
                if 0 < v.deform.weight3 < 1:
                    weighted_bone_indecies.append(v.deform.index3)

            if n > 0 and n % 1000 == 0:
                logger.info("-- ウェイト確認: %s個目:終了", n)

        semi_standard_bone_indecies = [-1]
        for bone_name in SEMI_STANDARD_BONE_NAMES:
            if bone_name in model.bones:
                semi_standard_bone_indecies.append(model.bones[bone_name].index)

        for bone in model.bones.values():
            # 子ボーンも削除
            if (
                bone.parent_index in model.bone_indexes
                and bone.index not in weighted_bone_indecies
                and bone.parent_index in weighted_bone_indecies
                and bone.name not in SEMI_STANDARD_BONE_NAMES
            ):
                weighted_bone_indecies.append(bone.index)

            # 中心ボーンもあれば削除
            if (
                bone.index in weighted_bone_indecies
                and bone.parent_index in model.bone_indexes
                and "中心" in model.bone_indexes[bone.parent_index]
                and bone.parent_index not in weighted_bone_indecies
            ):
                weighted_bone_indecies.append(bone.parent_index)

        # 重複除去(ついでに準標準ボーンも対象から削除)
        weighted_bone_indecies = list(sorted(list(set(weighted_bone_indecies) - set(semi_standard_bone_indecies))))

        is_executable = True
        if is_overwrite:
            for bone_index in weighted_bone_indecies:
                if bone_index not in model.bone_indexes:
                    continue
                bone = model.bones[model.bone_indexes[bone_index]]

                for morph in model.org_morphs.values():
                    if morph.morph_type == 2:
                        for offset in morph.offsets:
                            if type(offset) is BoneMorphData:
                                if offset.bone_index == bone.index:
                                    logger.error(
                                        "削除対象ボーンがボーンモーフとして登録されているため、削除出来ません。\n"
                                        + "事前にボーンモーフから外すか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s), モーフ名: %s",
                                        bone.name,
                                        bone.index,
                                        morph.name,
                                        decoration=MLogger.DECORATION_BOX,
                                    )
                                    is_executable = False

                for sub_bone in model.bones.values():
                    if sub_bone.parent_index == bone.index and sub_bone.index not in weighted_bone_indecies:
                        logger.error(
                            "削除対象ボーンが削除対象外ボーンの親ボーンとして登録されているため、削除出来ません。\n"
                            + "事前に親子関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外子ボーン: %s(%s)",
                            bone.name,
                            bone.index,
                            sub_bone.name,
                            sub_bone.index,
                            decoration=MLogger.DECORATION_BOX,
                        )
                        is_executable = False

                    if (
                        (sub_bone.getExternalRotationFlag() or sub_bone.getExternalTranslationFlag())
                        and sub_bone.effect_index == bone.index
                        and sub_bone.index not in weighted_bone_indecies
                    ):
                        logger.error(
                            "削除対象ボーンが削除対象外ボーンの付与親ボーンとして登録されているため、削除出来ません。\n"
                            + "事前に付与関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外付与子ボーン: %s(%s)",
                            bone.name,
                            bone.index,
                            sub_bone.name,
                            sub_bone.index,
                            decoration=MLogger.DECORATION_BOX,
                        )
                        is_executable = False

                    if sub_bone.getIkFlag():
                        if sub_bone.ik.target_index == bone.index and sub_bone.index not in weighted_bone_indecies:
                            logger.error(
                                "削除対象ボーンが削除対象外ボーンのリンクターゲットボーンとして登録されているため、削除出来ません。\n"
                                + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)",
                                bone.name,
                                bone.index,
                                sub_bone.name,
                                sub_bone.index,
                                decoration=MLogger.DECORATION_BOX,
                            )
                            is_executable = False

                        for link in sub_bone.ik.link:
                            if link.bone_index == bone.index and sub_bone.index not in weighted_bone_indecies:
                                logger.error(
                                    "削除対象ボーンが削除対象外ボーンのリンクボーンとして登録されているため、削除出来ません。\n"
                                    + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)",
                                    bone.name,
                                    bone.index,
                                    sub_bone.name,
                                    sub_bone.index,
                                    decoration=MLogger.DECORATION_BOX,
                                )
                                is_executable = False

                if n > 0 and n % 20 == 0:
                    logger.info("-- ウェイトボーンチェック確認: %s個目:終了", n)

        if not is_executable:
            return None

        weighted_rigidbody_indexes = {}
        for rigidbody in model.rigidbodies.values():
            if (
                rigidbody.index not in list(weighted_rigidbody_indexes.values())
                and rigidbody.bone_index in weighted_bone_indecies
                and model.bone_indexes[rigidbody.bone_index] not in SEMI_STANDARD_BONE_NAMES
            ):
                weighted_rigidbody_indexes[rigidbody.name] = rigidbody.index

        weighted_joint_indexes = {}
        for joint in model.joints.values():
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_a in list(
                weighted_rigidbody_indexes.values()
            ):
                weighted_joint_indexes[joint.name] = joint.name
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_b in list(
                weighted_rigidbody_indexes.values()
            ):
                weighted_joint_indexes[joint.name] = joint.name

        logger.info("ジョイント削除: %s", ", ".join((weighted_joint_indexes.keys())))

        # 削除
        for joint_name in weighted_joint_indexes.keys():
            del model.joints[joint_name]

        logger.info("剛体削除: %s", ", ".join(list(weighted_rigidbody_indexes.keys())))

        for rigidbody_name in weighted_rigidbody_indexes.keys():
            del model.rigidbodies[rigidbody_name]

        if is_overwrite:
            logger.info(
                "ボーン削除: %s", ", ".join([model.bone_indexes[bone_index] for bone_index in weighted_bone_indecies])
            )

            for bone_index in weighted_bone_indecies:
                del model.bones[model.bone_indexes[bone_index]]

        reset_rigidbodies = {}
        for ridx, (rigidbody_name, rigidbody) in enumerate(model.rigidbodies.items()):
            reset_rigidbodies[rigidbody.index] = {"name": rigidbody_name, "index": ridx}
            model.rigidbodies[rigidbody_name].index = ridx

        reset_bones = {}
        for bidx, (bone_name, bone) in enumerate(model.bones.items()):
            reset_bones[bone.index] = {"name": bone_name, "index": bidx}
            model.bones[bone_name].index = bidx
            model.bone_indexes[bidx] = bone_name

        logger.info("ジョイント再設定")

        for n, (joint_name, joint) in enumerate(model.joints.items()):
            if joint.rigidbody_index_a in reset_rigidbodies:
                joint.rigidbody_index_a = reset_rigidbodies[joint.rigidbody_index_a]["index"]
            if joint.rigidbody_index_b in reset_rigidbodies:
                joint.rigidbody_index_b = reset_rigidbodies[joint.rigidbody_index_b]["index"]

            if n > 0 and n % 100 == 0:
                logger.info("-- ジョイント再設定: %s個目:終了", n)

        logger.info("剛体再設定")

        for n, rigidbody in enumerate(model.rigidbodies.values()):
            if rigidbody.bone_index in reset_bones:
                rigidbody.bone_index = reset_bones[rigidbody.bone_index]["index"]
            else:
                rigidbody.bone_index = -1

            if n > 0 and n % 100 == 0:
                logger.info("-- 剛体再設定: %s個目:終了", n)

        if is_overwrite:
            logger.info("表示枠再設定")

            for n, display_slot in enumerate(model.display_slots.values()):
                new_references = []
                for display_type, bone_idx in display_slot.references:
                    if display_type == 0:
                        if bone_idx in reset_bones:
                            new_references.append((display_type, reset_bones[bone_idx]["index"]))
                    else:
                        new_references.append((display_type, bone_idx))
                display_slot.references = new_references

                if n > 0 and n % 100 == 0:
                    logger.info("-- 表示枠再設定: %s個目:終了", n)

            logger.info("モーフ再設定")

            for n, morph in enumerate(model.org_morphs.values()):
                if morph.morph_type == 2:
                    new_offsets = []
                    for offset in morph.offsets:
                        if type(offset) is BoneMorphData:
                            if offset.bone_index in reset_bones:
                                offset.bone_index = reset_bones[offset.bone_index]["index"]
                                new_offsets.append(offset)
                            else:
                                offset.bone_index = -1
                                new_offsets.append(offset)
                        else:
                            new_offsets.append(offset)
                    morph.offsets = new_offsets

                if n > 0 and n % 100 == 0:
                    logger.info("-- モーフ再設定: %s個目:終了", n)

            logger.info("ボーン再設定")

            for n, bone in enumerate(model.bones.values()):
                if bone.parent_index in reset_bones:
                    bone.parent_index = reset_bones[bone.parent_index]["index"]
                else:
                    bone.parent_index = -1

                if bone.getConnectionFlag():
                    if bone.tail_index in reset_bones:
                        bone.tail_index = reset_bones[bone.tail_index]["index"]
                    else:
                        bone.tail_index = -1

                if bone.getExternalRotationFlag() or bone.getExternalTranslationFlag():
                    if bone.effect_index in reset_bones:
                        bone.effect_index = reset_bones[bone.effect_index]["index"]
                    else:
                        bone.effect_index = -1

                if bone.getIkFlag():
                    if bone.ik.target_index in reset_bones:
                        bone.ik.target_index = reset_bones[bone.ik.target_index]["index"]
                        for link in bone.ik.link:
                            link.bone_index = reset_bones[link.bone_index]["index"]
                    else:
                        bone.ik.target_index = -1
                        for link in bone.ik.link:
                            link.bone_index = -1

                if n > 0 and n % 100 == 0:
                    logger.info("-- ボーン再設定: %s個目:終了", n)

            logger.info("頂点再設定")

            if param_option["parent_bone_name"] not in model.bones:
                logger.warning(
                    "指定された名前の親ボーンが存在しないため、物理クリア処理を中断します 親ボーン: %s",
                    param_option["parent_bone_name"],
                    decoration=MLogger.DECORATION_BOX,
                )
                return model

            parent_bone = model.bones[param_option["parent_bone_name"]]

            for n, vertex in enumerate(model.vertex_dict.values()):
                if type(vertex.deform) is Bdef1:
                    vertex.deform.index0 = (
                        reset_bones[vertex.deform.index0]["index"]
                        if vertex.deform.index0 in reset_bones
                        else parent_bone.index
                    )
                elif type(vertex.deform) is Bdef2:
                    v_indices = [
                        (reset_bones[vertex.deform.index0]["index"] if vertex.deform.index0 in reset_bones else 0),
                        (reset_bones[vertex.deform.index1]["index"] if vertex.deform.index1 in reset_bones else 0),
                    ]
                    v_weights = [
                        (vertex.deform.weight0 if v_indices[0] else 0),
                    ]
                    v_weights.append(1 - v_weights[0])

                    v_weights_idxs = np.nonzero(v_weights)[0]
                    v_indices = np.array(v_indices)[v_weights_idxs]
                    v_weights = np.array(v_weights)[v_weights_idxs]

                    if len(v_indices) == 0:
                        vertex.deform = Bdef1(parent_bone.index)
                    elif len(v_indices) == 1:
                        vertex.deform = Bdef1(v_indices[0])
                    else:
                        deform_weights = v_weights / v_weights.sum(axis=0, keepdims=1)
                        vertex.deform = Bdef2(v_indices[0], v_indices[1], deform_weights[0])
                elif type(vertex.deform) is Bdef4:
                    v_indices = [
                        (reset_bones[vertex.deform.index0]["index"] if vertex.deform.index0 in reset_bones else 0),
                        (reset_bones[vertex.deform.index1]["index"] if vertex.deform.index1 in reset_bones else 0),
                        (reset_bones[vertex.deform.index2]["index"] if vertex.deform.index2 in reset_bones else 0),
                        (reset_bones[vertex.deform.index3]["index"] if vertex.deform.index3 in reset_bones else 0),
                    ]
                    v_weights = [
                        (vertex.deform.weight0 if v_indices[0] else 0),
                        (vertex.deform.weight1 if v_indices[1] else 0),
                        (vertex.deform.weight2 if v_indices[2] else 0),
                        (vertex.deform.weight3 if v_indices[3] else 0),
                    ]
                    v_weights_idxs = np.nonzero(v_weights)[0]
                    v_indices = np.array(v_indices)[v_weights_idxs]
                    v_weights = np.array(v_weights)[v_weights_idxs]

                    if len(v_indices) == 0:
                        vertex.deform = Bdef1(parent_bone.index)
                    elif len(v_indices) == 1:
                        vertex.deform = Bdef1(v_indices[0])
                    elif len(v_indices) == 2:
                        deform_weights = v_weights / v_weights.sum(axis=0, keepdims=1)
                        vertex.deform = Bdef2(v_indices[0], v_indices[1], deform_weights[0])
                    elif len(v_indices) == 3:
                        deform_weights = v_weights / v_weights.sum(axis=0, keepdims=1)
                        vertex.deform = Bdef4(
                            v_indices[0],
                            v_indices[1],
                            v_indices[2],
                            parent_bone.index,
                            deform_weights[0],
                            deform_weights[1],
                            deform_weights[2],
                            0,
                        )
                    else:
                        deform_weights = v_weights / v_weights.sum(axis=0, keepdims=1)
                        vertex.deform = Bdef4(
                            v_indices[0],
                            v_indices[1],
                            v_indices[2],
                            v_indices[3],
                            deform_weights[0],
                            deform_weights[1],
                            deform_weights[2],
                            deform_weights[3],
                        )

                elif type(vertex.deform) is Sdef:
                    vertex.deform.index0 = (
                        reset_bones[vertex.deform.index0]["index"] if vertex.deform.index0 in reset_bones else 0
                    )
                    vertex.deform.index1 = (
                        reset_bones[vertex.deform.index1]["index"] if vertex.deform.index1 in reset_bones else 0
                    )

                if n > 0 and n % 1000 == 0:
                    logger.info("-- 頂点再設定: %s個目:終了", n)

        return model

    def change_physics_parent(
        self, model: PmxModel, param_option: dict, all_virtual_vertices: list, all_vertex_maps: list, pidx: int
    ):
        material_name = param_option["material_name"]
        abb_name = param_option["abb_name"]

        logger.info("【%s:%s】物理親置換", material_name, abb_name, decoration=MLogger.DECORATION_LINE)

        physics_parent_idx = param_option["physics_parent"] - 1

        if len(all_virtual_vertices) - 1 < physics_parent_idx:
            logger.warning(
                "存在しない物理親が指定されているため、処理を終了します 対象物理設定: %s, 物理親INDEX: %s",
                abb_name,
                (physics_parent_idx + 1),
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        target_virtual_vertices = all_virtual_vertices[pidx]
        target_vertex_maps = all_vertex_maps[pidx]

        parent_virtual_vertices = all_virtual_vertices[physics_parent_idx]
        parent_vertex_maps = all_vertex_maps[physics_parent_idx]

        logger.info("-- 【%s:%s】親物理末端ボーン位置取得", material_name, abb_name)

        # 親物理のボーン位置一覧を取得（特に上限は設けない）
        parent_bottom_bone_poses = {}
        for midx, vertex_map in parent_vertex_maps.items():
            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    vkey = tuple(vertex_map[v_yidx, v_xidx])
                    if np.isnan(vkey).any() or tuple(vkey) not in parent_virtual_vertices:
                        continue

                    vv = parent_virtual_vertices[tuple(vkey)]
                    bone = vv.map_bones.get(midx, None)
                    if not bone:
                        continue

                    parent_bottom_bone_poses[tuple(vkey)] = bone.position.data()

        logger.info("-- 【%s:%s】処理対象物理剛体取得", material_name, abb_name)

        rigidbodies = []
        for midx, vertex_map in target_vertex_maps.items():
            for vkey in vertex_map[0, :]:
                if np.isnan(vkey).any() or tuple(vkey) not in target_virtual_vertices:
                    continue

                vv = target_virtual_vertices[tuple(vkey)]
                bone = vv.map_bones.get(midx, None)
                rigidbody = vv.map_rigidbodies.get(midx, None)
                if not bone or not rigidbody:
                    continue
                rigidbodies.append(rigidbody.index)

        logger.info("-- 【%s:%s】処理対象物理ジョイント取得", material_name, abb_name)

        target_joints = {}
        for joint in model.joints.values():
            if joint.rigidbody_index_b in rigidbodies and (
                "↓" in joint.name or "↑" in joint.name or "／" in joint.name or "＼" in joint.name
            ):
                # 剛体B（接続先）に処理対象剛体がある場合、ジョイント保持
                if joint.rigidbody_index_b not in target_joints:
                    target_joints[joint.rigidbody_index_b] = []
                target_joints[joint.rigidbody_index_b].append(joint)

        logger.info("-- 【%s:%s】処理対象親物理置換", material_name, abb_name)

        jcnt = 0
        for midx, vertex_map in target_vertex_maps.items():
            for vkey in vertex_map[0, :]:
                if np.isnan(vkey).any() or tuple(vkey) not in target_virtual_vertices:
                    continue

                vv = target_virtual_vertices[tuple(vkey)]
                bone = vv.map_bones.get(midx, None)
                rigidbody = vv.map_rigidbodies.get(midx, None)
                if not bone or not rigidbody:
                    continue

                # 処理対象ボーンから最も近い親ボーンを選択する
                parent_distances = np.linalg.norm(
                    (np.array(list(parent_bottom_bone_poses.values())) - bone.position.data()), ord=2, axis=1
                )

                nearest_parent_bone = None
                nearest_parent_rigidbody = None
                for distance_idx in np.argsort(parent_distances):
                    # 最も近い親ボーンINDEX
                    nearest_parent_bone_vkey = list(parent_bottom_bone_poses.keys())[distance_idx]

                    nearest_parent_vv = parent_virtual_vertices[nearest_parent_bone_vkey]
                    if not (
                        [b for b in nearest_parent_vv.map_bones.values() if b]
                        and [r for r in nearest_parent_vv.map_rigidbodies.values() if r]
                    ):
                        # ボーンか剛体がない場合、スルー
                        continue

                    nearest_parent_bone = [b for b in nearest_parent_vv.map_bones.values() if b][0]
                    nearest_parent_rigidbody = [r for r in nearest_parent_vv.map_rigidbodies.values() if r][0]
                    break

                if not (nearest_parent_bone and nearest_parent_rigidbody):
                    logger.warning(
                        "物理親に相当するボーンもしくは剛体が検出できなかったため、処理を終了します 対象物理設定: %s, 物理親INDEX: %s",
                        abb_name,
                        (physics_parent_idx + 1),
                        decoration=MLogger.DECORATION_BOX,
                    )
                    return False

                # 処理対象ボーンの親ボーンに最近親ボーンを設定
                bone.parent_index = nearest_parent_bone.index

                # 剛体を物理設定に切替
                rigidbody.mode = 1

                # ジョイントの剛体Aを親ボーンに紐付く剛体INDEXに変換
                if rigidbody.index in target_joints:
                    for joint in target_joints[rigidbody.index]:
                        joint.name = f"{joint.name[0]}|{nearest_parent_rigidbody.name}|{rigidbody.name}"
                        joint.rigidbody_index_a = nearest_parent_rigidbody.index
                        jcnt += 1

                if jcnt > 0 and jcnt % 1000 == 0:
                    logger.info("-- -- 【%s:%s】処理対象親物理置換: %s個目:終了", material_name, abb_name, jcnt)

        logger.info("-- -- 【%s:%s】処理対象親物理置換: %s個目:終了", material_name, abb_name, jcnt)

        return True

    def create_physics(self, model: PmxModel, param_option: dict):
        model.comment += f"\r\n{logger.transtext('材質')}: {param_option['material_name']} --------------"
        model.comment += f"\r\n　　{logger.transtext('略称')}: {param_option['abb_name']}"

        if param_option["advance_comment"]:
            model.comment += f", {logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"
            model.comment += f", {logger.transtext('質量')}: {round(param_option['mass'], 3)}"
            model.comment += f", {logger.transtext('柔らかさ')}: {round(param_option['air_resistance'], 3)}"
            model.comment += f", {logger.transtext('張り')}: {round(param_option['shape_maintenance'], 3)}"
            model.comment += f", {logger.transtext('特殊形状')}: {param_option['special_shape']}"
            model.comment += f"\r\n　　{logger.transtext('ボーン密度')}"
            model.comment += f", {logger.transtext('縦密度')}: {param_option['vertical_bone_density']}"
            model.comment += f", {logger.transtext('横密度')}: {param_option['horizonal_bone_density']}"
            model.comment += f", {logger.transtext('オフセット')}: {param_option['horizonal_bone_offset']}"
            model.comment += f", {logger.transtext('密度基準')}: {param_option['density_type']}"
            model.comment += f"\r\n　　{logger.transtext('根元剛体')}"
            model.comment += f", {logger.transtext('質量')}: {round(param_option['rigidbody'].param.mass, 3)}"
            model.comment += (
                f", {logger.transtext('移動減衰')}: {round(param_option['rigidbody'].param.linear_damping, 3)}"
            )
            model.comment += (
                f", {logger.transtext('回転減衰')}: {round(param_option['rigidbody'].param.angular_damping, 3)}"
            )
            model.comment += f", {logger.transtext('反発力')}: {round(param_option['rigidbody'].param.restitution, 3)}"
            model.comment += f", {logger.transtext('摩擦力')}: {round(param_option['rigidbody'].param.friction, 3)}"
            model.comment += f", {logger.transtext('係数')}: {round(param_option['rigidbody_coefficient'], 3)}"
            model.comment += f"\r\n　　{logger.transtext('剛体の厚み')}"
            model.comment += f", {logger.transtext('根元厚み')}: {round(param_option['rigidbody_root_thicks'], 3)}"
            model.comment += f", {logger.transtext('末端厚み')}: {round(param_option['rigidbody_end_thicks'], 3)}"
            model.comment += f"\r\n　　{logger.transtext('詳細オプション')}"
            model.comment += f", {logger.transtext('物理接続')}: {param_option['parent_type']}"
            model.comment += f", {logger.transtext('物理タイプ')}: {param_option['physics_type']}"
            model.comment += f", {logger.transtext('ジョイント位置')}: {param_option['joint_pos_type']}"
            model.comment += f", {logger.transtext('ルート探索')}: {param_option['route_search_type']}"
            model.comment += f", {logger.transtext('根元頂点推定')}: {param_option['route_estimate_type']}"
            model.comment += f", {logger.transtext('物理親')}: {param_option['physics_parent']}"

            if param_option["vertical_joint"]:
                model.comment += f"\r\n　　{logger.transtext('縦ジョイント')}"
                model.comment += (
                    f", {logger.transtext('制限係数')}: {round(param_option['vertical_joint_coefficient'], 3)}"
                )
                model.comment += f"\r\n　　　　{logger.transtext('移動X(最小)')}: {round(param_option['vertical_joint'].translation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最小)')}: {round(param_option['vertical_joint'].translation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最小)')}: {round(param_option['vertical_joint'].translation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('移動X(最大)')}: {round(param_option['vertical_joint'].translation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最大)')}: {round(param_option['vertical_joint'].translation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最大)')}: {round(param_option['vertical_joint'].translation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('回転X(最小)')}: {round(param_option['vertical_joint'].rotation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最小)')}: {round(param_option['vertical_joint'].rotation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最小)')}: {round(param_option['vertical_joint'].rotation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('回転X(最大)')}: {round(param_option['vertical_joint'].rotation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最大)')}: {round(param_option['vertical_joint'].rotation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最大)')}: {round(param_option['vertical_joint'].rotation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('ばね(移動X)')}: {round(param_option['vertical_joint'].spring_constant_translation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Y)')}: {round(param_option['vertical_joint'].spring_constant_translation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Z)')}: {round(param_option['vertical_joint'].spring_constant_translation.z(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転X)')}: {round(param_option['vertical_joint'].spring_constant_rotation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Y)')}: {round(param_option['vertical_joint'].spring_constant_rotation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Z)')}: {round(param_option['vertical_joint'].spring_constant_rotation.z(), 3)}"

            if param_option["horizonal_joint"]:
                model.comment += f"\r\n　　{logger.transtext('横ジョイント')}"
                model.comment += f", {logger.transtext('制限係数')}: {param_option['horizonal_joint_coefficient']}"
                model.comment += f", {logger.transtext('親剛体距離制限')}: {param_option['horizonal_joint_restruct']}"
                model.comment += f"\r\n　　　　{logger.transtext('移動X(最小)')}: {round(param_option['horizonal_joint'].translation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最小)')}: {round(param_option['horizonal_joint'].translation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最小)')}: {round(param_option['horizonal_joint'].translation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('移動X(最大)')}: {round(param_option['horizonal_joint'].translation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最大)')}: {round(param_option['horizonal_joint'].translation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最大)')}: {round(param_option['horizonal_joint'].translation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('回転X(最小)')}: {round(param_option['horizonal_joint'].rotation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最小)')}: {round(param_option['horizonal_joint'].rotation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最小)')}: {round(param_option['horizonal_joint'].rotation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('回転X(最大)')}: {round(param_option['horizonal_joint'].rotation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最大)')}: {round(param_option['horizonal_joint'].rotation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最大)')}: {round(param_option['horizonal_joint'].rotation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('ばね(移動X)')}: {round(param_option['horizonal_joint'].spring_constant_translation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Y)')}: {round(param_option['horizonal_joint'].spring_constant_translation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Z)')}: {round(param_option['horizonal_joint'].spring_constant_translation.z(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転X)')}: {round(param_option['horizonal_joint'].spring_constant_rotation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Y)')}: {round(param_option['horizonal_joint'].spring_constant_rotation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Z)')}: {round(param_option['horizonal_joint'].spring_constant_rotation.z(), 3)}"

            if param_option["diagonal_joint"]:
                model.comment += f"\r\n　　{logger.transtext('斜めジョイント')}"
                model.comment += (
                    f", {logger.transtext('制限係数')}: {round(param_option['diagonal_joint_coefficient'], 3)}"
                )
                model.comment += f"\r\n　　　　{logger.transtext('移動X(最小)')}: {round(param_option['diagonal_joint'].translation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最小)')}: {round(param_option['diagonal_joint'].translation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最小)')}: {round(param_option['diagonal_joint'].translation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('移動X(最大)')}: {round(param_option['diagonal_joint'].translation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最大)')}: {round(param_option['diagonal_joint'].translation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最大)')}: {round(param_option['diagonal_joint'].translation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('回転X(最小)')}: {round(param_option['diagonal_joint'].rotation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最小)')}: {round(param_option['diagonal_joint'].rotation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最小)')}: {round(param_option['diagonal_joint'].rotation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('回転X(最大)')}: {round(param_option['diagonal_joint'].rotation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最大)')}: {round(param_option['diagonal_joint'].rotation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最大)')}: {round(param_option['diagonal_joint'].rotation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('ばね(移動X)')}: {round(param_option['diagonal_joint'].spring_constant_translation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Y)')}: {round(param_option['diagonal_joint'].spring_constant_translation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Z)')}: {round(param_option['diagonal_joint'].spring_constant_translation.z(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転X)')}: {round(param_option['diagonal_joint'].spring_constant_rotation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Y)')}: {round(param_option['diagonal_joint'].spring_constant_rotation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Z)')}: {round(param_option['diagonal_joint'].spring_constant_rotation.z(), 3)}"

            if param_option["vertical_reverse_joint"]:
                model.comment += f"\r\n　　{logger.transtext('縦逆ジョイント')}"
                model.comment += (
                    f", {logger.transtext('制限係数')}: {round(param_option['vertical_reverse_joint_coefficient'], 3)}"
                )
                model.comment += f"\r\n　　　　{logger.transtext('移動X(最小)')}: {round(param_option['vertical_reverse_joint'].translation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最小)')}: {round(param_option['vertical_reverse_joint'].translation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最小)')}: {round(param_option['vertical_reverse_joint'].translation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('移動X(最大)')}: {round(param_option['vertical_reverse_joint'].translation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最大)')}: {round(param_option['vertical_reverse_joint'].translation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最大)')}: {round(param_option['vertical_reverse_joint'].translation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('回転X(最小)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最小)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最小)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('回転X(最大)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最大)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最大)')}: {round(param_option['vertical_reverse_joint'].rotation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('ばね(移動X)')}: {round(param_option['vertical_reverse_joint'].spring_constant_translation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Y)')}: {round(param_option['vertical_reverse_joint'].spring_constant_translation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Z)')}: {round(param_option['vertical_reverse_joint'].spring_constant_translation.z(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転X)')}: {round(param_option['vertical_reverse_joint'].spring_constant_rotation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Y)')}: {round(param_option['vertical_reverse_joint'].spring_constant_rotation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Z)')}: {round(param_option['vertical_reverse_joint'].spring_constant_rotation.z(), 3)}"

            if param_option["horizonal_reverse_joint"]:
                model.comment += f"\r\n　　{logger.transtext('横逆ジョイント')}"
                model.comment += (
                    f", {logger.transtext('制限係数')}: {round(param_option['horizonal_reverse_joint_coefficient'], 3)}"
                )
                model.comment += f"\r\n　　　　{logger.transtext('移動X(最小)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最小)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最小)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('移動X(最大)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('移動Y(最大)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('移動Z(最大)')}: {round(param_option['horizonal_reverse_joint'].translation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('回転X(最小)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_min.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最小)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_min.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最小)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_min.z(), 3)}"
                model.comment += f", {logger.transtext('回転X(最大)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_max.x(), 3)}"
                model.comment += f", {logger.transtext('回転Y(最大)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_max.y(), 3)}"
                model.comment += f", {logger.transtext('回転Z(最大)')}: {round(param_option['horizonal_reverse_joint'].rotation_limit_max.z(), 3)}"
                model.comment += f"\r\n　　　　{logger.transtext('ばね(移動X)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_translation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Y)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_translation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(移動Z)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_translation.z(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転X)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_rotation.x(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Y)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_rotation.y(), 3)}"
                model.comment += f", {logger.transtext('ばね(回転Z)')}: {round(param_option['horizonal_reverse_joint'].spring_constant_rotation.z(), 3)}"

        material_name = param_option["material_name"]

        grad_vertices = []
        # 物理対象頂点CSVが指定されている場合、対象頂点リスト生成
        if param_option["vertices_csv"]:
            # まずは物理対象材質だけでチェック
            target_vertices = read_vertices_from_file(param_option["vertices_csv"], model, material_name)
            # 他材質をグラデーションウェイトに割り当てる事がありうるので、材質指定なしにチェック
            grad_vertices = read_vertices_from_file(param_option["vertices_csv"], model, None)
            # 物理材質の頂点は除外
            grad_vertices = list(set(grad_vertices) - set(target_vertices))
            logger.debug("grad_vertices: %s", grad_vertices)
            if not target_vertices:
                logger.warning(
                    "物理対象頂点CSVが正常に読み込めなかったため、処理を終了します\n物理材質の中に物理対象CSVで指定した頂点が含まれている事を確認してください。",
                    decoration=MLogger.DECORATION_BOX,
                )
                return False, None, None
        else:
            target_vertices = list(model.material_vertices[material_name])

        # 方向に応じて判定値を変える
        # デフォルトは下
        base_vertical_axis = MVector3D(0, -1, 0)
        target_idx = 1
        if param_option["direction"] == logger.transtext("上"):
            base_vertical_axis = MVector3D(0, 1, 0)
            target_idx = 1
        elif param_option["direction"] == logger.transtext("右"):
            base_vertical_axis = MVector3D(-1, 0, 0)
            target_idx = 0
        elif param_option["direction"] == logger.transtext("左"):
            base_vertical_axis = MVector3D(1, 0, 0)
            target_idx = 0
        base_reverse_axis = MVector3D(np.logical_not(np.abs(base_vertical_axis.data())))

        if param_option["exist_physics_clear"] == logger.transtext("再利用") and param_option["physics_type"] in [
            logger.transtext("胸")
        ]:
            # 胸の場合、N式おっぱい構造ボーンとウェイトを設定する
            (
                virtual_vertices,
                vertex_maps,
                all_registered_bones,
                all_bone_connected,
                root_bone,
                is_material_horizonal,
            ) = self.create_bone_map(
                model,
                param_option,
                material_name,
                target_vertices,
                MVector3D(0, 0, -1),
                is_root_bone=(param_option["parent_type"] == logger.transtext("中心")),
            )

            self.create_bust_physics(
                model,
                param_option,
                material_name,
                target_vertices,
                virtual_vertices,
                vertex_maps,
            )
        else:
            if param_option["exist_physics_clear"] == logger.transtext("再利用"):
                (
                    virtual_vertices,
                    vertex_maps,
                    all_registered_bones,
                    all_bone_connected,
                    root_bone,
                    is_material_horizonal,
                ) = self.create_bone_map(
                    model,
                    param_option,
                    material_name,
                    target_vertices,
                    base_vertical_axis,
                    is_root_bone=(param_option["parent_type"] == logger.transtext("中心")),
                )

                left_root_bone = right_root_bone = None
            else:
                (
                    vertex_maps,
                    virtual_vertices,
                    remaining_vertices,
                    back_vertices,
                    threshold,
                    is_material_horizonal,
                    horizonal_top_edge_keys,
                ) = self.create_vertex_map(
                    model,
                    param_option,
                    material_name,
                    target_vertices,
                    base_vertical_axis,
                    base_reverse_axis,
                    target_idx,
                )

                if not vertex_maps:
                    return False, None, None

                # 各頂点の有効INDEX数が最も多いものをベースとする
                map_cnt = []
                for vertex_map in vertex_maps.values():
                    map_cnt.append(np.count_nonzero(~np.isnan(vertex_map)) / 3)

                if len(map_cnt) == 0:
                    logger.warning("有効な頂点マップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
                    return False, None, None

                # ボーンはどのマップにも作成する
                vertex_map_orders = [k for k in np.argsort(-np.array(map_cnt))]

                (
                    root_bone,
                    left_root_bone,
                    right_root_bone,
                    virtual_vertices,
                    all_registered_bones,
                    all_bone_vertical_distances,
                    all_bone_horizonal_distances,
                    all_bone_connected,
                ) = self.create_bone(
                    model,
                    param_option,
                    material_name,
                    virtual_vertices,
                    vertex_maps,
                    vertex_map_orders,
                    threshold,
                    base_vertical_axis,
                )

                remaining_vertices = self.create_weight(
                    model,
                    param_option,
                    material_name,
                    virtual_vertices,
                    vertex_maps,
                    all_registered_bones,
                    all_bone_vertical_distances,
                    all_bone_horizonal_distances,
                    all_bone_connected,
                    remaining_vertices,
                    threshold,
                    root_bone,
                    left_root_bone,
                    right_root_bone,
                )

                # グラデウェイト
                remaining_vertices = self.create_grad_weight(
                    model,
                    param_option,
                    target_vertices,
                    virtual_vertices,
                    remaining_vertices,
                    threshold,
                    base_vertical_axis,
                    base_reverse_axis,
                    horizonal_top_edge_keys,
                    grad_vertices,
                    root_bone,
                    left_root_bone,
                    right_root_bone,
                )

                # 裏ウェイト
                remaining_vidxs = self.create_back_weight(
                    model,
                    param_option,
                    target_vertices,
                    grad_vertices,
                    virtual_vertices,
                    back_vertices,
                    remaining_vertices,
                    threshold,
                )

                # 残ウェイト
                self.create_remaining_weight(
                    model,
                    param_option,
                    material_name,
                    virtual_vertices,
                    vertex_maps,
                    all_registered_bones,
                    remaining_vidxs,
                    threshold,
                )

            root_rigidbody, left_root_rigidbody, right_root_rigidbody, parent_bone_rigidbody = self.create_rigidbody(
                model,
                param_option,
                material_name,
                virtual_vertices,
                vertex_maps,
                all_registered_bones,
                all_bone_connected,
                root_bone,
                left_root_bone,
                right_root_bone,
                base_vertical_axis,
                base_reverse_axis,
                is_material_horizonal,
            )

            self.create_joint(
                model,
                param_option,
                material_name,
                virtual_vertices,
                vertex_maps,
                all_registered_bones,
                all_bone_connected,
                root_rigidbody,
                left_root_rigidbody,
                right_root_rigidbody,
                parent_bone_rigidbody,
            )

        return True, virtual_vertices, vertex_maps

    def create_joint(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        all_registered_bones: dict,
        all_bone_connected: dict,
        root_rigidbody: RigidBody,
        left_root_rigidbody: RigidBody,
        right_root_rigidbody: RigidBody,
        parent_bone_rigidbody: RigidBody,
    ):
        logger.info("【%s:%s】ジョイント生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        # ジョイント生成
        created_joints = {}
        prev_joint_cnt = 0

        # 中央配置か否か
        is_center = param_option["density_type"] == logger.transtext("中央")

        for base_map_idx, vertex_map in vertex_maps.items():
            logger.info("--【No.%s】ジョイント生成", base_map_idx + 1)

            registered_bones = all_registered_bones[base_map_idx]

            # キーは縦段の数分生成
            vv_keys = list(range(vertex_map.shape[0]))

            # 縦ジョイント情報
            (
                vertical_limit_min_mov_xs,
                vertical_limit_min_mov_ys,
                vertical_limit_min_mov_zs,
                vertical_limit_max_mov_xs,
                vertical_limit_max_mov_ys,
                vertical_limit_max_mov_zs,
                vertical_limit_min_rot_xs,
                vertical_limit_min_rot_ys,
                vertical_limit_min_rot_zs,
                vertical_limit_max_rot_xs,
                vertical_limit_max_rot_ys,
                vertical_limit_max_rot_zs,
                vertical_spring_constant_mov_xs,
                vertical_spring_constant_mov_ys,
                vertical_spring_constant_mov_zs,
                vertical_spring_constant_rot_xs,
                vertical_spring_constant_rot_ys,
                vertical_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["vertical_joint"], vv_keys, param_option["vertical_joint_coefficient"]
            )

            # 横ジョイント情報
            (
                horizonal_limit_min_mov_xs,
                horizonal_limit_min_mov_ys,
                horizonal_limit_min_mov_zs,
                horizonal_limit_max_mov_xs,
                horizonal_limit_max_mov_ys,
                horizonal_limit_max_mov_zs,
                horizonal_limit_min_rot_xs,
                horizonal_limit_min_rot_ys,
                horizonal_limit_min_rot_zs,
                horizonal_limit_max_rot_xs,
                horizonal_limit_max_rot_ys,
                horizonal_limit_max_rot_zs,
                horizonal_spring_constant_mov_xs,
                horizonal_spring_constant_mov_ys,
                horizonal_spring_constant_mov_zs,
                horizonal_spring_constant_rot_xs,
                horizonal_spring_constant_rot_ys,
                horizonal_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["horizonal_joint"], vv_keys, param_option["horizonal_joint_coefficient"]
            )

            # 斜めジョイント情報
            (
                diagonal_limit_min_mov_xs,
                diagonal_limit_min_mov_ys,
                diagonal_limit_min_mov_zs,
                diagonal_limit_max_mov_xs,
                diagonal_limit_max_mov_ys,
                diagonal_limit_max_mov_zs,
                diagonal_limit_min_rot_xs,
                diagonal_limit_min_rot_ys,
                diagonal_limit_min_rot_zs,
                diagonal_limit_max_rot_xs,
                diagonal_limit_max_rot_ys,
                diagonal_limit_max_rot_zs,
                diagonal_spring_constant_mov_xs,
                diagonal_spring_constant_mov_ys,
                diagonal_spring_constant_mov_zs,
                diagonal_spring_constant_rot_xs,
                diagonal_spring_constant_rot_ys,
                diagonal_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["diagonal_joint"], vv_keys, param_option["diagonal_joint_coefficient"]
            )

            # 縦逆ジョイント情報
            (
                vertical_reverse_limit_min_mov_xs,
                vertical_reverse_limit_min_mov_ys,
                vertical_reverse_limit_min_mov_zs,
                vertical_reverse_limit_max_mov_xs,
                vertical_reverse_limit_max_mov_ys,
                vertical_reverse_limit_max_mov_zs,
                vertical_reverse_limit_min_rot_xs,
                vertical_reverse_limit_min_rot_ys,
                vertical_reverse_limit_min_rot_zs,
                vertical_reverse_limit_max_rot_xs,
                vertical_reverse_limit_max_rot_ys,
                vertical_reverse_limit_max_rot_zs,
                vertical_reverse_spring_constant_mov_xs,
                vertical_reverse_spring_constant_mov_ys,
                vertical_reverse_spring_constant_mov_zs,
                vertical_reverse_spring_constant_rot_xs,
                vertical_reverse_spring_constant_rot_ys,
                vertical_reverse_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["vertical_reverse_joint"], vv_keys, param_option["vertical_reverse_joint_coefficient"]
            )

            # 横逆ジョイント情報
            (
                horizonal_reverse_limit_min_mov_xs,
                horizonal_reverse_limit_min_mov_ys,
                horizonal_reverse_limit_min_mov_zs,
                horizonal_reverse_limit_max_mov_xs,
                horizonal_reverse_limit_max_mov_ys,
                horizonal_reverse_limit_max_mov_zs,
                horizonal_reverse_limit_min_rot_xs,
                horizonal_reverse_limit_min_rot_ys,
                horizonal_reverse_limit_min_rot_zs,
                horizonal_reverse_limit_max_rot_xs,
                horizonal_reverse_limit_max_rot_ys,
                horizonal_reverse_limit_max_rot_zs,
                horizonal_reverse_spring_constant_mov_xs,
                horizonal_reverse_spring_constant_mov_ys,
                horizonal_reverse_spring_constant_mov_zs,
                horizonal_reverse_spring_constant_rot_xs,
                horizonal_reverse_spring_constant_rot_ys,
                horizonal_reverse_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["horizonal_reverse_joint"], vv_keys, param_option["horizonal_reverse_joint_coefficient"]
            )

            for v_xidx in range(vertex_map.shape[1]):
                # 実際に剛体を作ってる最後のY
                actual_v_yidx = (
                    np.min(np.sort(np.where(registered_bones[:, v_xidx])[0])[-2:])
                    if registered_bones[:, v_xidx].any()
                    else 0
                )

                for v_yidx in range(vertex_map.shape[0]):

                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        continue

                    bone_key = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices.get(bone_key, None)

                    if not vv:
                        logger.warning("ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s", bone_key)
                        continue

                    if not registered_bones[v_yidx, v_xidx]:
                        # ボーンがない場合ジョイントは不要
                        continue

                    # 親剛体の計算用カプセル
                    mat = MMatrix4x4()
                    mat.setToIdentity()
                    mat.translate(parent_bone_rigidbody.shape_position)
                    mat.rotate(parent_bone_rigidbody.shape_qq)

                    parent_capsule = MCapsule(
                        MSegment(
                            mat * MVector3D(-parent_bone_rigidbody.shape_size.y(), 0, 0),
                            mat * MVector3D(parent_bone_rigidbody.shape_size.y(), 0, 0),
                        ),
                        parent_bone_rigidbody.shape_size.x(),
                    )

                    bone_y_idx = v_yidx

                    (
                        prev_map_idx,
                        prev_xidx,
                        prev_connected,
                        next_map_idx,
                        next_xidx,
                        next_connected,
                        above_yidx,
                        below_yidx,
                        target_v_yidx,
                        target_v_xidx,
                        registered_max_v_yidx,
                        registered_max_v_xidx,
                        max_v_yidx,
                        max_v_xidx,
                    ) = self.get_block_vidxs(
                        v_yidx,
                        v_xidx,
                        vertex_maps,
                        all_registered_bones,
                        all_bone_connected,
                        base_map_idx,
                    )

                    prev_above_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and above_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and not np.isnan(vertex_maps[prev_map_idx][above_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    prev_now_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and v_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and not np.isnan(vertex_maps[prev_map_idx][v_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    prev_below_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][below_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and below_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and not np.isnan(vertex_maps[prev_map_idx][below_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_above_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][above_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and above_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and not np.isnan(vertex_maps[base_map_idx][above_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_now_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][v_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and v_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and not np.isnan(vertex_maps[base_map_idx][v_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_below_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][below_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and below_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and not np.isnan(vertex_maps[base_map_idx][below_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_above_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][above_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and above_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and not np.isnan(vertex_maps[next_map_idx][above_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_now_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][v_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and v_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and not np.isnan(vertex_maps[next_map_idx][v_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_below_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][below_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and below_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and not np.isnan(vertex_maps[next_map_idx][below_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    if param_option["vertical_joint"]:
                        # 縦ジョイント
                        if v_yidx == 0:
                            if root_rigidbody:
                                a_rigidbody = root_rigidbody
                            else:
                                a_rigidbody = (
                                    left_root_rigidbody
                                    if now_now_vv.position().x() > 0 and left_root_rigidbody
                                    else right_root_rigidbody
                                )
                            b_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                            a_xidx = v_xidx
                            a_yidx = -1
                            b_xidx = v_xidx
                            b_yidx = v_yidx
                        else:
                            a_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)
                            if not a_rigidbody:
                                # スリットで上の剛体が見つからない場合、ひとつ前の頂点マップを確認する
                                a_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx - 1, None)
                            b_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                            a_xidx = v_xidx
                            a_yidx = above_yidx
                            b_xidx = v_xidx
                            b_yidx = v_yidx

                        if not (a_rigidbody and a_rigidbody.index >= 0 and b_rigidbody and b_rigidbody.index >= 0):
                            if not (
                                a_rigidbody
                                and b_rigidbody
                                and b_rigidbody.index >= 0
                                and a_rigidbody.index != b_rigidbody.index
                            ):
                                if v_yidx == 0:
                                    logger.warning(
                                        "縦ジョイント生成に必要な情報が取得できなかった為、スルーします。\n"
                                        + "縦ジョイントがないと、物理がチーズのように伸びる可能性があります。\n"
                                        + "頂点マップで一行目（根元）がNoneの要素がないか確認してください。\n"
                                        + "Noneの要素が根元にあり、かつ根元CSVが未指定の場合、根元CSVを指定して下さい。　処理対象: %s",
                                        vv.map_bones[base_map_idx].name
                                        if vv.map_bones.get(base_map_idx, None)
                                        else vv.vidxs(),
                                        decoration=MLogger.DECORATION_BOX,
                                    )
                                elif v_yidx < registered_max_v_yidx:
                                    logger.warning(
                                        "縦ジョイント生成に必要な情報が取得できなかった為、スルーします。　処理対象: %s",
                                        vv.map_bones[base_map_idx].name
                                        if vv.map_bones.get(base_map_idx, None)
                                        else vv.vidxs(),
                                    )
                        else:
                            a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                            b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                            if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                # 剛体が重なる箇所の交点
                                above_mat = MMatrix4x4()
                                above_point = MVector3D()
                                if a_rigidbody:
                                    above_mat.setToIdentity()
                                    above_mat.translate(a_rigidbody.shape_position)
                                    above_mat.rotate(a_rigidbody.shape_qq)
                                    above_point = above_mat * MVector3D(0, -a_rigidbody.shape_size.y(), 0)

                                now_mat = MMatrix4x4()
                                now_mat.setToIdentity()
                                now_mat.translate(b_rigidbody.shape_position)
                                now_mat.rotate(b_rigidbody.shape_qq)
                                now_point = now_mat * MVector3D(0, b_rigidbody.shape_size.y(), 0)

                                if v_yidx == 0:
                                    joint_pos = now_point
                                else:
                                    joint_pos = (above_point + now_point) / 2
                            else:
                                joint_pos = b_pos

                            if v_yidx == 0:
                                joint_qq = b_rigidbody.shape_qq
                            else:
                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                if v_yidx == 0:
                                    z_direction_pos = b_rigidbody.z_direction.normalized()
                                else:
                                    z_direction_pos = (
                                        (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                    ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                            bone_distance = a_pos.distanceToPoint(b_pos)

                            joint_key, joint = self.build_joint(
                                "↓",
                                10,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                base_map_idx,
                                a_xidx,
                                a_yidx,
                                b_xidx,
                                b_yidx,
                                vertical_limit_min_mov_xs * bone_distance,
                                vertical_limit_min_mov_ys * bone_distance,
                                vertical_limit_min_mov_zs * bone_distance,
                                vertical_limit_max_mov_xs * bone_distance,
                                vertical_limit_max_mov_ys * bone_distance,
                                vertical_limit_max_mov_zs * bone_distance,
                                vertical_limit_min_rot_xs,
                                vertical_limit_min_rot_ys,
                                vertical_limit_min_rot_zs,
                                vertical_limit_max_rot_xs,
                                vertical_limit_max_rot_ys,
                                vertical_limit_max_rot_zs,
                                vertical_spring_constant_mov_xs,
                                vertical_spring_constant_mov_ys,
                                vertical_spring_constant_mov_zs,
                                vertical_spring_constant_rot_xs,
                                vertical_spring_constant_rot_ys,
                                vertical_spring_constant_rot_zs,
                            )
                            if joint.name not in [j.name for j in created_joints.values()]:
                                created_joints[joint_key] = joint

                            if param_option["vertical_reverse_joint"]:
                                # 縦逆ジョイント
                                a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                                b_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)
                                # スリットで上の剛体が見つからない場合、ひとつ前の頂点マップを確認する
                                if not b_rigidbody:
                                    b_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx - 1, None)

                                a_xidx = v_xidx
                                a_yidx = v_yidx
                                b_xidx = v_xidx
                                b_yidx = above_yidx

                                a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                    # 剛体が重なる箇所の交点
                                    above_mat = MMatrix4x4()
                                    above_point = MVector3D()
                                    if a_rigidbody:
                                        above_mat.setToIdentity()
                                        above_mat.translate(a_rigidbody.shape_position)
                                        above_mat.rotate(a_rigidbody.shape_qq)
                                        above_point = above_mat * MVector3D(0, a_rigidbody.shape_size.y(), 0)

                                    now_mat = MMatrix4x4()
                                    now_mat.setToIdentity()
                                    now_mat.translate(b_rigidbody.shape_position)
                                    now_mat.rotate(b_rigidbody.shape_qq)
                                    now_point = now_mat * MVector3D(0, -b_rigidbody.shape_size.y(), 0)

                                    if v_yidx == 0:
                                        joint_pos = above_point
                                    else:
                                        joint_pos = (above_point + now_point) / 2
                                else:
                                    joint_pos = b_pos

                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                if v_yidx == 0:
                                    z_direction_pos = b_rigidbody.z_direction.normalized()
                                else:
                                    z_direction_pos = (
                                        (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                    ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                bone_distance = a_pos.distanceToPoint(b_pos)

                                joint_key, joint = self.build_joint(
                                    "↑",
                                    11,
                                    bone_y_idx,
                                    a_rigidbody,
                                    b_rigidbody,
                                    joint_pos,
                                    joint_qq,
                                    base_map_idx,
                                    a_xidx,
                                    a_yidx,
                                    b_xidx,
                                    b_yidx,
                                    vertical_reverse_limit_min_mov_xs * bone_distance,
                                    vertical_reverse_limit_min_mov_ys * bone_distance,
                                    vertical_reverse_limit_min_mov_zs * bone_distance,
                                    vertical_reverse_limit_max_mov_xs * bone_distance,
                                    vertical_reverse_limit_max_mov_ys * bone_distance,
                                    vertical_reverse_limit_max_mov_zs * bone_distance,
                                    vertical_reverse_limit_min_rot_xs,
                                    vertical_reverse_limit_min_rot_ys,
                                    vertical_reverse_limit_min_rot_zs,
                                    vertical_reverse_limit_max_rot_xs,
                                    vertical_reverse_limit_max_rot_ys,
                                    vertical_reverse_limit_max_rot_zs,
                                    vertical_reverse_spring_constant_mov_xs,
                                    vertical_reverse_spring_constant_mov_ys,
                                    vertical_reverse_spring_constant_mov_zs,
                                    vertical_reverse_spring_constant_rot_xs,
                                    vertical_reverse_spring_constant_rot_ys,
                                    vertical_reverse_spring_constant_rot_zs,
                                )
                                if joint.name not in [j.name for j in created_joints.values()]:
                                    created_joints[joint_key] = joint

                            # バランサー剛体が必要な場合
                            if param_option["rigidbody_balancer"]:
                                a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                                b_rigidbody = now_now_vv.map_balance_rigidbodies.get(base_map_idx, None)

                                a_xidx = v_xidx
                                a_yidx = v_yidx
                                b_xidx = v_xidx
                                b_yidx = v_yidx

                                if not (
                                    a_rigidbody
                                    and b_rigidbody
                                    and a_rigidbody.index != b_rigidbody.index
                                    and a_rigidbody.index >= 0
                                    and b_rigidbody.index >= 0
                                ):
                                    logger.warning(
                                        "バランサー剛体ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                                        vv.map_bones[base_map_idx].name
                                        if vv.map_bones.get(base_map_idx, None)
                                        else vv.vidxs(),
                                    )
                                else:
                                    joint_axis_up = (
                                        now_now_vv.map_bones[base_map_idx].position
                                        - now_above_vv.map_bones[base_map_idx].position
                                    ).normalized()
                                    joint_axis = (
                                        now_now_vv.map_balance_rigidbodies[base_map_idx].shape_position
                                        - now_now_vv.map_bones[base_map_idx].position
                                    ).normalized()
                                    joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                                    joint_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                                    joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                    joint_key, joint = self.build_joint(
                                        "B",
                                        13,
                                        0,
                                        a_rigidbody,
                                        b_rigidbody,
                                        a_rigidbody.shape_position.copy(),
                                        MQuaternion(),
                                        base_map_idx,
                                        a_xidx,
                                        a_yidx,
                                        b_xidx,
                                        b_yidx,
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [0],
                                        [100000],
                                        [100000],
                                        [100000],
                                        [100000],
                                        [100000],
                                        [100000],
                                    )
                                    if joint.name not in [j.name for j in created_joints.values()]:
                                        created_joints[joint_key] = joint

                                    a_rigidbody = now_above_vv.map_balance_rigidbodies.get(base_map_idx, None)
                                    b_rigidbody = now_now_vv.map_balance_rigidbodies.get(base_map_idx, None)

                                    a_xidx = v_xidx
                                    a_yidx = above_yidx
                                    b_xidx = v_xidx
                                    b_yidx = v_yidx

                                    if (
                                        a_rigidbody
                                        and b_rigidbody
                                        and a_rigidbody.index != b_rigidbody.index
                                        and a_rigidbody.index >= 0
                                        and b_rigidbody.index >= 0
                                    ):
                                        # バランサー補助剛体
                                        joint_key, joint = self.build_joint(
                                            "BS",
                                            14,
                                            0,
                                            a_rigidbody,
                                            b_rigidbody,
                                            MVector3D(),
                                            MQuaternion(),
                                            base_map_idx,
                                            a_xidx,
                                            a_yidx,
                                            b_xidx,
                                            b_yidx,
                                            [-50],
                                            [-50],
                                            [-50],
                                            [50],
                                            [50],
                                            [50],
                                            [1],
                                            [1],
                                            [1],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                            [0],
                                        )
                                        if joint.name not in [j.name for j in created_joints.values()]:
                                            created_joints[joint_key] = joint

                    if param_option["horizonal_joint"] and not is_center:
                        # 横ジョイント
                        a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                        b_rigidbody = next_now_vv.map_rigidbodies.get(next_map_idx, None)

                        a_xidx = v_xidx
                        a_yidx = v_yidx
                        b_xidx = next_xidx
                        b_yidx = v_yidx

                        if not b_rigidbody or (not next_connected and prev_connected):
                            # 横がない場合、もしくは繋がってない場合、入れ替える
                            a_rigidbody = prev_now_vv.map_rigidbodies.get(prev_map_idx, None)
                            b_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)

                            a_xidx = prev_xidx
                            a_yidx = v_yidx
                            b_xidx = v_xidx
                            b_yidx = v_yidx

                        if v_yidx < registered_max_v_yidx and not (
                            a_rigidbody and b_rigidbody and a_rigidbody.index >= 0 and b_rigidbody.index >= 0
                        ):
                            logger.warning(
                                "横ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                                vv.map_bones[base_map_idx].name
                                if vv.map_bones.get(base_map_idx, None)
                                else vv.vidxs(),
                            )
                        # elif not (
                        #     v_xidx < vertex_map.shape[1] - 1 or (v_xidx == vertex_map.shape[1] - 1 and next_connected)
                        # ):
                        #     # 端の場合は最初と繋がっていることを条件とする
                        #     pass
                        elif a_rigidbody and b_rigidbody and a_rigidbody.index == b_rigidbody.index:
                            # 同じ剛体なのは同一頂点からボーンが出る場合に有り得るので、警告は出さない
                            pass
                        elif v_yidx < registered_max_v_yidx:
                            a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                            b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                            if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                # 剛体が重なる箇所の交点
                                above_mat = MMatrix4x4()
                                above_point = MVector3D()
                                if a_rigidbody:
                                    above_mat.setToIdentity()
                                    above_mat.translate(a_rigidbody.shape_position)
                                    above_mat.rotate(a_rigidbody.shape_qq)
                                    above_point = above_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                                now_mat = MMatrix4x4()
                                now_mat.setToIdentity()
                                now_mat.translate(b_rigidbody.shape_position)
                                now_mat.rotate(b_rigidbody.shape_qq)
                                now_point = now_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                                joint_pos = (above_point + now_point) / 2
                            else:
                                joint_pos = b_pos

                            # ボーン進行方向(x)
                            x_direction_pos = (b_pos - a_pos).normalized()
                            # ボーン進行方向に対しての縦軸(z)
                            z_direction_pos = ((a_rigidbody.z_direction + b_rigidbody.z_direction) / 2).normalized()
                            joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                            joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                            ratio = 1
                            if param_option["horizonal_joint_restruct"]:
                                # 親剛体との距離制限を加える場合
                                for n in range(6):
                                    check_ratio = 1 + (n * 0.06)

                                    a_mat = MMatrix4x4()
                                    a_mat.setToIdentity()
                                    a_mat.translate(a_rigidbody.shape_position)
                                    a_mat.rotate(a_rigidbody.shape_qq)

                                    a_length = np.mean(a_rigidbody.shape_size.data()) * check_ratio

                                    a_capsule = MCapsule(
                                        MSegment(
                                            a_mat * MVector3D(-a_length, 0, 0),
                                            a_mat * MVector3D(a_length, 0, 0),
                                        ),
                                        a_length * check_ratio,
                                    )

                                    a_col = is_col_capsule_capsule(parent_capsule, a_capsule)
                                    b_col = False

                                    if not a_col:
                                        b_mat = MMatrix4x4()
                                        b_mat.setToIdentity()
                                        b_mat.translate(b_rigidbody.shape_position)
                                        b_mat.rotate(b_rigidbody.shape_qq)

                                        b_length = np.mean(b_rigidbody.shape_size.data()) * check_ratio

                                        b_capsule = MCapsule(
                                            MSegment(
                                                b_mat * MVector3D(-b_length, 0, 0),
                                                b_mat * MVector3D(b_length, 0, 0),
                                            ),
                                            b_length * check_ratio,
                                        )

                                        b_col = is_col_capsule_capsule(parent_capsule, b_capsule)

                                    if a_col or b_col:
                                        # 剛体のどちらかが衝突していたらその分動きを制限する
                                        ratio = n * 0.05
                                        logger.info(
                                            "親剛体と距離が近接しているため、横ジョイントの可動域を制限します。剛体A: %s, 剛体B: %s, 制限率: %s",
                                            a_rigidbody.name,
                                            b_rigidbody.name,
                                            round(ratio, 3),
                                        )
                                        break

                            bone_distance = a_pos.distanceToPoint(b_pos)

                            joint_key, joint = self.build_joint(
                                "→",
                                21,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                base_map_idx,
                                a_xidx,
                                a_yidx,
                                b_xidx,
                                b_yidx,
                                horizonal_limit_min_mov_xs * bone_distance,
                                horizonal_limit_min_mov_ys * bone_distance,
                                horizonal_limit_min_mov_zs * bone_distance,
                                horizonal_limit_max_mov_xs * bone_distance,
                                horizonal_limit_max_mov_ys * bone_distance,
                                horizonal_limit_max_mov_zs * bone_distance,
                                horizonal_limit_min_rot_xs,
                                horizonal_limit_min_rot_ys,
                                horizonal_limit_min_rot_zs,
                                horizonal_limit_max_rot_xs,
                                horizonal_limit_max_rot_ys,
                                horizonal_limit_max_rot_zs,
                                horizonal_spring_constant_mov_xs,
                                horizonal_spring_constant_mov_ys,
                                horizonal_spring_constant_mov_zs,
                                horizonal_spring_constant_rot_xs,
                                horizonal_spring_constant_rot_ys,
                                horizonal_spring_constant_rot_zs,
                                ratio,
                            )
                            if joint.name not in [j.name for j in created_joints.values()]:
                                created_joints[joint_key] = joint

                        if param_option["joint_pos_type"] == logger.transtext("ボーン位置") and v_yidx == actual_v_yidx:
                            # 裾横ジョイント
                            # ひとつ上のボーンに紐付くジョイントをチェック
                            a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                            b_rigidbody = next_now_vv.map_rigidbodies.get(next_map_idx, None)

                            a_xidx = v_xidx
                            a_yidx = v_yidx
                            b_xidx = next_xidx
                            b_yidx = v_yidx

                            if (
                                next_connected
                                and a_rigidbody
                                and b_rigidbody
                                and a_rigidbody.index >= 0
                                and b_rigidbody.index >= 0
                            ):
                                a_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_position
                                )

                                b_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_position
                                )

                                joint_pos = b_pos

                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                bone_distance = a_pos.distanceToPoint(b_pos)

                                joint_key, joint = self.build_joint(
                                    "→",
                                    21,
                                    bone_y_idx,
                                    a_rigidbody,
                                    b_rigidbody,
                                    joint_pos,
                                    joint_qq,
                                    base_map_idx,
                                    a_xidx,
                                    a_yidx + 1,
                                    b_xidx,
                                    b_yidx + 1,
                                    horizonal_limit_min_mov_xs * bone_distance,
                                    horizonal_limit_min_mov_ys * bone_distance,
                                    horizonal_limit_min_mov_zs * bone_distance,
                                    horizonal_limit_max_mov_xs * bone_distance,
                                    horizonal_limit_max_mov_ys * bone_distance,
                                    horizonal_limit_max_mov_zs * bone_distance,
                                    horizonal_limit_min_rot_xs,
                                    horizonal_limit_min_rot_ys,
                                    horizonal_limit_min_rot_zs,
                                    horizonal_limit_max_rot_xs,
                                    horizonal_limit_max_rot_ys,
                                    horizonal_limit_max_rot_zs,
                                    horizonal_spring_constant_mov_xs,
                                    horizonal_spring_constant_mov_ys,
                                    horizonal_spring_constant_mov_zs,
                                    horizonal_spring_constant_rot_xs,
                                    horizonal_spring_constant_rot_ys,
                                    horizonal_spring_constant_rot_zs,
                                    ratio,
                                    override_joint_name=f"→|{a_rigidbody.name}T|{b_rigidbody.name}T",
                                )
                                if joint.name not in [j.name for j in created_joints.values()]:
                                    created_joints[joint_key] = joint

                        last_v_xidx = np.max(np.sort(np.where(registered_bones[v_yidx, :])[0])[-2:])
                        (
                            last_prev_map_idx,
                            last_prev_xidx,
                            last_prev_connected,
                            last_next_map_idx,
                            last_next_xidx,
                            last_next_connected,
                            last_above_yidx,
                            last_below_yidx,
                            last_target_v_yidx,
                            last_target_v_xidx,
                            last_registered_max_v_yidx,
                            last_registered_max_v_xidx,
                            last_max_v_yidx,
                            last_max_v_xidx,
                        ) = self.get_block_vidxs(
                            v_yidx,
                            last_v_xidx,
                            vertex_maps,
                            all_registered_bones,
                            all_bone_connected,
                            base_map_idx,
                            is_center=is_center,
                        )

                        if (
                            param_option["joint_pos_type"] == logger.transtext("ボーン位置")
                            and v_xidx == registered_max_v_xidx
                        ):
                            # 横末端ジョイント
                            # ひとつ上のボーンに紐付くジョイントをチェック
                            a_rigidbody = prev_now_vv.map_rigidbodies.get(prev_map_idx, None)
                            b_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)

                            a_xidx = prev_xidx
                            a_yidx = v_yidx
                            b_xidx = v_xidx
                            b_yidx = v_yidx

                            if (
                                not next_connected
                                and prev_connected
                                and a_rigidbody
                                and b_rigidbody
                                and a_rigidbody.index >= 0
                                and b_rigidbody.index >= 0
                            ):
                                a_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_position
                                )

                                b_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_position
                                )

                                joint_pos = b_pos

                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                bone_distance = a_pos.distanceToPoint(b_pos)

                                joint_key, joint = self.build_joint(
                                    "→",
                                    22,
                                    bone_y_idx,
                                    a_rigidbody,
                                    b_rigidbody,
                                    joint_pos,
                                    joint_qq,
                                    base_map_idx,
                                    a_xidx,
                                    a_yidx + 1,
                                    b_xidx,
                                    b_yidx + 1,
                                    horizonal_limit_min_mov_xs * bone_distance,
                                    horizonal_limit_min_mov_ys * bone_distance,
                                    horizonal_limit_min_mov_zs * bone_distance,
                                    horizonal_limit_max_mov_xs * bone_distance,
                                    horizonal_limit_max_mov_ys * bone_distance,
                                    horizonal_limit_max_mov_zs * bone_distance,
                                    horizonal_limit_min_rot_xs,
                                    horizonal_limit_min_rot_ys,
                                    horizonal_limit_min_rot_zs,
                                    horizonal_limit_max_rot_xs,
                                    horizonal_limit_max_rot_ys,
                                    horizonal_limit_max_rot_zs,
                                    horizonal_spring_constant_mov_xs,
                                    horizonal_spring_constant_mov_ys,
                                    horizonal_spring_constant_mov_zs,
                                    horizonal_spring_constant_rot_xs,
                                    horizonal_spring_constant_rot_ys,
                                    horizonal_spring_constant_rot_zs,
                                    ratio,
                                    override_joint_name=f"→|{a_rigidbody.name}T|{b_rigidbody.name}T",
                                )
                                if joint.name not in [j.name for j in created_joints.values()]:
                                    created_joints[joint_key] = joint

                            # ---------------------------
                            # 横末端逆ジョイント
                            a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                            b_rigidbody = prev_now_vv.map_rigidbodies.get(prev_map_idx, None)

                            a_xidx = v_xidx
                            a_yidx = v_yidx
                            b_xidx = prev_xidx
                            b_yidx = v_yidx

                            if (
                                not next_connected
                                and prev_connected
                                and a_rigidbody
                                and b_rigidbody
                                and a_rigidbody.index >= 0
                                and b_rigidbody.index >= 0
                            ):
                                # 末端横逆ジョイント
                                a_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_position
                                )

                                b_pos = (
                                    model.bones[
                                        model.bone_indexes[
                                            model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index
                                        ]
                                    ].position
                                    if model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index >= 0
                                    else model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                    + model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_position
                                )

                                joint_pos = b_pos

                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                bone_distance = a_pos.distanceToPoint(b_pos)

                                joint_key, joint = self.build_joint(
                                    "←",
                                    23,
                                    bone_y_idx,
                                    a_rigidbody,
                                    b_rigidbody,
                                    joint_pos,
                                    joint_qq,
                                    base_map_idx,
                                    a_xidx,
                                    a_yidx + 1,
                                    b_xidx,
                                    b_yidx + 1,
                                    horizonal_limit_min_mov_xs * bone_distance,
                                    horizonal_limit_min_mov_ys * bone_distance,
                                    horizonal_limit_min_mov_zs * bone_distance,
                                    horizonal_limit_max_mov_xs * bone_distance,
                                    horizonal_limit_max_mov_ys * bone_distance,
                                    horizonal_limit_max_mov_zs * bone_distance,
                                    horizonal_limit_min_rot_xs * 0 + 1,
                                    horizonal_limit_min_rot_ys * 0 + 1,
                                    horizonal_limit_min_rot_zs * 0 + 1,
                                    horizonal_limit_max_rot_xs * 0,
                                    horizonal_limit_max_rot_ys * 0,
                                    horizonal_limit_max_rot_zs * 0,
                                    horizonal_spring_constant_mov_xs,
                                    horizonal_spring_constant_mov_ys,
                                    horizonal_spring_constant_mov_zs,
                                    horizonal_spring_constant_rot_xs * 0,
                                    horizonal_spring_constant_rot_ys * 0,
                                    horizonal_spring_constant_rot_zs * 0,
                                    ratio,
                                    override_joint_name=f"←|{a_rigidbody.name}T|{b_rigidbody.name}T",
                                )
                                if joint.name not in [j.name for j in created_joints.values()]:
                                    created_joints[joint_key] = joint

                        if param_option["horizonal_reverse_joint"]:
                            # 横逆ジョイント
                            a_rigidbody = next_now_vv.map_rigidbodies.get(next_map_idx, None)
                            b_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)

                            a_xidx = next_xidx
                            a_yidx = v_yidx
                            b_xidx = v_xidx
                            b_yidx = v_yidx

                            if a_rigidbody and b_rigidbody and a_rigidbody.index >= 0 and b_rigidbody.index >= 0:
                                a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                    # 剛体が重なる箇所の交点
                                    above_mat = MMatrix4x4()
                                    above_point = MVector3D()
                                    if a_rigidbody:
                                        above_mat.setToIdentity()
                                        above_mat.translate(a_rigidbody.shape_position)
                                        above_mat.rotate(a_rigidbody.shape_qq)
                                        above_point = above_mat * MVector3D(-a_rigidbody.shape_size.x(), 0, 0)

                                    now_mat = MMatrix4x4()
                                    now_mat.setToIdentity()
                                    now_mat.translate(b_rigidbody.shape_position)
                                    now_mat.rotate(b_rigidbody.shape_qq)
                                    now_point = now_mat * MVector3D(b_rigidbody.shape_size.x(), 0, 0)

                                    joint_pos = (above_point + now_point) / 2
                                else:
                                    joint_pos = b_pos

                                # ボーン進行方向(x)
                                x_direction_pos = (b_pos - a_pos).normalized()
                                # ボーン進行方向に対しての縦軸(z)
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                                joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                bone_distance = a_pos.distanceToPoint(b_pos)

                                joint_key, joint = self.build_joint(
                                    "←",
                                    23,
                                    bone_y_idx,
                                    a_rigidbody,
                                    b_rigidbody,
                                    joint_pos,
                                    joint_qq,
                                    base_map_idx,
                                    a_xidx,
                                    a_yidx,
                                    b_xidx,
                                    b_yidx,
                                    horizonal_reverse_limit_min_mov_xs * bone_distance,
                                    horizonal_reverse_limit_min_mov_ys * bone_distance,
                                    horizonal_reverse_limit_min_mov_zs * bone_distance,
                                    horizonal_reverse_limit_max_mov_xs * bone_distance,
                                    horizonal_reverse_limit_max_mov_ys * bone_distance,
                                    horizonal_reverse_limit_max_mov_zs * bone_distance,
                                    horizonal_reverse_limit_min_rot_xs,
                                    horizonal_reverse_limit_min_rot_ys,
                                    horizonal_reverse_limit_min_rot_zs,
                                    horizonal_reverse_limit_max_rot_xs,
                                    horizonal_reverse_limit_max_rot_ys,
                                    horizonal_reverse_limit_max_rot_zs,
                                    horizonal_reverse_spring_constant_mov_xs,
                                    horizonal_reverse_spring_constant_mov_ys,
                                    horizonal_reverse_spring_constant_mov_zs,
                                    horizonal_reverse_spring_constant_rot_xs,
                                    horizonal_reverse_spring_constant_rot_ys,
                                    horizonal_reverse_spring_constant_rot_zs,
                                    ratio,
                                )
                                if joint.name not in [j.name for j in created_joints.values()]:
                                    created_joints[joint_key] = joint

                            if (
                                param_option["joint_pos_type"] == logger.transtext("ボーン位置")
                                and v_yidx == registered_max_v_yidx
                            ):
                                # 末端横ジョイント
                                # ひとつ上のボーンに紐付くジョイントをチェック
                                a_rigidbody = next_above_vv.map_rigidbodies.get(next_map_idx, None)
                                b_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)

                                a_xidx = next_xidx
                                a_yidx = above_yidx
                                b_xidx = v_xidx
                                b_yidx = above_yidx

                                if a_rigidbody and b_rigidbody and a_rigidbody.index >= 0 and b_rigidbody.index >= 0:
                                    # 末端横逆ジョイント
                                    a_pos = (
                                        model.bones[
                                            model.bone_indexes[
                                                model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index
                                            ]
                                        ].position
                                        if model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_index >= 0
                                        else model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                                        + model.bones[model.bone_indexes[a_rigidbody.bone_index]].tail_position
                                    )

                                    b_pos = (
                                        model.bones[
                                            model.bone_indexes[
                                                model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index
                                            ]
                                        ].position
                                        if model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_index >= 0
                                        else model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                                        + model.bones[model.bone_indexes[b_rigidbody.bone_index]].tail_position
                                    )

                                    joint_pos = b_pos

                                    # ボーン進行方向(x)
                                    x_direction_pos = (b_pos - a_pos).normalized()
                                    # ボーン進行方向に対しての縦軸(z)
                                    z_direction_pos = (
                                        (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                    ).normalized()
                                    joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                                    joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                                    bone_distance = a_pos.distanceToPoint(b_pos)

                                    joint_key, joint = self.build_joint(
                                        "←",
                                        23,
                                        bone_y_idx,
                                        a_rigidbody,
                                        b_rigidbody,
                                        joint_pos,
                                        joint_qq,
                                        base_map_idx,
                                        a_xidx,
                                        a_yidx + 1,
                                        b_xidx,
                                        b_yidx + 1,
                                        horizonal_limit_min_mov_xs * bone_distance,
                                        horizonal_limit_min_mov_ys * bone_distance,
                                        horizonal_limit_min_mov_zs * bone_distance,
                                        horizonal_limit_max_mov_xs * bone_distance,
                                        horizonal_limit_max_mov_ys * bone_distance,
                                        horizonal_limit_max_mov_zs * bone_distance,
                                        horizonal_limit_min_rot_xs,
                                        horizonal_limit_min_rot_ys,
                                        horizonal_limit_min_rot_zs,
                                        horizonal_limit_max_rot_xs,
                                        horizonal_limit_max_rot_ys,
                                        horizonal_limit_max_rot_zs,
                                        horizonal_spring_constant_mov_xs,
                                        horizonal_spring_constant_mov_ys,
                                        horizonal_spring_constant_mov_zs,
                                        horizonal_spring_constant_rot_xs,
                                        horizonal_spring_constant_rot_ys,
                                        horizonal_spring_constant_rot_zs,
                                        ratio,
                                        override_joint_name=f"←|{a_rigidbody.name}T|{b_rigidbody.name}T",
                                    )
                                    if joint.name not in [j.name for j in created_joints.values()]:
                                        created_joints[joint_key] = joint

                    if param_option["diagonal_joint"] and next_connected and next_below_vv and not is_center:
                        # 斜めジョイント
                        a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                        b_rigidbody = next_below_vv.map_rigidbodies.get(next_map_idx, None)

                        a_xidx = v_xidx
                        a_yidx = v_yidx
                        b_xidx = next_xidx
                        b_yidx = below_yidx

                        if not (
                            a_rigidbody
                            and b_rigidbody
                            and a_rigidbody.index >= 0
                            and b_rigidbody.index >= 0
                            and a_rigidbody.index != b_rigidbody.index
                        ):
                            if (
                                registered_bones.shape[0] - 1 > below_yidx
                                and registered_bones.shape[1] > next_xidx
                                and registered_bones[below_yidx, next_xidx]
                                and v_yidx < registered_max_v_yidx
                            ):
                                logger.warning(
                                    "斜め（＼）ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                                    vv.map_bones[base_map_idx].name
                                    if vv.map_bones.get(base_map_idx, None)
                                    else vv.vidxs(),
                                )
                        else:
                            a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                            b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                            if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                # 剛体が重なる箇所の交点
                                above_mat = MMatrix4x4()
                                above_point = MVector3D()
                                if a_rigidbody:
                                    above_mat.setToIdentity()
                                    above_mat.translate(a_rigidbody.shape_position)
                                    above_mat.rotate(a_rigidbody.shape_qq)
                                    above_point = above_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                                now_mat = MMatrix4x4()
                                now_mat.setToIdentity()
                                now_mat.translate(b_rigidbody.shape_position)
                                now_mat.rotate(b_rigidbody.shape_qq)
                                now_point = now_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                                joint_pos = (above_point + now_point) / 2
                            else:
                                joint_pos = b_pos

                            # ボーン進行方向(x)
                            x_direction_pos = (b_pos - a_pos).normalized()
                            # ボーン進行方向に対しての縦軸(z)
                            if v_yidx == 0:
                                z_direction_pos = b_rigidbody.z_direction.normalized()
                            else:
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                            joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                            joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                            bone_distance = a_pos.distanceToPoint(b_pos)

                            joint_key, joint = self.build_joint(
                                "＼",
                                31,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                base_map_idx,
                                a_xidx,
                                a_yidx,
                                b_xidx,
                                b_yidx,
                                diagonal_limit_min_mov_xs * bone_distance,
                                diagonal_limit_min_mov_ys * bone_distance,
                                diagonal_limit_min_mov_zs * bone_distance,
                                diagonal_limit_max_mov_xs * bone_distance,
                                diagonal_limit_max_mov_ys * bone_distance,
                                diagonal_limit_max_mov_zs * bone_distance,
                                diagonal_limit_min_rot_xs,
                                diagonal_limit_min_rot_ys,
                                diagonal_limit_min_rot_zs,
                                diagonal_limit_max_rot_xs,
                                diagonal_limit_max_rot_ys,
                                diagonal_limit_max_rot_zs,
                                diagonal_spring_constant_mov_xs,
                                diagonal_spring_constant_mov_ys,
                                diagonal_spring_constant_mov_zs,
                                diagonal_spring_constant_rot_xs,
                                diagonal_spring_constant_rot_ys,
                                diagonal_spring_constant_rot_zs,
                            )
                            if joint.name not in [j.name for j in created_joints.values()]:
                                created_joints[joint_key] = joint

                    if (
                        param_option["diagonal_joint"]
                        and v_yidx > 0
                        and next_connected
                        and next_above_vv
                        and not is_center
                    ):
                        # 斜めジョイント
                        a_rigidbody = now_now_vv.map_rigidbodies.get(base_map_idx, None)
                        b_rigidbody = next_above_vv.map_rigidbodies.get(next_map_idx, None)

                        a_xidx = v_xidx
                        a_yidx = v_yidx
                        b_xidx = next_xidx
                        b_yidx = above_yidx

                        if not (
                            a_rigidbody
                            and b_rigidbody
                            and a_rigidbody.index >= 0
                            and b_rigidbody.index >= 0
                            and a_rigidbody.index != b_rigidbody.index
                        ):
                            if (
                                registered_bones.shape[0] > above_yidx
                                and registered_bones.shape[1] > v_xidx
                                and registered_bones[above_yidx, v_xidx]
                                and v_yidx < registered_max_v_yidx
                            ):
                                logger.warning(
                                    "斜め（／）ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                                    vv.map_bones[base_map_idx].name
                                    if vv.map_bones.get(base_map_idx, None)
                                    else vv.vidxs(),
                                )
                        else:
                            a_pos = model.bones[model.bone_indexes[a_rigidbody.bone_index]].position
                            b_pos = model.bones[model.bone_indexes[b_rigidbody.bone_index]].position
                            if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                                # 剛体が重なる箇所の交点
                                above_mat = MMatrix4x4()
                                above_point = MVector3D()
                                if a_rigidbody:
                                    above_mat.setToIdentity()
                                    above_mat.translate(a_rigidbody.shape_position)
                                    above_mat.rotate(a_rigidbody.shape_qq)
                                    above_point = above_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                                now_mat = MMatrix4x4()
                                now_mat.setToIdentity()
                                now_mat.translate(b_rigidbody.shape_position)
                                now_mat.rotate(b_rigidbody.shape_qq)
                                now_point = now_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                                joint_pos = (above_point + now_point) / 2
                            else:
                                joint_pos = b_pos

                            # ボーン進行方向(x)
                            x_direction_pos = (b_pos - a_pos).normalized()
                            # ボーン進行方向に対しての縦軸(z)
                            if v_yidx == 0:
                                z_direction_pos = b_rigidbody.z_direction.normalized()
                            else:
                                z_direction_pos = (
                                    (a_rigidbody.z_direction + b_rigidbody.z_direction) / 2
                                ).normalized()
                            joint_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                            joint_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                            bone_distance = a_pos.distanceToPoint(b_pos)

                            joint_key, joint = self.build_joint(
                                "／",
                                32,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                base_map_idx,
                                a_xidx,
                                a_yidx,
                                b_xidx,
                                b_yidx,
                                diagonal_limit_min_mov_xs * bone_distance,
                                diagonal_limit_min_mov_ys * bone_distance,
                                diagonal_limit_min_mov_zs * bone_distance,
                                diagonal_limit_max_mov_xs * bone_distance,
                                diagonal_limit_max_mov_ys * bone_distance,
                                diagonal_limit_max_mov_zs * bone_distance,
                                diagonal_limit_min_rot_xs,
                                diagonal_limit_min_rot_ys,
                                diagonal_limit_min_rot_zs,
                                diagonal_limit_max_rot_xs,
                                diagonal_limit_max_rot_ys,
                                diagonal_limit_max_rot_zs,
                                diagonal_spring_constant_mov_xs,
                                diagonal_spring_constant_mov_ys,
                                diagonal_spring_constant_mov_zs,
                                diagonal_spring_constant_rot_xs,
                                diagonal_spring_constant_rot_ys,
                                diagonal_spring_constant_rot_zs,
                            )
                            if joint.name not in [j.name for j in created_joints.values()]:
                                created_joints[joint_key] = joint

                    if len(created_joints) > 0 and len(created_joints) // 50 > prev_joint_cnt:
                        logger.info("-- -- 【No.%s】ジョイント: %s個目:終了", base_map_idx + 1, len(created_joints))
                        prev_joint_cnt = len(created_joints) // 50

        logger.info("-- ジョイント: %s個目:終了", len(created_joints))

        for joint_key in sorted(created_joints.keys()):
            # ジョイントを登録
            joint = created_joints[joint_key]
            joint.index = len(model.joints)

            if joint.name in model.joints:
                logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
                joint.name += randomname(3)

            model.joints[joint.name] = joint
            logger.test(f"joint: {joint}")

    def build_joint(
        self,
        direction_mark: str,
        direction_idx: int,
        bone_y_idx: int,
        a_rigidbody: RigidBody,
        b_rigidbody: RigidBody,
        joint_pos: MVector3D,
        joint_qq: MQuaternion,
        base_map_idx: int,
        a_xidx: int,
        a_yidx: int,
        b_xidx: int,
        b_yidx: int,
        limit_min_mov_xs: np.ndarray,
        limit_min_mov_ys: np.ndarray,
        limit_min_mov_zs: np.ndarray,
        limit_max_mov_xs: np.ndarray,
        limit_max_mov_ys: np.ndarray,
        limit_max_mov_zs: np.ndarray,
        limit_min_rot_xs: np.ndarray,
        limit_min_rot_ys: np.ndarray,
        limit_min_rot_zs: np.ndarray,
        limit_max_rot_xs: np.ndarray,
        limit_max_rot_ys: np.ndarray,
        limit_max_rot_zs: np.ndarray,
        spring_constant_mov_xs: np.ndarray,
        spring_constant_mov_ys: np.ndarray,
        spring_constant_mov_zs: np.ndarray,
        spring_constant_rot_xs: np.ndarray,
        spring_constant_rot_ys: np.ndarray,
        spring_constant_rot_zs: np.ndarray,
        ratio=1,
        override_joint_name=None,
    ):
        joint_name = (
            f"{direction_mark}|{a_rigidbody.name}|{b_rigidbody.name}"
            if not override_joint_name
            else override_joint_name
        )
        joint_key = f"{direction_idx:02d}:{a_rigidbody.index:09d}:{b_rigidbody.index:09d}"

        joint_euler = joint_qq.toEulerAngles()
        joint_rotation_radians = MVector3D(
            math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z())
        )

        joint = Joint(
            joint_name,
            joint_name,
            0,
            a_rigidbody.index,
            b_rigidbody.index,
            joint_pos,
            joint_rotation_radians,
            MVector3D(
                limit_min_mov_xs[bone_y_idx] * ratio,
                limit_min_mov_ys[bone_y_idx] * ratio,
                limit_min_mov_zs[bone_y_idx] * ratio,
            ),
            MVector3D(
                limit_max_mov_xs[bone_y_idx] * ratio,
                limit_max_mov_ys[bone_y_idx] * ratio,
                limit_max_mov_zs[bone_y_idx] * ratio,
            ),
            MVector3D(
                math.radians(limit_min_rot_xs[bone_y_idx] * ratio),
                math.radians(limit_min_rot_ys[bone_y_idx] * ratio),
                math.radians(limit_min_rot_zs[bone_y_idx] * ratio),
            ),
            MVector3D(
                math.radians(limit_max_rot_xs[bone_y_idx] * ratio),
                math.radians(limit_max_rot_ys[bone_y_idx] * ratio),
                math.radians(limit_max_rot_zs[bone_y_idx] * ratio),
            ),
            MVector3D(
                spring_constant_mov_xs[bone_y_idx] * ratio,
                spring_constant_mov_ys[bone_y_idx] * ratio,
                spring_constant_mov_zs[bone_y_idx] * ratio,
            ),
            MVector3D(
                spring_constant_rot_xs[bone_y_idx] * ratio,
                spring_constant_rot_ys[bone_y_idx] * ratio,
                spring_constant_rot_zs[bone_y_idx] * ratio,
            ),
        )
        return joint_key, joint

    def create_joint_param(self, param_joint: Joint, vv_keys: np.ndarray, coefficient: float):
        max_vy = len(vv_keys)
        middle_vy = max_vy * 0.3
        min_vy = 0
        xs = np.arange(min_vy, max_vy, step=1)

        limit_min_mov_xs = []
        limit_min_mov_ys = []
        limit_min_mov_zs = []
        limit_max_mov_xs = []
        limit_max_mov_ys = []
        limit_max_mov_zs = []
        limit_min_rot_xs = []
        limit_min_rot_ys = []
        limit_min_rot_zs = []
        limit_max_rot_xs = []
        limit_max_rot_ys = []
        limit_max_rot_zs = []
        spring_constant_mov_xs = []
        spring_constant_mov_ys = []
        spring_constant_mov_zs = []
        spring_constant_rot_xs = []
        spring_constant_rot_ys = []
        spring_constant_rot_zs = []

        if param_joint:
            limit_min_mov_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_min.x() / coefficient,
                                param_joint.translation_limit_min.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_min_mov_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_min.y() / coefficient,
                                param_joint.translation_limit_min.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_min_mov_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_min.z() / coefficient,
                                param_joint.translation_limit_min.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

            limit_max_mov_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_max.x() / coefficient,
                                param_joint.translation_limit_max.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_max_mov_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_max.y() / coefficient,
                                param_joint.translation_limit_max.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_max_mov_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, max_vy],
                            [
                                param_joint.translation_limit_max.z() / coefficient,
                                param_joint.translation_limit_max.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

            limit_min_rot_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_min.x() / coefficient,
                                param_joint.rotation_limit_min.x() / coefficient,
                                param_joint.rotation_limit_min.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_min_rot_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_min.y() / coefficient,
                                param_joint.rotation_limit_min.y() / coefficient,
                                param_joint.rotation_limit_min.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_min_rot_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_min.z() / coefficient,
                                param_joint.rotation_limit_min.z() / coefficient,
                                param_joint.rotation_limit_min.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

            limit_max_rot_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_max.x() / coefficient,
                                param_joint.rotation_limit_max.x() / coefficient,
                                param_joint.rotation_limit_max.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_max_rot_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_max.y() / coefficient,
                                param_joint.rotation_limit_max.y() / coefficient,
                                param_joint.rotation_limit_max.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            limit_max_rot_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.rotation_limit_max.z() / coefficient,
                                param_joint.rotation_limit_max.z() / coefficient,
                                param_joint.rotation_limit_max.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

            spring_constant_mov_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_translation.x() / coefficient,
                                param_joint.spring_constant_translation.x() / coefficient,
                                param_joint.spring_constant_translation.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            spring_constant_mov_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_translation.y() / coefficient,
                                param_joint.spring_constant_translation.y() / coefficient,
                                param_joint.spring_constant_translation.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            spring_constant_mov_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_translation.z() / coefficient,
                                param_joint.spring_constant_translation.z() / coefficient,
                                param_joint.spring_constant_translation.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

            spring_constant_rot_xs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_rotation.x() / coefficient,
                                param_joint.spring_constant_rotation.x() / coefficient,
                                param_joint.spring_constant_rotation.x(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            spring_constant_rot_ys = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_rotation.y() / coefficient,
                                param_joint.spring_constant_rotation.y() / coefficient,
                                param_joint.spring_constant_rotation.y(),
                            ],
                        ]
                    )
                ),
                xs,
            )
            spring_constant_rot_zs = MBezierUtils.intersect_by_x(
                bezier.Curve.from_nodes(
                    np.asfortranarray(
                        [
                            [min_vy, middle_vy, max_vy],
                            [
                                param_joint.spring_constant_rotation.z() / coefficient,
                                param_joint.spring_constant_rotation.z() / coefficient,
                                param_joint.spring_constant_rotation.z(),
                            ],
                        ]
                    )
                ),
                xs,
            )

        return (
            limit_min_mov_xs,
            limit_min_mov_ys,
            limit_min_mov_zs,
            limit_max_mov_xs,
            limit_max_mov_ys,
            limit_max_mov_zs,
            limit_min_rot_xs,
            limit_min_rot_ys,
            limit_min_rot_zs,
            limit_max_rot_xs,
            limit_max_rot_ys,
            limit_max_rot_zs,
            spring_constant_mov_xs,
            spring_constant_mov_ys,
            spring_constant_mov_zs,
            spring_constant_rot_xs,
            spring_constant_rot_ys,
            spring_constant_rot_zs,
        )

    def create_rigidbody(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        all_registered_bones: dict,
        all_bone_connected: dict,
        root_bone: Bone,
        left_root_bone: Bone,
        right_root_bone: Bone,
        base_vertical_axis: MVector3D,
        base_reverse_axis: MVector3D,
        is_material_horizonal: bool,
    ):
        logger.info("【%s:%s】剛体生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        # 剛体生成
        created_rigidbodies = {}
        created_rigidbody_vvs = {}
        prev_rigidbody_cnt = 0

        # 剛体情報
        param_rigidbody = param_option["rigidbody"]
        # 剛体係数
        coefficient = param_option["rigidbody_coefficient"]
        # 剛体形状
        rigidbody_shape_type = param_option["rigidbody_shape_type"]
        # 中央配置か否か
        is_center = param_option["density_type"] == logger.transtext("中央")

        # 親ボーンに紐付く剛体がある場合、それを利用
        parent_bone = model.bones[param_option["parent_bone_name"]]
        parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)

        if not parent_bone_rigidbody:
            # 親ボーンに紐付く剛体がない場合、自前で作成
            parent_bone_rigidbody = RigidBody(
                parent_bone.name,
                parent_bone.english_name,
                parent_bone.index,
                param_rigidbody.collision_group,
                param_rigidbody.no_collision_group,
                0,
                MVector3D(1, 1, 1),
                parent_bone.position,
                MVector3D(),
                1,
                0.5,
                0.5,
                0,
                0,
                0,
            )
            parent_bone_rigidbody.index = len(model.rigidbodies)

            if parent_bone_rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
                parent_bone_rigidbody.name += randomname(3)

            model.rigidbodies[parent_bone.name] = parent_bone_rigidbody

        # 全非衝突対象
        all_no_collision_group = 0

        top_bone_positions = []
        for vertex_map in vertex_maps.values():
            for vkey in vertex_map[0, :]:
                if np.isnan(vkey).any() or tuple(vkey) not in virtual_vertices:
                    continue
                top_bone_positions.append(virtual_vertices[tuple(vkey)].position().data())

        root_rigidbody = left_root_rigidbody = right_root_rigidbody = None
        if param_option["parent_type"] == logger.transtext("中心"):
            if root_bone:
                root_rigidbody = self.get_rigidbody(model, root_bone.name)
                if not root_rigidbody:
                    # 中心剛体を接触なしボーン追従剛体で生成
                    root_rigidbody = RigidBody(
                        root_bone.name,
                        root_bone.english_name,
                        root_bone.index,
                        param_rigidbody.collision_group,
                        0,
                        0,
                        MVector3D(0.5, 0.5, 0.5),
                        MVector3D(np.mean(top_bone_positions, axis=0)),
                        MVector3D(),
                        1,
                        0.5,
                        0.5,
                        0,
                        0,
                        0,
                    )
                    root_rigidbody.index = len(model.rigidbodies)
                    model.rigidbodies[root_rigidbody.name] = root_rigidbody
            else:
                # 左右に別れる場合、左右剛体を作成する
                if left_root_bone:
                    left_root_rigidbody = self.get_rigidbody(model, left_root_bone.name)
                    left_root_rigidbody = RigidBody(
                        left_root_bone.name,
                        left_root_bone.english_name,
                        left_root_bone.index,
                        param_rigidbody.collision_group,
                        0,
                        0,
                        MVector3D(0.5, 0.5, 0.5),
                        left_root_bone.position.copy(),
                        MVector3D(),
                        1,
                        0.5,
                        0.5,
                        0,
                        0,
                        0,
                    )
                    left_root_rigidbody.index = len(model.rigidbodies)
                    model.rigidbodies[left_root_rigidbody.name] = left_root_rigidbody
                if right_root_bone:
                    right_root_rigidbody = self.get_rigidbody(model, right_root_bone.name)
                    right_root_rigidbody = RigidBody(
                        right_root_bone.name,
                        right_root_bone.english_name,
                        right_root_bone.index,
                        param_rigidbody.collision_group,
                        0,
                        0,
                        MVector3D(0.5, 0.5, 0.5),
                        right_root_bone.position.copy(),
                        MVector3D(),
                        1,
                        0.5,
                        0.5,
                        0,
                        0,
                        0,
                    )
                    right_root_rigidbody.index = len(model.rigidbodies)
                    model.rigidbodies[right_root_rigidbody.name] = right_root_rigidbody
        else:
            # 中心剛体を作らない場合、ルートは親剛体
            root_rigidbody = parent_bone_rigidbody

        for base_map_idx, vertex_map in vertex_maps.items():
            logger.info("--【No.%s】剛体生成", base_map_idx + 1)

            registered_bones = all_registered_bones[base_map_idx]

            # 厚みの判定

            # 縦段INDEX
            v_yidxs = list(range(registered_bones.shape[0]))
            rigidbody_masses = np.linspace(
                param_rigidbody.param.mass / max(1, (coefficient / 2)),
                param_rigidbody.param.mass / coefficient,
                len(v_yidxs),
            )
            rigidbody_masses[0] = param_rigidbody.param.mass

            # 厚みは比較キーの数分だけ作る
            rigidbody_limit_thicks = np.linspace(
                param_option["rigidbody_root_thicks"], param_option["rigidbody_end_thicks"], len(v_yidxs)
            )

            linear_dampings = np.linspace(
                param_rigidbody.param.linear_damping,
                min(0.999, param_rigidbody.param.linear_damping * coefficient),
                len(v_yidxs),
            )
            angular_dampings = np.linspace(
                param_rigidbody.param.angular_damping,
                min(0.999, param_rigidbody.param.angular_damping * coefficient),
                len(v_yidxs),
            )
            # rigidbody_size_ratios = np.linspace(1, 0.7, len(v_yidxs))

            for v_yidx in range(vertex_map.shape[0] - 1):
                for v_xidx in range(vertex_map.shape[1]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any() or not registered_bones[v_yidx, v_xidx]:
                        continue

                    rigidbody_bone_key = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices.get(rigidbody_bone_key, None)

                    if not vv:
                        logger.warning(
                            "剛体生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                            rigidbody_bone_key,
                        )
                        continue

                    (
                        prev_map_idx,
                        prev_xidx,
                        prev_connected,
                        next_map_idx,
                        next_xidx,
                        next_connected,
                        above_yidx,
                        below_yidx,
                        target_v_yidx,
                        target_v_xidx,
                        registered_max_v_yidx,
                        registered_max_v_xidx,
                        max_v_yidx,
                        max_v_xidx,
                    ) = self.get_block_vidxs(
                        v_yidx,
                        v_xidx,
                        vertex_maps,
                        all_registered_bones,
                        all_bone_connected,
                        base_map_idx,
                        is_center=is_center,
                    )

                    below_yidx = below_yidx if below_yidx > v_yidx else max_v_yidx
                    # now_below_connected = all_bone_connected[base_map_idx][
                    #     below_yidx,
                    #     v_xidx,
                    # ]
                    prev_below_connected = all_bone_connected[prev_map_idx][
                        min(below_yidx, all_bone_connected[prev_map_idx].shape[0] - 1),
                        min(prev_xidx, all_bone_connected[prev_map_idx].shape[1] - 1),
                    ]
                    next_below_connected = all_bone_connected[next_map_idx][
                        min(below_yidx, all_bone_connected[next_map_idx].shape[0] - 1),
                        min(next_xidx, all_bone_connected[next_map_idx].shape[1] - 1),
                    ]

                    prev_above_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and above_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and above_yidx < v_yidx
                        and not np.isnan(vertex_maps[prev_map_idx][above_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    prev_now_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and v_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and not np.isnan(vertex_maps[prev_map_idx][v_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    prev_below_vv = (
                        virtual_vertices[tuple(vertex_maps[prev_map_idx][below_yidx, prev_xidx])]
                        if prev_map_idx in vertex_maps
                        and below_yidx < vertex_maps[prev_map_idx].shape[0]
                        and prev_xidx < vertex_maps[prev_map_idx].shape[1]
                        and below_yidx > v_yidx
                        and not np.isnan(vertex_maps[prev_map_idx][below_yidx, prev_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_above_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][above_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and above_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and above_yidx < v_yidx
                        and not np.isnan(vertex_maps[base_map_idx][above_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_now_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][v_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and v_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and not np.isnan(vertex_maps[base_map_idx][v_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    now_below_vv = (
                        virtual_vertices[tuple(vertex_maps[base_map_idx][below_yidx, v_xidx])]
                        if base_map_idx in vertex_maps
                        and below_yidx < vertex_maps[base_map_idx].shape[0]
                        and v_xidx < vertex_maps[base_map_idx].shape[1]
                        and below_yidx > v_yidx
                        and not np.isnan(vertex_maps[base_map_idx][below_yidx, v_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_above_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][above_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and above_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and above_yidx < v_yidx
                        and not np.isnan(vertex_maps[next_map_idx][above_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_now_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][v_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and v_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and not np.isnan(vertex_maps[next_map_idx][v_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    next_below_vv = (
                        virtual_vertices[tuple(vertex_maps[next_map_idx][below_yidx, next_xidx])]
                        if next_map_idx in vertex_maps
                        and below_yidx < vertex_maps[next_map_idx].shape[0]
                        and next_xidx < vertex_maps[next_map_idx].shape[1]
                        and below_yidx > v_yidx
                        and not np.isnan(vertex_maps[next_map_idx][below_yidx, next_xidx]).all()
                        else VirtualVertex("")
                    )

                    if not (now_now_vv and now_below_vv):
                        logger.warning(
                            "剛体生成に必要な情報(仮想頂点)が取得できなかった為、スルーします。 処理対象: %s",
                            vv.map_bones[base_map_idx].name if vv.map_bones.get(base_map_idx, None) else vv.vidxs(),
                        )
                        continue

                    prev_above_bone = prev_above_vv.map_bones.get(prev_map_idx, None)
                    prev_now_bone = prev_now_vv.map_bones.get(prev_map_idx, None)
                    prev_below_bone = prev_below_vv.map_bones.get(prev_map_idx, None)

                    now_above_bone = now_above_vv.map_bones.get(base_map_idx, None)
                    now_now_bone = now_now_vv.map_bones.get(base_map_idx, None)
                    now_below_bone = now_below_vv.map_bones.get(base_map_idx, None)

                    next_above_bone = next_above_vv.map_bones.get(next_map_idx, None)
                    next_now_bone = next_now_vv.map_bones.get(next_map_idx, None)
                    next_below_bone = next_below_vv.map_bones.get(next_map_idx, None)

                    last_v_xidx = np.max(np.sort(np.where(registered_bones[v_yidx, :])[0])[-2:])

                    if not (now_now_bone and now_below_bone):
                        if now_now_bone and now_now_bone.getVisibleFlag():
                            logger.warning(
                                "剛体生成に必要な情報(ボーン)が取得できなかった為、スルーします。 処理対象: %s",
                                vv.map_bones[base_map_idx].name
                                if vv.map_bones.get(base_map_idx, None)
                                else vv.vidxs(),
                            )
                        continue

                    if not (now_now_bone.index in model.vertices):
                        # ウェイトを持ってないボーンは登録対象外（有り得るので警告なし）
                        continue

                    x_ratio = 0.5
                    if is_center:
                        # 中央は全部を覆う
                        x_sizes = []
                        if prev_now_vv and next_now_vv:
                            x_sizes.append(next_now_vv.position().distanceToPoint(prev_now_vv.position()))
                        elif prev_now_vv and not next_now_vv:
                            x_sizes.append(now_now_vv.position().distanceToPoint(prev_now_vv.position()))
                        elif not prev_now_vv and next_now_vv:
                            x_sizes.append(next_now_vv.position().distanceToPoint(now_now_vv.position()))
                        x_size = np.max(x_sizes) if x_sizes else 0.2
                    else:
                        # それ以外はとりあえず隣の仮想頂点から横幅を図る
                        if next_now_vv.position() != MVector3D() and (
                            v_xidx < last_v_xidx or (v_xidx == last_v_xidx and next_connected)
                        ):
                            x_sizes = [now_now_bone.position.distanceToPoint(next_now_vv.position())]
                            x_sizes.append(now_below_vv.position().distanceToPoint(next_below_vv.position()))
                            x_size = np.mean(x_sizes)
                        elif (
                            (v_xidx < last_v_xidx or (v_xidx == last_v_xidx and next_connected))
                            and now_below_vv.position() != MVector3D()
                            and next_below_vv.position() != MVector3D()
                        ):
                            x_size = now_below_vv.position().distanceToPoint(next_below_vv.position())
                        elif (
                            v_xidx > 0 or (v_xidx == 0 and prev_connected)
                        ) and prev_now_vv.position() != MVector3D():
                            x_sizes = [now_now_bone.position.distanceToPoint(prev_now_vv.position())]
                            if (
                                now_below_vv.position() != MVector3D()
                                and prev_below_vv.position() != MVector3D()
                                and (v_xidx > 0 or (v_xidx == 0 and prev_connected))
                            ):
                                x_sizes.append(now_below_vv.position().distanceToPoint(prev_below_vv.position()))
                            x_size = np.mean(x_sizes)
                        elif (
                            (v_xidx > 0 or (v_xidx == 0 and prev_below_connected))
                            and now_below_vv.position() != MVector3D()
                            and prev_below_vv.position() != MVector3D()
                        ):
                            x_size = now_below_vv.position().distanceToPoint(prev_below_vv.position())
                        else:
                            v_poses = [
                                (v.position * base_reverse_axis).data() for v in model.vertices[now_now_bone.index]
                            ]

                            # # 次と繋がっている場合、次のボーンの頂点を追加する
                            # if next_connected and next_now_bone and next_now_bone.index in model.vertices:
                            #     v_poses += [(v.position * base_reverse_axis).data() for v in model.vertices[next_now_bone.index]]

                            # 重複は除外
                            v_poses = np.unique(v_poses, axis=0)

                            # 剛体の幅
                            all_vertex_diffs = np.expand_dims(v_poses, axis=1) - np.expand_dims(v_poses, axis=0)
                            all_vertex_distances = np.sqrt(np.sum(all_vertex_diffs**2, axis=-1))
                            x_size = np.median(all_vertex_distances)

                    if now_below_vv.position() != MVector3D():
                        # 基本は下のボーンとの距離
                        y_size = now_now_bone.position.distanceToPoint(now_below_vv.position())
                    else:
                        # 上と繋がっている場合
                        y_size = now_now_bone.position.distanceToPoint(now_above_vv.position())

                    if rigidbody_shape_type == 0:
                        # 球剛体の場合
                        ball_size = np.mean([x_size, y_size]) * rigidbody_limit_thicks[v_yidx]
                        shape_size = MVector3D(ball_size, ball_size, ball_size)
                        shape_volume = 4 / 3 * math.pi * shape_size.x()

                    elif rigidbody_shape_type == 1:
                        # 箱剛体の場合
                        shape_size = MVector3D(
                            max(0.5, x_size) * x_ratio,
                            max(0.5, y_size) * 0.5,
                            rigidbody_limit_thicks[v_yidx],
                        )
                        shape_volume = shape_size.x() * shape_size.y() * shape_size.z()
                    else:
                        # カプセル剛体の場合
                        shape_size = MVector3D(
                            x_size * rigidbody_limit_thicks[v_yidx],
                            max(0.25, y_size) * 0.8,
                            rigidbody_limit_thicks[v_yidx],
                        )
                        shape_volume = (shape_size.x() * shape_size.x() * math.pi * shape_size.y()) + (
                            4 / 3 * math.pi * shape_size.x()
                        )

                    logger.test(
                        "name: %s, size: %s, volume: %s",
                        vv.map_bones[base_map_idx].name,
                        shape_size.to_log(),
                        shape_volume,
                    )
                    if 0 < shape_volume < 0.002:
                        logger.warning(
                            "剛体体積が小さいため、物理が溶ける可能性があります 剛体名: %s, 剛体体積: %s",
                            vv.map_bones[base_map_idx].name,
                            round(shape_volume, 5),
                        )

                    if param_option["joint_pos_type"] == logger.transtext("ボーン間"):
                        # ジョイントがボーン間にある場合、剛体はボーン位置
                        if is_material_horizonal:
                            # 平行である場合、下ボーンとの中間
                            shape_position = (now_now_bone.position + now_below_bone.position) / 2
                        else:
                            # 平行ではない場合、後で調整し直すので一旦ボーン位置
                            shape_position = now_now_bone.position
                    else:
                        # ボーン位置は複数のボーンの中央に剛体を配置する
                        shape_positions = []
                        shape_positions.append(now_now_bone.position)

                        if now_above_bone and now_now_bone == now_below_bone:
                            # 末端の場合、上のボーンとの位置を測る
                            if now_above_vv.position() != MVector3D():
                                shape_positions.append(now_above_vv.position())

                            if next_connected:
                                if next_now_vv.position() != MVector3D():
                                    shape_positions.append(next_now_vv.position())

                                if next_above_vv.position() != MVector3D():
                                    shape_positions.append(next_above_vv.position())
                        else:
                            # それ以外の場合、下のボーンとの位置を測る
                            if now_below_vv.position() != MVector3D():
                                shape_positions.append(now_below_vv.position())

                            if next_connected:
                                if next_now_vv.position() != MVector3D():
                                    shape_positions.append(next_now_vv.position())

                                if next_below_vv.position() != MVector3D():
                                    shape_positions.append(next_below_vv.position())

                            elif prev_connected:
                                # 次と繋がってない場合、前と繋げる
                                if prev_now_vv.position() != MVector3D():
                                    shape_positions.append(prev_now_vv.position())

                                if prev_below_vv.position() != MVector3D():
                                    shape_positions.append(prev_below_vv.position())

                        shape_position = MVector3D(np.mean(shape_positions, axis=0))

                    is_y_direction_prev = False
                    y_direction_from_pos = now_below_bone.position

                    if param_option["joint_pos_type"] == logger.transtext("ボーン位置") and next_connected:
                        # ボーン位置の場合、次ボーンとの間を進行方向とする
                        x_direction_from_pos = (now_below_vv.position() + next_below_vv.position()) / 2
                        x_direction_to_pos = (now_now_bone.position + next_now_vv.position()) / 2
                        # ボーン進行方向に対しての横軸(y)
                        y_direction_to_pos = next_below_vv.position()
                        y_direction_pos = (y_direction_to_pos - y_direction_from_pos).normalized()
                    elif param_option["joint_pos_type"] == logger.transtext("ボーン位置") and prev_connected:
                        # 次と繋がってない場合、前と繋げる
                        x_direction_from_pos = (now_below_vv.position() + prev_below_vv.position()) / 2
                        x_direction_to_pos = (now_now_bone.position + prev_now_vv.position()) / 2
                        # ボーン進行方向に対しての横軸(y)
                        y_direction_to_pos = prev_below_vv.position()
                        y_direction_pos = (y_direction_to_pos - y_direction_from_pos).normalized()
                        is_y_direction_prev = True
                    else:
                        x_direction_from_pos = now_below_vv.position()
                        x_direction_to_pos = now_now_bone.position
                        # ボーン進行方向(x)
                        x_direction_pos = (x_direction_to_pos - x_direction_from_pos).normalized()
                        # ボーン進行方向に対しての横軸(y)
                        # 外積から仮の横軸を求める
                        y_direction_pos = MVector3D.crossProduct(x_direction_pos, base_vertical_axis).normalized()

                    # ボーン進行方向(x)
                    x_direction_pos = (x_direction_to_pos - x_direction_from_pos).normalized()

                    # ボーン進行方向に対しての縦軸(z)
                    z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos).normalized()

                    if is_material_horizonal:
                        shape_qq = MQuaternion.rotationTo(MVector3D(0, 0, 1), x_direction_pos)
                        # shape_qq *= MQuaternion.fromEulerAngles(90, 0, 0)
                        y_euler = (
                            90 * -x_direction_pos.y()
                            if np.isclose(abs(x_direction_pos.y()), 1)
                            else shape_qq.toEulerAngles().y()
                        )
                        shape_qq = MQuaternion.fromEulerAngles(90, -y_euler, 0)
                        shape_euler = MVector3D(90, y_euler, 0)
                    else:
                        shape_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
                        if is_y_direction_prev:
                            shape_qq *= MQuaternion.fromEulerAngles(0, 180, 0)
                        shape_euler = shape_qq.toEulerAngles()
                    shape_rotation_radians = MVector3D(
                        math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z())
                    )

                    if (
                        param_option["joint_pos_type"] == logger.transtext("ボーン間")
                        and rigidbody_shape_type == 1
                        and not is_material_horizonal
                    ):
                        if v_yidx == 0:
                            # ボーン間の箱剛体の根元は剛体位置を中間に来るよう調整
                            mat = MMatrix4x4()
                            mat.setToIdentity()
                            mat.translate(shape_position)
                            mat.rotate(shape_qq)
                            shape_position = mat * MVector3D(0, -shape_size.y(), 0)
                        elif (
                            now_above_vv
                            and base_map_idx in now_above_vv.map_rigidbodies
                            and now_above_vv.map_rigidbodies[base_map_idx].name in created_rigidbodies
                        ):
                            # 中間は上の剛体の端から計算しなおす
                            mat = MMatrix4x4()
                            mat.setToIdentity()
                            mat.translate(now_above_vv.map_rigidbodies[base_map_idx].shape_position)
                            mat.rotate(now_above_vv.map_rigidbodies[base_map_idx].shape_qq)
                            mat.translate(MVector3D(0, -now_above_vv.map_rigidbodies[base_map_idx].shape_size.y(), 0))
                            shape_position = mat * MVector3D(0, -shape_size.y(), 0)
                        elif (
                            now_above_vv
                            and base_map_idx - 1 in now_above_vv.map_rigidbodies
                            and now_above_vv.map_rigidbodies[base_map_idx - 1].name in created_rigidbodies
                        ):
                            # 中間は上の剛体の端から計算しなおす
                            mat = MMatrix4x4()
                            mat.setToIdentity()
                            mat.translate(now_above_vv.map_rigidbodies[base_map_idx - 1].shape_position)
                            mat.rotate(now_above_vv.map_rigidbodies[base_map_idx - 1].shape_qq)
                            mat.translate(
                                MVector3D(0, -now_above_vv.map_rigidbodies[base_map_idx - 1].shape_size.y(), 0)
                            )
                            shape_position = mat * MVector3D(0, -shape_size.y(), 0)

                    # 全て物理剛体
                    mode = 1

                    # 前で繋がってないX
                    prev_not_connected_x_idxs = np.where(
                        np.diff(all_bone_connected[base_map_idx][v_yidx, :v_xidx]) == -1
                    )[0]
                    prev_not_connected_x_idx = (
                        np.max(prev_not_connected_x_idxs + 1) if prev_not_connected_x_idxs.any() else 0
                    )
                    # 自分を含む後で繋がってないX
                    next_not_connected_x_idxs = np.where(
                        np.diff(all_bone_connected[base_map_idx][v_yidx, (prev_not_connected_x_idx + 1) :]) == -1
                    )[0]
                    next_not_connected_x_idx = (
                        np.min(next_not_connected_x_idxs + (prev_not_connected_x_idx + 1) + 1)
                        if next_not_connected_x_idxs.any()
                        else 0
                    )
                    next_not_connected_prev_x_idx = (
                        np.max(np.where(registered_bones[v_yidx, :next_not_connected_x_idx])[0])
                        if next_not_connected_x_idx
                        else 0
                    )

                    # 実際に剛体を作ってる最後のY
                    actual_v_yidx = np.min(np.sort(np.where(registered_bones[:, v_xidx])[0])[-2:])

                    (
                        last_prev_map_idx,
                        last_prev_xidx,
                        last_prev_connected,
                        last_next_map_idx,
                        last_next_xidx,
                        last_next_connected,
                        last_above_yidx,
                        last_below_yidx,
                        last_target_v_yidx,
                        last_target_v_xidx,
                        last_registered_max_v_yidx,
                        last_registered_max_v_xidx,
                        last_max_v_yidx,
                        last_max_v_xidx,
                    ) = self.get_block_vidxs(
                        v_yidx,
                        last_v_xidx,
                        vertex_maps,
                        all_registered_bones,
                        all_bone_connected,
                        base_map_idx,
                        is_center=is_center,
                    )

                    is_folding = v_xidx in [next_not_connected_prev_x_idx, next_not_connected_x_idx,] and param_option[
                        "joint_pos_type"
                    ] == logger.transtext("ボーン位置")
                    logger.debug(
                        "is_folding: base_map_idx[%s], v_xidx[%s], v_yidx[%s], is_folding[%s], next_not_connected_prev_x_idx[%s], next_not_connected_x_idx[%s], last_next_connected[%s]",
                        base_map_idx,
                        v_xidx,
                        v_yidx,
                        is_folding,
                        next_not_connected_prev_x_idx,
                        next_not_connected_x_idx,
                        last_next_connected,
                    )

                    # 横端が自分の先が繋がって無い場合、折り返しは質量半分
                    mass_ratio = 0.5 if next_not_connected_x_idx and is_folding else 1
                    mass = rigidbody_masses[v_yidx] * mass_ratio

                    if is_folding and next_not_connected_x_idx and v_xidx == next_not_connected_x_idx:
                        # 折り返しの場合、ひとつ前と揃える
                        next_not_connected_prev_x_key = tuple(vertex_map[v_yidx, next_not_connected_prev_x_idx])
                        prev_rigidbody = virtual_vertices[next_not_connected_prev_x_key].map_rigidbodies.get(
                            base_map_idx, None
                        )
                        if prev_rigidbody:
                            shape_size = prev_rigidbody.shape_size
                            shape_position = prev_rigidbody.shape_position
                            shape_rotation_radians = prev_rigidbody.shape_rotation
                            mass = prev_rigidbody.param.mass

                        logger.debug(
                            "**folding copy: base_map_idx[%s], v_xidx[%s], v_yidx[%s], is_folding[%s]",
                            base_map_idx,
                            v_xidx,
                            v_yidx,
                            is_folding,
                        )

                    vv.map_rigidbodies[base_map_idx] = RigidBody(
                        vv.map_bones[base_map_idx].name,
                        vv.map_bones[base_map_idx].name,
                        vv.map_bones[base_map_idx].index,
                        param_rigidbody.collision_group,
                        (param_rigidbody.no_collision_group if v_yidx < actual_v_yidx else all_no_collision_group),
                        rigidbody_shape_type,
                        shape_size,
                        shape_position,
                        shape_rotation_radians,
                        mass,
                        linear_dampings[v_yidx],
                        angular_dampings[v_yidx],
                        param_rigidbody.param.restitution,
                        param_rigidbody.param.friction,
                        mode,
                    )
                    vv.map_rigidbodies[base_map_idx].shape_qq = shape_qq
                    vv.map_rigidbodies[base_map_idx].x_direction = x_direction_pos
                    vv.map_rigidbodies[base_map_idx].y_direction = y_direction_pos
                    vv.map_rigidbodies[base_map_idx].z_direction = z_direction_pos

                    # 別途保持しておく
                    if vv.map_rigidbodies[base_map_idx].name not in created_rigidbodies:
                        if base_map_idx not in created_rigidbody_vvs:
                            created_rigidbody_vvs[base_map_idx] = {}
                        if v_xidx not in created_rigidbody_vvs[base_map_idx]:
                            created_rigidbody_vvs[base_map_idx][v_xidx] = {}
                        created_rigidbody_vvs[base_map_idx][v_xidx][v_yidx] = vv
                        created_rigidbodies[vv.map_rigidbodies[base_map_idx].name] = vv.map_rigidbodies[base_map_idx]
                    else:
                        # 既に保持済みの剛体である場合、前のを参照する
                        vv.map_rigidbodies[base_map_idx] = created_rigidbodies[vv.map_rigidbodies[base_map_idx].name]

                    if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 50 > prev_rigidbody_cnt:
                        logger.info("-- -- 【No.%s】剛体: %s個目:終了", base_map_idx + 1, len(created_rigidbodies))
                        prev_rigidbody_cnt = len(created_rigidbodies) // 50

        for rigidbody_name in sorted(created_rigidbodies.keys()):
            # 剛体を登録
            rigidbody = created_rigidbodies[rigidbody_name]
            rigidbody.index = len(model.rigidbodies)

            if rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                rigidbody.name += randomname(3)

            model.rigidbodies[rigidbody.name] = rigidbody
            logger.test(f"rigidbody: {rigidbody}")

        logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

        if param_option["rigidbody_balancer"]:
            # バランサー剛体が必要な場合

            prev_rigidbody_cnt = 0
            # すべて非衝突対象
            balancer_no_collision_group = 0
            # 剛体生成
            created_rigidbodies = {}
            # ボーン生成
            created_bones = {}

            for base_map_idx in sorted(created_rigidbody_vvs.keys()):
                logger.info("--【No.%s】バランサー剛体生成", base_map_idx + 1)

                for v_xidx in sorted(created_rigidbody_vvs[base_map_idx].keys()):

                    rigidbody_volume = MVector3D(1, 1, 1)
                    rigidbody_mass = 0
                    for v_yidx in reversed(created_rigidbody_vvs[base_map_idx][v_xidx].keys()):
                        vv = created_rigidbody_vvs[base_map_idx][v_xidx][v_yidx]
                        # 元の剛体
                        org_rigidbody = vv.map_rigidbodies[base_map_idx]
                        org_bone = vv.map_bones[base_map_idx]
                        org_tail_position = org_bone.tail_position + org_bone.position
                        if org_bone.tail_index >= 0:
                            org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
                        # ボーンの向き
                        org_axis = (org_tail_position - org_bone.position).normalized()

                        if rigidbody_mass > 0:
                            # 元剛体の重量は子の1.5倍
                            org_rigidbody.param.mass = rigidbody_mass * 1.5
                            # バランサー剛体のサイズ
                            shape_size = MVector3D(
                                0.5,
                                (org_rigidbody.shape_size.y() + rigidbody_volume.y()) * 2,
                                org_rigidbody.shape_size.z(),
                            )
                        else:
                            # バランサー剛体のサイズ
                            shape_size = MVector3D(0.5, org_rigidbody.shape_size.y(), org_rigidbody.shape_size.z())

                        # 名前にバランサー追加
                        rigidbody_name = f"B-{org_rigidbody.name}"

                        # バランサー剛体の回転
                        if org_axis.y() < 0:
                            # 下を向いてたらY方向に反転
                            shape_qq = MQuaternion.fromEulerAngles(0, 180, 0)
                            shape_qq *= org_rigidbody.shape_qq.copy()
                        else:
                            # 上を向いてたらX方向に反転
                            shape_qq = org_rigidbody.shape_qq.copy()
                            shape_qq *= MQuaternion.fromEulerAngles(180, 0, 0)
                        shape_qq *= MQuaternion.fromEulerAngles(0, 180, 0)

                        shape_euler = shape_qq.toEulerAngles()
                        shape_rotation_radians = MVector3D(
                            math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z())
                        )

                        # バランサー剛体の位置はバランサー剛体の上端から反対向き
                        mat = MMatrix4x4()
                        mat.setToIdentity()
                        mat.translate(org_rigidbody.shape_position)
                        mat.rotate(org_rigidbody.shape_qq)
                        mat.translate(MVector3D(0, org_rigidbody.shape_size.y() / 2, 0))
                        mat.rotate(org_rigidbody.shape_qq.inverted())
                        mat.rotate(shape_qq)

                        # バランサー剛体の位置
                        shape_position = mat * MVector3D(0, -shape_size.y() / 2, 0)

                        # バランサー剛体用のボーン
                        balancer_bone = Bone(
                            rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002
                        )
                        created_bones[balancer_bone.name] = balancer_bone

                        vv.map_balance_rigidbodies[base_map_idx] = RigidBody(
                            rigidbody_name,
                            rigidbody_name,
                            -1,
                            org_rigidbody.collision_group,
                            balancer_no_collision_group,
                            2,
                            shape_size,
                            shape_position,
                            shape_rotation_radians,
                            org_rigidbody.param.mass,
                            org_rigidbody.param.linear_damping,
                            org_rigidbody.param.angular_damping,
                            org_rigidbody.param.restitution,
                            org_rigidbody.param.friction,
                            1,
                        )
                        vv.map_balance_rigidbodies[base_map_idx].shape_qq = shape_qq

                        # 別途保持しておく
                        if vv.map_balance_rigidbodies[base_map_idx].name not in created_rigidbodies:
                            created_rigidbodies[
                                vv.map_balance_rigidbodies[base_map_idx].name
                            ] = vv.map_balance_rigidbodies[base_map_idx]
                        else:
                            # 既に保持済みの剛体である場合、前のを参照する
                            vv.map_balance_rigidbodies[base_map_idx] = created_rigidbodies[
                                vv.map_balance_rigidbodies[base_map_idx].name
                            ]

                        if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 50 > prev_rigidbody_cnt:
                            logger.info("-- -- 【No.%s】バランサー剛体: %s個目:終了", base_map_idx + 1, len(created_rigidbodies))
                            prev_rigidbody_cnt = len(created_rigidbodies) // 50

                        # 子剛体のサイズを保持
                        rigidbody_volume += org_rigidbody.shape_size
                        # 質量は子の1.5倍
                        rigidbody_mass = org_rigidbody.param.mass

            for rigidbody_name in sorted(created_rigidbodies.keys()):
                # ボーンを登録
                bone = created_bones[rigidbody_name]
                bone.index = len(model.bones)
                model.bones[bone.name] = bone

                # 剛体を登録
                rigidbody = created_rigidbodies[rigidbody_name]
                rigidbody.index = len(model.rigidbodies)
                rigidbody.bone_index = bone.index

                if rigidbody.name in model.rigidbodies:
                    logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                    rigidbody.name += randomname(3)
                    bone.name = copy.deepcopy(rigidbody.name)

                model.rigidbodies[rigidbody.name] = rigidbody
                logger.test(f"rigidbody: {rigidbody}")

            logger.info("-- バランサー剛体: %s個目:終了", len(created_rigidbodies))

        return root_rigidbody, left_root_rigidbody, right_root_rigidbody, parent_bone_rigidbody

    def create_grad_weight(
        self,
        model: PmxModel,
        param_option: dict,
        target_vertices: list,
        virtual_vertices: dict,
        remaining_vertices: dict,
        threshold: float,
        base_vertical_axis: MVector3D,
        base_reverse_axis: MVector3D,
        horizonal_top_edge_keys: list,
        grad_vertices: list,
        root_bone: Bone,
        left_root_bone: Bone,
        right_root_bone: Bone,
    ):
        remaining_vidxs = [vidx for vv in remaining_vertices.values() for vidx in vv.vidxs()]
        # 足に繋げるか否か
        is_rigidbody_leg = param_option["rigidbody_leg"]

        # グラデ頂点＋残頂点がないもしくは根元頂点が指定されていない場合、スルー
        if not (grad_vertices + remaining_vidxs) or not param_option["top_vertices_csv"]:
            return remaining_vertices

        logger.info(
            "【%s:%s】グラデーションウェイト生成",
            param_option["material_name"],
            param_option["abb_name"],
            decoration=MLogger.DECORATION_LINE,
        )

        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]

        # 塗り終わった仮想頂点キーリスト
        weighted_vkeys = list(set(virtual_vertices.keys()) - set(remaining_vertices.keys()))
        axis_idx = np.where(np.abs(base_vertical_axis.data()))[0][0]

        # 塗り終わった頂点のリスト
        weighted_vertices = {}
        for vkey in weighted_vkeys:
            for vidx in virtual_vertices[vkey].vidxs():
                v = model.vertex_dict[vidx]
                weighted_vertices[v.index] = v.position.data()

        weight_cnt = 0
        prev_weight_cnt = 0

        # 親ボーンの評価軸位置
        parent_axis_val = parent_bone.position.data()[axis_idx]

        # 下半身評価軸位置
        grad_trunk_bone_name = parent_bone.name
        grad_other_bone_name = parent_bone.name
        grad_other2_bone_name = parent_bone.name
        grad_tail_bone_name = parent_bone.name
        grad_leg_indexes = [-1, -1]
        grad_leg_poses = [np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float)]
        if is_rigidbody_leg:
            grad_trunk_bone_name = "下半身"
            grad_other_bone_name = "上半身"
            grad_other2_bone_name = "上半身2" if "上半身2" in model.bones else None
            grad_tail_bone_name = "上半身"
            grad_leg_indexes = [model.bones["左足"].index, model.bones["右足"].index]
            grad_leg_poses = [model.bones["左足"].position.data(), model.bones["右足"].position.data()]

        # グラデ末端位置
        if model.bones[grad_tail_bone_name].tail_index >= 0:
            grad_tail_vec = model.bones[model.bone_indexes[model.bones[grad_tail_bone_name].tail_index]].position
            grad_other_vec = (model.bones[grad_tail_bone_name].position + grad_tail_vec) / 2
        else:
            grad_tail_vec = model.bones[grad_tail_bone_name].position + model.bones[grad_tail_bone_name].tail_position
            grad_other_vec = model.bones[grad_tail_bone_name].position + (
                model.bones[grad_tail_bone_name].tail_position / 2
            )
        tail_axis_val = grad_tail_vec.data()[axis_idx]

        for vidx in grad_vertices:
            v = model.vertex_dict[vidx]

            # 仮想頂点登録（該当頂点対象）
            vkey = v.position.to_key(threshold)
            if vkey not in virtual_vertices:
                virtual_vertices[vkey] = VirtualVertex(vkey)
            virtual_vertices[vkey].append([v], [], [])

            if vkey not in remaining_vertices:
                remaining_vertices[vkey] = VirtualVertex(vkey)
            remaining_vertices[vkey].append([v], [], [])

            logger.test("grad_vv: vidxs[%s]", remaining_vertices[vkey].vidxs())

        weighted_grad_vkeys = []
        for v_key, vv in remaining_vertices.items():
            if not vv.vidxs():
                continue

            # 仮想頂点の評価軸位置
            vv_axis_val = vv.position().data()[axis_idx]

            # # 頂点マップ上部との距離
            # top_distances = np.linalg.norm((top_vv_poses - vv.position().data()), ord=2, axis=1)
            # nearest_top_vidxs = virtual_vertices[horizonal_top_edge_keys[np.argmin(top_distances)]].vidxs()
            # nearest_top_pos = top_vv_poses[np.argmin(top_distances)]
            # # 頂点マップ上部直近頂点の評価軸位置
            # nearest_top_axis_val = nearest_top_pos[np.where(np.abs(base_vertical_axis.data()))][0]

            # if nearest_top_axis_val > vv_axis_val + 0.1 or parent_axis_val < vv_axis_val:
            #     # 評価軸にTOPより遠い（スカートの場合、TOPより下）の場合、スルー
            #     logger.debug(
            #         f"×グラデスルー: target [{vv.vidxs()}], nearest_top_vidxs[{nearest_top_vidxs}], nearest_top_axis_val[{round(nearest_top_axis_val, 3)}], vv[{vv.vidxs()}], vv_axis_val[{round(vv_axis_val, 3)}]"
            #     )
            #     continue

            # logger.debug(
            #     f"○グラデターゲット: target [{vv.vidxs()}], nearest_top_vidxs[{nearest_top_vidxs}], nearest_top_axis_val[{round(nearest_top_axis_val, 3)}], vv[{vv.vidxs()}], vv_axis_val[{round(vv_axis_val, 3)}], parent_axis_val[{round(parent_axis_val, 3)}]"
            # )

            if vv_axis_val > tail_axis_val + 0.1:
                logger.debug(
                    f"×グラデスルー: vv [{vv.vidxs()}], vv_axis_val[{round(vv_axis_val, 3)}], parent_axis_val[{round(parent_axis_val, 3)}]"
                )
                continue

            # 各頂点の位置との差分から距離を測る
            vv_distances = np.linalg.norm(
                (np.array(list(weighted_vertices.values())) - vv.position().data()), ord=2, axis=1
            )

            # 直近頂点INDEXのウェイト
            copy_weighted_vertex_idx = list(weighted_vertices.keys())[np.argmin(vv_distances)]
            copy_weighted_vertex_deform = model.vertex_dict[copy_weighted_vertex_idx].deform

            # ウェイト対象となるボーンINDEXリスト
            grad_weight_bone_idxs = []
            grad_weight_axis_vals = []

            for bone_idx, weight in zip(
                copy_weighted_vertex_deform.get_idx_list(), copy_weighted_vertex_deform.get_weights()
            ):
                if weight > 0:
                    grad_weight_bone_idxs.append(bone_idx)
                    # ウェイト対象の評価軸の位置
                    grad_weight_axis_vals.append(
                        (
                            model.bones[model.bone_indexes[bone_idx]].position.data()[axis_idx]
                            - vv.position().data()[axis_idx]
                        )
                    )

            # if parent_bone.name == "下半身" and is_rigidbody_leg:
            #     # # 物理ウェイトは最も近いのひとつだけ残す
            #     # wb_idx = grad_weight_bone_idxs[np.argmin(grad_weight_axis_vals)]
            #     # wb = grad_weight_axis_vals[np.argmin(grad_weight_axis_vals)]

            #     # grad_weight_bone_idxs = [wb_idx]
            #     # grad_weight_axis_vals = [wb]

            #     # 上半身は常にウェイト対象とする
            #     grad_weight_axis_vals.append(tail_axis_val - vv_axis_val)
            #     grad_weight_bone_idxs.append(model.bones["上半身"].index)

            #     # 足
            #     leg_distances = np.linalg.norm(grad_leg_poses - vv.position().data(), ord=2, axis=1)

            #     grad_weight_axis_vals.append(
            #         leg_distances[0]
            #         * max(0.2, (grad_leg_poses[0][np.where(np.abs(base_vertical_axis.data()))][0] - vv_axis_val))
            #     )
            #     grad_weight_bone_idxs.append(grad_leg_indexes[0])

            #     grad_weight_axis_vals.append(
            #         leg_distances[1]
            #         * max(0.2, (grad_leg_poses[1][np.where(np.abs(base_vertical_axis.data()))][0] - vv_axis_val))
            #     )
            #     grad_weight_bone_idxs.append(grad_leg_indexes[1])

            # # 親ボーンとの距離
            # grad_weight_axis_vals.append((parent_axis_val - vv_axis_val) * 2)
            # grad_weight_bone_idxs.append(parent_bone.index)

            # grad_weight_bone_idxs = np.array(grad_weight_bone_idxs)[np.where(grad_weight_axis_vals)]
            # grad_target_weights = np.abs(grad_weight_axis_vals)[np.where(grad_weight_axis_vals)]
            # grad_weights = np.min(grad_target_weights) / grad_target_weights
            # grad_weights = grad_weights / grad_weights.sum(axis=0, keepdims=1)
            # grad_weights[grad_weights > 0.7] *= 0.5

            # # ウェイト上位4件まで
            # weight_idxs = np.argsort(grad_weights)[-4:]
            # weight_bone_idxs = grad_weight_bone_idxs[weight_idxs]
            # weights = grad_weights[weight_idxs]
            # # ウェイト正規化
            # weights = weights / weights.sum(axis=0, keepdims=1)
            # # 補正
            # weights[weights > 0.6] *= 0.5
            # weights = weights / weights.sum(axis=0, keepdims=1)

            if is_rigidbody_leg:
                # # 物理ウェイトは最も近いのひとつだけ残す
                # wb_idx = grad_weight_bone_idxs[np.argmin(grad_weight_axis_vals)]
                # wb = grad_weight_axis_vals[np.argmin(grad_weight_axis_vals)]

                # grad_weight_bone_idxs = [wb_idx]
                # grad_weight_axis_vals = [wb]

                # 物理ウェイトはちょっと判定を厳しくしておく
                grad_weight_axis_vals = [wb * 1.5 for wb in grad_weight_axis_vals]

                # 体幹のもう片方も常にウェイト対象とする
                grad_weight_bone_idxs.append(model.bones[grad_other_bone_name].index)
                grad_weight_axis_vals.append(grad_other_vec.data()[axis_idx] - vv.position().data()[axis_idx])

                if grad_other2_bone_name:
                    # 上半身2がある場合、そのウェイトも割り当てる
                    grad_weight_bone_idxs.append(model.bones[grad_other2_bone_name].index)
                    grad_weight_axis_vals.append(
                        model.bones[grad_other2_bone_name].position.data()[axis_idx] - vv.position().data()[axis_idx]
                    )

                # 足の距離
                leg_distances = np.linalg.norm(grad_leg_poses - vv.position().data(), ord=2, axis=1)

                # 左足
                grad_weight_bone_idxs.append(grad_leg_indexes[0])
                grad_weight_axis_vals.append(leg_distances[0] * (0.3 if vv.position().x() > 0 else 0.5))

                # 右足
                grad_weight_bone_idxs.append(grad_leg_indexes[1])
                grad_weight_axis_vals.append(leg_distances[1] * (0.3 if vv.position().x() <= 0 else 0.5))

            # 親ボーンとの距離
            grad_weight_axis_vals.append(
                model.bones[grad_trunk_bone_name].position.data()[axis_idx] - vv.position().data()[axis_idx]
            )
            grad_weight_bone_idxs.append(model.bones[grad_trunk_bone_name].index)

            if not np.where(grad_weight_axis_vals):
                continue

            grad_weight_bone_idxs = np.array(grad_weight_bone_idxs)[np.where(grad_weight_axis_vals)]
            grad_target_weights = np.abs(grad_weight_axis_vals)[np.where(grad_weight_axis_vals)]
            grad_weights = np.min(grad_target_weights) / grad_target_weights
            grad_weights = grad_weights / grad_weights.sum(axis=0, keepdims=1)

            # ウェイト上位4件まで
            weight_idxs = np.argsort(grad_weights)[-4:]
            weight_bone_idxs = grad_weight_bone_idxs[weight_idxs]
            weights = grad_weights[weight_idxs]
            # ウェイト正規化
            weights = weights / weights.sum(axis=0, keepdims=1)

            logger.debug(
                f"グラデウェイト: target [{vv.vidxs()}], weighted [{copy_weighted_vertex_idx}], grad_weight_bone_idxs[{grad_weight_bone_idxs}], grad_weights[{np.round(grad_weights, decimals=3)}], weight_idx[{weight_bone_idxs}], weight[{np.round(weights, decimals=3)}]"
            )

            if np.count_nonzero(weights) == 1:
                vv.deform = Bdef1(weight_bone_idxs[-1])
            elif np.count_nonzero(weights) == 2:
                vv.deform = Bdef2(
                    weight_bone_idxs[-1],
                    weight_bone_idxs[-2],
                    weights[-1],
                )
            elif np.count_nonzero(weights) == 3:
                vv.deform = Bdef4(
                    weight_bone_idxs[-1],
                    weight_bone_idxs[-2],
                    weight_bone_idxs[-3],
                    0,
                    weights[-1],
                    weights[-2],
                    weights[-3],
                    0,
                )
            else:
                vv.deform = Bdef4(
                    weight_bone_idxs[-1],
                    weight_bone_idxs[-2],
                    weight_bone_idxs[-3],
                    weight_bone_idxs[-4],
                    weights[-1],
                    weights[-2],
                    weights[-3],
                    weights[-4],
                )

            for rv in vv.real_vertices:
                rv.deform = vv.deform

            weighted_grad_vkeys.append(v_key)

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 200 > prev_weight_cnt:
                logger.info("-- グラデーション頂点ウェイト: %s個目:終了", weight_cnt)
                prev_weight_cnt = weight_cnt // 200

        for v_key in weighted_grad_vkeys:
            # 登録対象の場合、残対象から削除
            if v_key in remaining_vertices:
                del remaining_vertices[v_key]

        logger.info("-- グラデーション頂点ウェイト: %s個目:終了", weight_cnt)

        return remaining_vertices

    def create_back_weight(
        self,
        model: PmxModel,
        param_option: dict,
        target_vertices: list,
        grad_vertices: list,
        virtual_vertices: dict,
        back_vertices: list,
        remaining_vertices: dict,
        threshold: float,
    ):
        if param_option["back_material_name"]:
            # 表面で残った裏頂点と裏材質で指定されている頂点を全部対象とする
            back_vertices += list(model.material_vertices[param_option["back_material_name"]])
        if param_option["back_extend_material_names"]:
            for mname in param_option["back_extend_material_names"]:
                back_vertices += list(model.material_vertices[mname])

        if param_option["vertices_back_csv"]:
            try:
                for vertices_back_csv_path in glob(param_option["vertices_back_csv"]):
                    back_vertices.extend(read_vertices_from_file(vertices_back_csv_path, model, None))
            except Exception:
                logger.info(
                    "【%s:%s】裏ウェイト生成",
                    param_option["material_name"],
                    param_option["abb_name"],
                    decoration=MLogger.DECORATION_LINE,
                )

                logger.warning("裏面対象頂点CSVが正常に読み込めなかったため、処理をスキップします", decoration=MLogger.DECORATION_BOX)

        # 残っている頂点
        remaining_vidxs = dict([(vidx, rv) for rv in remaining_vertices.values() for vidx in rv.vidxs()])

        if not back_vertices:
            return list(remaining_vidxs.keys())

        # 重複を除外
        back_vertices = list(set(back_vertices))

        logger.info(
            "【%s:%s】裏ウェイト生成",
            param_option["material_name"],
            param_option["abb_name"],
            decoration=MLogger.DECORATION_LINE,
        )

        weight_cnt = 0
        prev_weight_cnt = 0

        # 処理対象材質頂点＋グラデ頂点から裏頂点を引いた頂点リストを裏ウェイトの対象とする
        front_vertices = {}
        for vidx in list(
            (
                set(list(model.material_vertices[param_option["material_name"]]))
                | set(target_vertices)
                | set(grad_vertices)
            )
            - set(back_vertices)
        ):
            v = model.vertex_dict[vidx]
            front_vertices[v.index] = v.position.data()

        back_weighted_vidxs = []
        for vertex_idx in back_vertices:
            bv = model.vertex_dict[vertex_idx]

            # 各頂点の位置との差分から距離を測る
            bv_distances = np.linalg.norm(
                (np.array(list(front_vertices.values())) - bv.position.data()), ord=2, axis=1
            )

            # 直近頂点INDEXのウェイトを転写
            copy_front_vertex_idx = list(front_vertices.keys())[np.argmin(bv_distances)]

            logger.debug(f"裏頂点: back [{bv.index}], front [{copy_front_vertex_idx}], 距離 [{np.min(bv_distances)}]")
            bv.deform = copy.deepcopy(model.vertex_dict[copy_front_vertex_idx].deform)

            back_weighted_vidxs.append(bv.index)

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 200 > prev_weight_cnt:
                logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)
                prev_weight_cnt = weight_cnt // 200

        logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)

        # 残頂点から塗り終わった裏頂点を除外する
        return list(set(list(remaining_vidxs.keys())) - set(back_weighted_vidxs))

    def create_remaining_weight(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        all_registered_bones: dict,
        remaining_vidxs: list,
        threshold: float,
    ):
        if not remaining_vidxs:
            return

        logger.info("【%s:%s】残ウェイト生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        vertex_cnt = 0
        prev_vertex_cnt = 0

        # ウェイト塗り終わった実頂点INDEXとその位置
        weighted_vidxs = dict(
            [
                (vidx, model.vertex_dict[vidx].position.data())
                for vv in virtual_vertices.values()
                for vidx in vv.vidxs()
                if vidx not in remaining_vidxs
            ]
        )

        # 登録済みのボーンの位置リスト
        bone_poses = {}
        for base_map_idx, vertex_map in vertex_maps.items():
            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        continue

                    v_key = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices[v_key]

                    bones = [b for b in vv.map_bones.values() if b]
                    if bones:
                        # ボーンが登録されている場合、ボーン位置を保持
                        bone_poses[bones[0].index] = bones[0].position.data()
                        continue

                    # まだウェイトが塗られてないのは残頂点に入れる
                    if not vv.deform:
                        remaining_vidxs.extend(vv.vidxs())

        # 裾材質を追加
        if param_option["edge_material_name"] or param_option["edge_extend_material_names"]:
            for mname in [param_option["edge_material_name"]] + param_option["edge_extend_material_names"]:
                if not mname:
                    continue
                remaining_vidxs.extend(model.material_vertices[param_option["edge_material_name"]])

        if param_option["vertices_edge_csv"]:
            try:
                for vertices_edge_csv_path in glob(param_option["vertices_edge_csv"]):
                    remaining_vidxs.extend(read_vertices_from_file(vertices_edge_csv_path, model, None))
            except Exception:
                logger.warning("裾対象頂点CSVが正常に読み込めなかったため、処理をスキップします", decoration=MLogger.DECORATION_BOX)

        # 重複を除外
        remaining_vidx_dicts = dict([(vidx, vidx) for vidx in list(set(remaining_vidxs))])

        remain_cnt = len(remaining_vidx_dicts) * 2
        while remaining_vidx_dicts and remain_cnt > 0:
            remain_cnt -= 1
            # ランダムで選ぶ
            vidx = list(remaining_vidx_dicts.keys())[random.randrange(len(remaining_vidx_dicts))]
            rv = model.vertex_dict[vidx]

            # ウェイト済み頂点のうち、最も近いのを抽出
            weighted_diff_distances = np.linalg.norm(
                np.array(list(weighted_vidxs.values())) - rv.position.data(), ord=2, axis=1
            )

            nearest_vidx = list(weighted_vidxs.keys())[np.argmin(weighted_diff_distances)]
            nearest_deform = model.vertex_dict[nearest_vidx].deform

            if type(nearest_deform) is Bdef1:
                logger.debug(
                    f"remaining1 vidx[{vidx}], nearest_vidx[{nearest_vidx}] weight_names: [{model.bone_indexes[nearest_deform.index0]}], total_weights: [1]"
                )
                rv.deform = Bdef1(nearest_deform.index0)

                del remaining_vidx_dicts[vidx]
            elif type(nearest_deform) is Bdef2:
                weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
                weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]

                bone1_distance = vv.position().distanceToPoint(weight_bone1.position)
                bone2_distance = (
                    vv.position().distanceToPoint(weight_bone2.position) if nearest_deform.weight0 < 1 else 0
                )
                weight_names = np.array([weight_bone1.name, weight_bone2.name])
                if bone1_distance + bone2_distance != 0:
                    total_weights = np.array(
                        [
                            bone1_distance / (bone1_distance + bone2_distance),
                            bone2_distance / (bone1_distance + bone2_distance),
                        ]
                    )
                else:
                    total_weights = np.array([1, 0])
                    logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", vv.vidxs())
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                if np.count_nonzero(weights) == 1:
                    rv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)

                    logger.debug(
                        f"remaining2->1 vidx[{vidx}], nearest_vidx[{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )
                elif np.count_nonzero(weights) == 2:
                    rv.deform = Bdef2(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        weights[weight_idxs[-1]],
                    )

                    logger.debug(
                        f"remaining2->2 vidx[{vidx}], nearest_vidx[{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )
                else:
                    rv.deform = Bdef4(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        model.bones[weight_names[weight_idxs[-3]]].index,
                        model.bones[weight_names[weight_idxs[-4]]].index,
                        weights[weight_idxs[-1]],
                        weights[weight_idxs[-2]],
                        weights[weight_idxs[-3]],
                        weights[weight_idxs[-4]],
                    )

                    logger.debug(
                        f"remaining2->4 vidx[{vidx}], nearest_vidx[{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )

                del remaining_vidx_dicts[vidx]
            elif type(nearest_deform) is Bdef4:
                weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
                weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]
                weight_bone3 = model.bones[model.bone_indexes[nearest_deform.index2]]
                weight_bone4 = model.bones[model.bone_indexes[nearest_deform.index3]]

                bone1_distance = (
                    vv.position().distanceToPoint(weight_bone1.position) if nearest_deform.weight0 > 0 else 0
                )
                bone2_distance = (
                    vv.position().distanceToPoint(weight_bone2.position) if nearest_deform.weight1 > 0 else 0
                )
                bone3_distance = (
                    vv.position().distanceToPoint(weight_bone3.position) if nearest_deform.weight2 > 0 else 0
                )
                bone4_distance = (
                    vv.position().distanceToPoint(weight_bone4.position) if nearest_deform.weight3 > 0 else 0
                )
                all_distance = bone1_distance + bone2_distance + bone3_distance + bone4_distance

                weight_names = np.array([weight_bone1.name, weight_bone2.name, weight_bone3.name, weight_bone4.name])
                if all_distance != 0:
                    total_weights = np.array(
                        [
                            bone1_distance / all_distance,
                            bone2_distance / all_distance,
                            bone3_distance / all_distance,
                            bone4_distance / all_distance,
                        ]
                    )
                else:
                    total_weights = np.array([1, bone2_distance, bone3_distance, bone4_distance])
                    logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", vv.vidxs())
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                if np.count_nonzero(weights) == 1:
                    rv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)

                    logger.debug(
                        f"remaining4->1 vidx[{vidx}], nearest_vidx: [{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )
                elif np.count_nonzero(weights) == 2:
                    rv.deform = Bdef2(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        weights[weight_idxs[-1]],
                    )

                    logger.debug(
                        f"remaining4->2 vidx[{vidx}], nearest_vidx: [{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )
                else:
                    rv.deform = Bdef4(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        model.bones[weight_names[weight_idxs[-3]]].index,
                        model.bones[weight_names[weight_idxs[-4]]].index,
                        weights[weight_idxs[-1]],
                        weights[weight_idxs[-2]],
                        weights[weight_idxs[-3]],
                        weights[weight_idxs[-4]],
                    )

                    logger.debug(
                        f"remaining4->4 vidx[{vidx}], nearest_vidx: [{nearest_vidx}], weight_names: [{weight_names}], total_weights: [{total_weights}]"
                    )

                del remaining_vidx_dicts[vidx]

            vertex_cnt += 1

            if vertex_cnt > 0 and vertex_cnt // 100 > prev_vertex_cnt:
                logger.info("-- 残ウェイト: %s個目:終了", vertex_cnt)
                prev_vertex_cnt = vertex_cnt // 100

        logger.info("-- 残ウェイト: %s個目:終了", vertex_cnt)

    def create_weight(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: list,
        all_registered_bones: dict,
        all_bone_vertical_distances: dict,
        all_bone_horizonal_distances: dict,
        all_bone_connected: dict,
        remaining_vertices: dict,
        threshold: float,
        root_bone: Bone,
        left_root_bone: Bone,
        right_root_bone: Bone,
    ):
        logger.info("【%s:%s】ウェイト生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        for base_map_idx, vertex_map in vertex_maps.items():
            logger.info("--【No.%s】ウェイト分布判定", base_map_idx + 1)

            if base_map_idx not in all_registered_bones:
                continue

            registered_bones = all_registered_bones[base_map_idx]

            # ウェイト分布
            prev_weight_cnt = 0
            weight_cnt = 0

            for v_xidx in range(vertex_map.shape[1]):
                for v_yidx in range(vertex_map.shape[0]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        continue

                    vkey = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices[vkey]

                    if vv.deform:
                        continue

                    (
                        prev_map_idx,
                        prev_xidx,
                        prev_connected,
                        next_map_idx,
                        next_xidx,
                        next_connected,
                        above_yidx,
                        below_yidx,
                        target_v_yidx,
                        target_v_xidx,
                        registered_max_v_yidx,
                        registered_max_v_xidx,
                        max_v_yidx,
                        max_v_xidx,
                    ) = self.get_block_vidxs(
                        v_yidx,
                        v_xidx,
                        vertex_maps,
                        all_registered_bones,
                        all_bone_connected,
                        base_map_idx,
                        is_weight=True,
                    )

                    if registered_bones[v_yidx, v_xidx]:
                        # 同じ仮想頂点上に登録されているボーンが複数ある場合、均等に割る
                        weight_bone_idxs = list(
                            set(
                                [
                                    mbone.index
                                    for mbone in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].map_bones.values()
                                    if mbone and mbone.name in model.bones and mbone.getVisibleFlag()
                                ]
                            )
                        )
                        weights = np.array([1 for _ in range(len(weight_bone_idxs))])

                        # 正規化
                        deform_weights = (weights / weights.sum(axis=0, keepdims=1)).tolist()

                        if np.count_nonzero(weight_bone_idxs) == 0:
                            continue
                        elif np.count_nonzero(weight_bone_idxs) == 1:
                            vv.deform = Bdef1(weight_bone_idxs[0])
                            logger.debug(f"ボーン位置 BDEF1 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        elif np.count_nonzero(weight_bone_idxs) == 2:
                            vv.deform = Bdef2(
                                weight_bone_idxs[0],
                                weight_bone_idxs[1],
                                deform_weights[0],
                            )
                            logger.debug(f"ボーン位置 BDEF2 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        else:
                            # 3つの場合にうまくいかないので、後ろに追加しておく
                            deform_weights += [0 for _ in range(4)]
                            if root_bone:
                                weight_bone_idxs += [root_bone.index for _ in range(4)]
                            else:
                                if vv.position().x() > 0:
                                    weight_bone_idxs += [left_root_bone.index for _ in range(4)]
                                else:
                                    weight_bone_idxs += [right_root_bone.index for _ in range(4)]

                            vv.deform = Bdef4(
                                weight_bone_idxs[0],
                                weight_bone_idxs[1],
                                weight_bone_idxs[2],
                                weight_bone_idxs[3],
                                deform_weights[0],
                                deform_weights[1],
                                deform_weights[2],
                                deform_weights[3],
                            )

                            logger.debug(f"ボーン位置 BDEF4 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")

                        # 頂点位置にボーンが登録されている場合、BDEF1登録対象
                        for rv in vv.real_vertices:
                            rv.deform = vv.deform

                            # 逆登録
                            for weight_bone_idx in weight_bone_idxs:
                                if weight_bone_idx not in model.vertices:
                                    model.vertices[weight_bone_idx] = []
                                model.vertices[weight_bone_idx].append(rv)

                        # 登録対象の場合、残対象から削除
                        if vv.real_vertices and vkey in remaining_vertices:
                            del remaining_vertices[vkey]

                    elif np.where(registered_bones[:, v_xidx])[0].shape[0] > 1:
                        # 同じX位置にボーンがある場合、縦のBDEF2登録対象
                        if (
                            not all_bone_vertical_distances[base_map_idx].any()
                            or np.isnan(vertex_map[above_yidx, v_xidx]).any()
                            or np.isnan(vertex_map[below_yidx, v_xidx]).any()
                            or not virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])].map_bones.get(
                                base_map_idx, None
                            )
                            or not virtual_vertices[tuple(vertex_map[below_yidx, v_xidx])].map_bones.get(
                                base_map_idx, None
                            )
                        ):
                            continue

                        above_weight = np.nan_to_num(
                            np.sum(all_bone_vertical_distances[base_map_idx][(above_yidx + 1) : (v_yidx + 1), v_xidx])
                            / np.sum(
                                all_bone_vertical_distances[base_map_idx][(above_yidx + 1) : (below_yidx + 1), v_xidx]
                            )
                        )

                        weight_bone_idx_0 = (
                            virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])].map_bones[base_map_idx].index
                        )
                        weight_bone_idx_1 = (
                            virtual_vertices[tuple(vertex_map[below_yidx, v_xidx])].map_bones[base_map_idx].index
                        )

                        if (
                            np.isclose(above_weight, 0)
                            and model.bones[model.bone_indexes[weight_bone_idx_0]].getVisibleFlag()
                        ):
                            vv.deform = Bdef1(weight_bone_idx_0)

                            logger.debug(f"縦 BDEF1(0) vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        elif (
                            np.isclose(above_weight, 1)
                            and model.bones[model.bone_indexes[weight_bone_idx_1]].getVisibleFlag()
                        ):
                            vv.deform = Bdef1(weight_bone_idx_1)

                            logger.debug(f"縦 BDEF1(1) vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        elif (
                            model.bones[model.bone_indexes[weight_bone_idx_0]].getVisibleFlag()
                            and model.bones[model.bone_indexes[weight_bone_idx_1]].getVisibleFlag()
                        ):
                            vv.deform = Bdef2(
                                weight_bone_idx_0,
                                weight_bone_idx_1,
                                1 - above_weight,
                            )

                            logger.debug(f"縦 BDEF2 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")

                        if vv.deform:
                            for rv in vv.real_vertices:
                                rv.deform = vv.deform

                                # 逆登録
                                if weight_bone_idx_0 not in model.vertices:
                                    model.vertices[weight_bone_idx_0] = []
                                model.vertices[weight_bone_idx_0].append(rv)
                                if weight_bone_idx_1 not in model.vertices:
                                    model.vertices[weight_bone_idx_1] = []
                                model.vertices[weight_bone_idx_1].append(rv)

                            # 登録対象の場合、残対象から削除
                            if vv.real_vertices and vkey in remaining_vertices:
                                del remaining_vertices[vkey]

                    elif np.where(registered_bones[v_yidx, :])[0].shape[0] > 1:
                        # 同じY位置にボーンがある場合、横のBDEF2登録対象
                        if v_xidx < registered_bones.shape[1] - 1 and registered_bones[v_yidx, (v_xidx + 1) :].any():
                            registered_next_xidx = next_xidx
                        else:
                            registered_next_xidx = 0

                        if (
                            not all_bone_horizonal_distances[base_map_idx].any()
                            or vertex_maps[prev_map_idx].shape[0] <= v_yidx
                            or vertex_maps[prev_map_idx].shape[1] <= prev_xidx
                            or np.isnan(vertex_maps[prev_map_idx][v_yidx, prev_xidx]).any()
                            or vertex_maps[next_map_idx].shape[0] <= v_yidx
                            or vertex_maps[next_map_idx].shape[1] <= registered_next_xidx
                            or np.isnan(vertex_maps[next_map_idx][v_yidx, registered_next_xidx]).any()
                            or not virtual_vertices[tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx])].map_bones.get(
                                prev_map_idx, None
                            )
                            or not virtual_vertices[
                                tuple(vertex_maps[next_map_idx][v_yidx, registered_next_xidx])
                            ].map_bones.get(next_map_idx, None)
                        ):
                            continue

                        if next_connected and next_xidx == 0:
                            # 最後の頂点の場合、とりあえず次の距離を対象とする
                            next_xidx = vertex_map.shape[1]

                        prev_weight = np.nan_to_num(
                            np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx])
                            / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                        )

                        weight_bone_idx_0 = (
                            virtual_vertices[tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx])]
                            .map_bones[prev_map_idx]
                            .index
                        )
                        weight_bone_idx_1 = (
                            virtual_vertices[tuple(vertex_maps[next_map_idx][v_yidx, registered_next_xidx])]
                            .map_bones[next_map_idx]
                            .index
                        )

                        if (
                            np.isclose(prev_weight, 0)
                            and model.bones[model.bone_indexes[weight_bone_idx_0]].getVisibleFlag()
                        ):
                            vv.deform = Bdef1(weight_bone_idx_0)

                            logger.debug(f"横 BDEF1(0) vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        elif (
                            np.isclose(prev_weight, 1)
                            and model.bones[model.bone_indexes[weight_bone_idx_1]].getVisibleFlag()
                        ):
                            vv.deform = Bdef1(weight_bone_idx_1)
                            logger.debug(f"横 BDEF1(1) vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")
                        elif (
                            model.bones[model.bone_indexes[weight_bone_idx_0]].getVisibleFlag()
                            and model.bones[model.bone_indexes[weight_bone_idx_1]].getVisibleFlag()
                        ):
                            vv.deform = Bdef2(
                                weight_bone_idx_0,
                                weight_bone_idx_1,
                                1 - prev_weight,
                            )

                            logger.debug(f"BDEF2 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")

                        if vv.deform:
                            for rv in vv.real_vertices:
                                rv.deform = vv.deform

                                # 逆登録
                                if weight_bone_idx_0 not in model.vertices:
                                    model.vertices[weight_bone_idx_0] = []
                                model.vertices[weight_bone_idx_0].append(rv)
                                if weight_bone_idx_1 not in model.vertices:
                                    model.vertices[weight_bone_idx_1] = []
                                model.vertices[weight_bone_idx_1].append(rv)

                            # 登録対象の場合、残対象から削除
                            if vv.real_vertices and vkey in remaining_vertices:
                                del remaining_vertices[vkey]

                    else:
                        if next_connected and next_xidx == 0:
                            # 最後の頂点の場合、とりあえず次の距離を対象とする
                            next_xidx = vertex_map.shape[1]
                            target_next_xidx = 0
                        else:
                            target_next_xidx = next_xidx

                        if (
                            not all_bone_vertical_distances[base_map_idx].any()
                            or not all_bone_horizonal_distances[base_map_idx].any()
                            or vertex_maps[prev_map_idx].shape[0] <= above_yidx
                            or vertex_maps[prev_map_idx].shape[1] <= prev_xidx
                            or np.isnan(vertex_maps[prev_map_idx][above_yidx, prev_xidx]).any()
                            or vertex_maps[next_map_idx].shape[0] <= above_yidx
                            or vertex_maps[next_map_idx].shape[1] <= target_next_xidx
                            or np.isnan(vertex_maps[next_map_idx][above_yidx, target_next_xidx]).any()
                            or vertex_maps[prev_map_idx].shape[0] <= below_yidx
                            or vertex_maps[prev_map_idx].shape[1] <= prev_xidx
                            or np.isnan(vertex_maps[prev_map_idx][below_yidx, prev_xidx]).any()
                            or vertex_maps[next_map_idx].shape[0] <= below_yidx
                            or vertex_maps[next_map_idx].shape[1] <= target_next_xidx
                            or np.isnan(vertex_maps[next_map_idx][below_yidx, target_next_xidx]).any()
                            or not virtual_vertices[
                                tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx])
                            ].map_bones.get(prev_map_idx, None)
                            or not virtual_vertices[tuple(vertex_map[above_yidx, target_next_xidx])].map_bones.get(
                                next_map_idx, None
                            )
                            or not virtual_vertices[
                                tuple(vertex_maps[prev_map_idx][below_yidx, prev_xidx])
                            ].map_bones.get(prev_map_idx, None)
                            or not virtual_vertices[
                                tuple(vertex_maps[next_map_idx][below_yidx, target_next_xidx])
                            ].map_bones.get(next_map_idx, None)
                        ):
                            continue

                        prev_above_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, v_xidx])
                                / np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, v_xidx])
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        next_above_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, v_xidx])
                                / np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, v_xidx])
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        prev_below_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, v_xidx])
                                / np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, v_xidx])
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        next_below_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, v_xidx])
                                / np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, v_xidx])
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        # ほぼ0のものは0に置換（円周用）
                        total_weights = np.array(
                            [prev_above_weight, next_above_weight, prev_below_weight, next_below_weight]
                        )
                        total_weights[np.isclose(total_weights, 0, equal_nan=True)] = 0

                        if np.count_nonzero(total_weights):
                            deform_weights = total_weights / total_weights.sum(axis=0, keepdims=1)

                            weight_bone_idx_0 = (
                                virtual_vertices[tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx])]
                                .map_bones[prev_map_idx]
                                .index
                            )
                            weight_bone_idx_1 = (
                                virtual_vertices[tuple(vertex_maps[next_map_idx][above_yidx, target_next_xidx])]
                                .map_bones[next_map_idx]
                                .index
                            )
                            weight_bone_idx_2 = (
                                virtual_vertices[tuple(vertex_maps[prev_map_idx][below_yidx, prev_xidx])]
                                .map_bones[prev_map_idx]
                                .index
                            )
                            weight_bone_idx_3 = (
                                virtual_vertices[tuple(vertex_maps[next_map_idx][below_yidx, target_next_xidx])]
                                .map_bones[next_map_idx]
                                .index
                            )

                            vv.deform = Bdef4(
                                weight_bone_idx_0,
                                weight_bone_idx_1,
                                weight_bone_idx_2,
                                weight_bone_idx_3,
                                deform_weights[0],
                                deform_weights[1],
                                deform_weights[2],
                                deform_weights[3],
                            )

                            for rv in vv.real_vertices:
                                rv.deform = vv.deform

                                # 逆登録
                                if weight_bone_idx_0 not in model.vertices:
                                    model.vertices[weight_bone_idx_0] = []
                                model.vertices[weight_bone_idx_0].append(rv)
                                if weight_bone_idx_1 not in model.vertices:
                                    model.vertices[weight_bone_idx_1] = []
                                model.vertices[weight_bone_idx_1].append(rv)
                                if weight_bone_idx_2 not in model.vertices:
                                    model.vertices[weight_bone_idx_2] = []
                                model.vertices[weight_bone_idx_1].append(rv)
                                if weight_bone_idx_3 not in model.vertices:
                                    model.vertices[weight_bone_idx_3] = []
                                model.vertices[weight_bone_idx_3].append(rv)

                            logger.debug(f"BDEF4 vkey[{vkey}], vidxs[{vv.vidxs()}], deform[{vv.deform}]")

                            # 登録対象の場合、残対象から削除
                            if vv.real_vertices and vkey in remaining_vertices:
                                del remaining_vertices[vkey]

                    weight_cnt += len(vv.real_vertices)
                    if weight_cnt > 0 and weight_cnt // 1000 > prev_weight_cnt:
                        logger.info("-- --【No.%s】頂点ウェイト: %s個目:終了", base_map_idx + 1, weight_cnt)
                        prev_weight_cnt = weight_cnt // 1000

            logger.info("-- --【No.%s】頂点ウェイト: %s個目:終了", base_map_idx + 1, weight_cnt)

        return remaining_vertices

    def create_root_bone(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        root_pos: MVector3D,
        base_vertical_axis: MVector3D,
        parent_bone_name: str,
        direction: str = "",
    ):
        # 略称
        abb_name = param_option["abb_name"]
        # 表示枠名
        display_name = f"{abb_name}{direction}:{material_name}"
        # 親ボーン
        parent_bone = model.bones[parent_bone_name]

        # 中心ボーン
        root_bone = Bone(
            f"{abb_name}{direction}中心",
            f"{abb_name}{direction}Root",
            root_pos,
            parent_bone.index,
            0,
            0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010,
        )
        if root_bone.name in model.bones:
            logger.warning("同じボーン名が既に登録されているため、末尾に乱数を追加します。 既存ボーン名: %s", root_bone.name)
            root_bone.name += randomname(3)

        root_bone.index = len(model.bones)
        root_bone.tail_position = base_vertical_axis * 2

        # 表示枠
        model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

        return display_name, root_bone

    def create_bone(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        vertex_map_orders: dict,
        threshold: float,
        base_vertical_axis: MVector3D,
    ):
        logger.info("【%s:%s】ボーン生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        # 中心ボーン生成

        # 略称
        abb_name = param_option["abb_name"]
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"
        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]
        # 足に繋げるか否か
        is_rigidbody_leg = param_option["rigidbody_leg"]

        # 中心ボーン
        root_bone = left_root_bone = right_root_bone = left_display_name = right_display_name = None
        root_bone_indexes = []
        if is_rigidbody_leg:
            # 全体のX位置を取得
            vxs = [
                virtual_vertices[tuple(vv)].position().x()
                for v_map in vertex_maps.values()
                for vvs in v_map
                for vv in vvs
            ]
            # 足に繋げる場合
            if np.max(vxs) >= -0.1:
                right_display_name, right_root_bone = self.create_root_bone(
                    model,
                    param_option,
                    material_name,
                    model.bones["右足"].position.copy(),
                    base_vertical_axis,
                    "右足",
                    "右",
                )
            if np.min(vxs) <= 0.1:
                left_display_name, left_root_bone = self.create_root_bone(
                    model,
                    param_option,
                    material_name,
                    model.bones["左足"].position.copy(),
                    base_vertical_axis,
                    "左足",
                    "左",
                )

            if left_root_bone and right_root_bone:
                # 両足あるならひとつ加算しておく
                right_root_bone.index += 1

                tmp_all_bones = {left_root_bone.name: left_root_bone, right_root_bone.name: right_root_bone}
                tmp_all_bone_indexes = {
                    left_root_bone.index: left_root_bone.name,
                    right_root_bone.index: right_root_bone.name,
                }
                root_bone_indexes.append(left_root_bone.index)
                root_bone_indexes.append(right_root_bone.index)
            elif right_root_bone:
                tmp_all_bones = {right_root_bone.name: right_root_bone}
                tmp_all_bone_indexes = {
                    right_root_bone.index: right_root_bone.name,
                }
                root_bone_indexes.append(right_root_bone.index)
            elif left_root_bone:
                tmp_all_bones = {left_root_bone.name: left_root_bone}
                tmp_all_bone_indexes = {
                    left_root_bone.index: left_root_bone.name,
                }
                root_bone_indexes.append(left_root_bone.index)
        else:
            display_name, root_bone = self.create_root_bone(
                model, param_option, material_name, parent_bone.position.copy(), base_vertical_axis, parent_bone.name
            )

            tmp_all_bones = {root_bone.name: root_bone}
            tmp_all_bone_indexes = {root_bone.index: root_bone.name}
            root_bone_indexes.append(root_bone.index)

        tmp_all_bone_vvs = {}

        logger.info("【%s】頂点距離の算出", material_name)

        all_bone_horizonal_distances = {}
        all_bone_vertical_distances = {}
        all_bone_connected = {}

        for base_map_idx, vertex_map in vertex_maps.items():
            logger.info("--【No.%s】頂点距離算出", base_map_idx + 1)

            prev_vertex_cnt = 0
            vertex_cnt = 0

            bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))
            bone_vertical_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))
            bone_connected = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

            # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
            for v_yidx in range(vertex_map.shape[0]):
                # 一列しかない場合は縦方向を計れるようにオフセットをOFFにする
                x_end_offset = 1 if vertex_map.shape[1] > 1 else 0
                for v_xidx in range(0, vertex_map.shape[1] - x_end_offset):
                    if (
                        v_xidx < vertex_map.shape[1] - 1
                        and not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                        and not np.isnan(vertex_map[v_yidx, v_xidx + 1]).any()
                    ):
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])].position()
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)

                        if (
                            tuple(vertex_map[v_yidx, v_xidx])
                            in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])].connected_vvs
                        ):
                            # 前の仮想頂点と繋がっている場合、True
                            bone_connected[v_yidx, v_xidx] = True

                    if (
                        v_yidx < vertex_map.shape[0] - 1
                        and not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                        and not np.isnan(vertex_map[v_yidx + 1, v_xidx]).any()
                    ):
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx + 1, v_xidx])].position()
                        bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)

                    vertex_cnt += 1
                    if vertex_cnt > 0 and vertex_cnt // 1000 > prev_vertex_cnt:
                        logger.info("-- --【No.%s】頂点距離算出: %s個目:終了", base_map_idx + 1, vertex_cnt)
                        prev_vertex_cnt = vertex_cnt // 1000

                if v_xidx > 0:
                    v_xidx += 1
                # 最終行の縦距離は前頂点の距離を流用
                bone_vertical_distances[v_yidx, v_xidx] = float(bone_vertical_distances[v_yidx, v_xidx - 1])
                if not np.isnan(vertex_map[v_yidx, v_xidx]).any():
                    # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
                    next_base_map_idx = base_map_idx + 1 if base_map_idx + 1 in vertex_maps else 0
                    if (
                        vertex_maps[next_base_map_idx].shape[0] > v_yidx
                        and tuple(vertex_maps[next_base_map_idx][v_yidx, 0])
                        in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].connected_vvs
                    ):
                        # 横の仮想頂点と繋がっている場合、Trueで有効な距離を入れておく
                        bone_connected[v_yidx, v_xidx] = True

                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_maps[next_base_map_idx][v_yidx, 0])].position()
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)
                    else:
                        # とりあえずINT最大値を入れておく
                        bone_horizonal_distances[v_yidx, v_xidx] = np.iinfo(np.int).max

            logger.debug("bone_horizonal_distances ------------")
            logger.debug(np.round(bone_horizonal_distances, decimals=3).tolist())
            logger.debug("bone_vertical_distances ------------")
            logger.debug(np.round(bone_vertical_distances, decimals=3).tolist())
            logger.debug("bone_connected ------------")
            logger.debug(bone_connected.tolist())

            all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
            all_bone_vertical_distances[base_map_idx] = bone_vertical_distances
            all_bone_connected[base_map_idx] = bone_connected

        if len(vertex_maps) > 1:
            for base_map_idx, map_next_idx in zip(
                vertex_maps.keys(), list(vertex_maps.keys())[1:] + [list(vertex_maps.keys())[0]]
            ):
                vertex_map = vertex_maps[base_map_idx]
                next_vertex_map = vertex_maps[map_next_idx]
                # 複数マップある場合、繋ぎ目をチェックする
                for v_yidx in range(vertex_map.shape[0]):
                    if (
                        not np.isnan(vertex_map[v_yidx, -1]).any()
                        and next_vertex_map.shape[0] > v_yidx
                        and not np.isnan(next_vertex_map[v_yidx, 0]).any()
                        and (
                            tuple(next_vertex_map[v_yidx, 0])
                            in virtual_vertices[tuple(vertex_map[v_yidx, -1])].connected_vvs
                        )
                    ):
                        # 繋がれた仮想頂点の場合、繋がれているとみなす
                        all_bone_connected[base_map_idx][v_yidx, -1] = True

        # 全体通してのX番号
        prev_xs = []
        all_registered_bones = {}
        all_y_bone_registers = []

        for base_map_idx, vertex_map in vertex_maps.items():
            # Y軸は全体を通して同じ箇所にボーンを張るため、上と繋がっていない箇所のピックアップ
            y_diffs = np.diff(all_bone_connected[base_map_idx], axis=0)
            y_diff_xs = np.argmin(y_diffs, axis=0) | np.argmax(y_diffs, axis=0)
            y_diff_xs_prev = []
            if len(vertex_maps) > 1:
                if base_map_idx == 0:
                    if set(np.unique(all_bone_connected[list(vertex_maps.keys())[1]][:, -1]).tolist()) == {0, 1}:
                        # 最後のマップと最初のマップが繋がってる箇所と繋がってない箇所がある場合
                        y_diff_xs_prev.append(np.max(np.where(all_bone_connected[list(vertex_maps.keys())[1]][:, -1])))
                else:
                    if set(np.unique(all_bone_connected[base_map_idx - 1][:, -1]).tolist()) == {0, 1}:
                        # 現在のマップと前のマップが繋がってる箇所と繋がってない箇所がある場合
                        y_diff_xs_prev.append(np.max(np.where(all_bone_connected[base_map_idx - 1][:, -1])))

            logger.debug(
                "[%s] y_diff_xs_prev: %s", base_map_idx, sorted(list(set(np.where(y_diff_xs_prev)[0].tolist())))
            )
            logger.debug(
                "[%s] y_diff_xs: %s", base_map_idx, sorted(list(set(y_diff_xs[np.where(np.diff(y_diff_xs))].tolist())))
            )

            all_y_bone_registers.extend(y_diff_xs_prev)
            all_y_bone_registers.extend(y_diff_xs[np.where(np.diff(y_diff_xs))].tolist())

        logger.debug("all_y_bone_registers ------------")
        logger.debug(sorted(list(set(np.where(all_y_bone_registers)[0].tolist()))))

        for base_map_idx, vertex_map in vertex_maps.items():

            prev_bone_cnt = 0
            bone_cnt = 0

            if base_map_idx not in vertex_map_orders:
                # ボーン生成対象外の場合、スルー
                continue

            # 全ボーン登録有無
            full_registered_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)
            # 実際のボーン登録有無
            registered_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

            logger.info("--【No.%s】ボーン生成", base_map_idx + 1)

            if param_option["density_type"] == logger.transtext("中央"):
                # 間隔がセンタータイプの場合、真ん中にのみボーンを張る
                # 末端は非表示で登録する
                for v_yidx in list(range(0, vertex_map.shape[0], param_option["vertical_bone_density"])) + [
                    vertex_map.shape[0] - 1
                ]:
                    # メッシュX個数：奇数 = 真ん中、偶数 = 始点寄り
                    v_xidx = (vertex_map.shape[1] // 2) - (1 - vertex_map.shape[1] % 2)
                    if not np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        registered_bones[v_yidx, v_xidx] = True
            else:
                # ボーンが繋がってるトコロが生成範囲
                full_registered_bones = all_bone_connected[base_map_idx].copy()
                # # 先頭は最初のをコピー
                # full_registered_bones[:, -1] = full_registered_bones[:, -2].copy()
                # プラスの場合、前と繋がっていないので登録対象
                full_registered_bones[:, 1:][np.where(np.diff(all_bone_connected[base_map_idx], axis=1) < 0)] = True

                y_registers = np.zeros(vertex_map.shape[0], dtype=np.int)
                x_registers = np.zeros(vertex_map.shape[1], dtype=np.int)

                if np.where(all_bone_connected[base_map_idx][:-1, -1] == 0)[0].any():
                    # X方向の末端がどこか繋がって無いとこがあったら末端にボーンを張る
                    x_registers[-1] = True

                if param_option["density_type"] == logger.transtext("距離"):
                    mean_vertical_distance = np.mean(
                        [
                            np.mean(np.array(distances)[np.array(distances) != np.iinfo(np.int).max])
                            for distances in all_bone_vertical_distances.values()
                        ]
                    )
                    mean_horizonal_distance = np.mean(
                        [
                            np.mean(np.array(distances)[np.array(distances) != np.iinfo(np.int).max])
                            for distances in all_bone_horizonal_distances.values()
                        ]
                    )

                    logger.debug(
                        f"median_horizonal_distance: {round(mean_horizonal_distance, 4)}, median_vertical_distance: {round(mean_vertical_distance, 4)}"
                    )

                    # 間隔が距離タイプの場合、均等になるように間を空ける
                    prev_y_registered = 1
                    for v_yidx in range(2, vertex_map.shape[0]):
                        if (
                            np.sum(
                                np.mean(all_bone_vertical_distances[base_map_idx][prev_y_registered:v_yidx, :], axis=1)
                            )
                            > mean_vertical_distance * param_option["vertical_bone_density"]
                        ):
                            # 前の登録ボーンから一定距離離れたら登録対象
                            y_registers[v_yidx] = True
                            prev_y_registered = v_yidx

                    prev_x_registered = 0
                    for v_xidx in range(vertex_map.shape[1]):
                        if (
                            np.sum(
                                np.mean(
                                    all_bone_horizonal_distances[base_map_idx][:, prev_x_registered:v_xidx], axis=0
                                )
                            )
                            > mean_horizonal_distance * param_option["horizonal_bone_density"]
                        ):
                            # 前の登録ボーンから一定距離離れたら登録対象
                            x_registers[v_xidx] = True
                            prev_x_registered = v_xidx

                elif param_option["density_type"] == logger.transtext("頂点"):
                    # 根元を一列だけ埋めるため、1余分に測る
                    if param_option["vertical_bone_density"] == 1:
                        y_registers = np.array([True for _ in range(vertex_map.shape[0])])
                    else:
                        y_registers = (
                            np.array(list(range(1, vertex_map.shape[0]))) % param_option["vertical_bone_density"] == 1
                        )
                    x_registers = (
                        np.array(list(range(vertex_map.shape[1]))) % param_option["horizonal_bone_density"] == 0
                    )

                # 根元＋最上段一列は必ず埋める
                y_registers[:2] = True
                # 最初は必ず登録する
                x_registers[0] = True

                logger.debug("[%s] x_registers: %s", base_map_idx, x_registers)
                logger.debug("[%s] y_registers: %s", base_map_idx, y_registers)

                # 前と繋がっていない箇所のピックアップ
                x_diffs = np.diff(all_bone_connected[base_map_idx], axis=1)
                x_bone_registers = sorted(
                    list(
                        set(np.where(x_registers)[0].tolist())
                        | set((np.where(x_diffs == 1)[1] + 1).tolist())
                        | set((np.where(x_diffs == -1)[1] + 1).tolist())
                    )
                )
                logger.debug("[%s] x_bone_registers: %s", base_map_idx, x_bone_registers)

                # Y軸は全体を通してボーンを張るべき箇所＋自身の登録対象位置を登録対象とする
                y_bone_registers = sorted(list(set(np.where(y_registers)[0].tolist()) | set(all_y_bone_registers)))
                logger.debug("[%s] y_bone_registers: %s", base_map_idx, y_bone_registers)

                # XY軸でボーンを登録する
                # 一旦実際に頂点があるところのみ対象とする
                registered_idxs = np.array(
                    [
                        np.array([y, x])
                        for y in y_bone_registers
                        for x in x_bone_registers
                        if y < vertex_map.shape[0]
                        and x < vertex_map.shape[1]
                        and not np.isnan(vertex_map[y, x]).any()
                        and virtual_vertices[tuple(vertex_map[y, x])].vidxs()
                    ]
                )
                registered_bones[registered_idxs[:, 0], registered_idxs[:, 1]] = True

                for v_xidx in range(vertex_map.shape[1]):
                    # 末端ボーンを追加登録
                    x_registered = np.where(registered_bones[:, v_xidx])[0]
                    if x_registered.any():
                        # first_connected_v_yidx = np.min(x_registered)
                        # 末端は横と繋がってない場合もあるので全体から見る
                        last_connected_v_yidx = max(
                            [
                                y
                                for y in range(vertex_map.shape[0])
                                if virtual_vertices[tuple(vertex_map[y, v_xidx])].vidxs()
                            ]
                        )

                        # 末端ボーン追加登録
                        registered_bones[
                            last_connected_v_yidx : min(last_connected_v_yidx + 2, vertex_map.shape[0]),
                            v_xidx,
                        ] = True

                if len(vertex_maps) > 1 and base_map_idx == 0:
                    # 最初の場合、最後と繋がっているトコロをチェックする
                    next_x_registered = np.where(all_bone_connected[len(all_bone_connected) - 1][:, -1])[0]
                    if next_x_registered.any():
                        next_last_connected_v_yidx = np.max(next_x_registered)
                        registered_bones[
                            next_last_connected_v_yidx : min(next_last_connected_v_yidx + 1, vertex_map.shape[0] - 1),
                            0,
                        ] = True
                elif base_map_idx > 0:
                    # 2枚目以降の場合、前と繋がってる最後のトコロにはボーンを張る
                    prev_x_registered = np.where(all_bone_connected[base_map_idx - 1][:, -1])[0]
                    if prev_x_registered.any():
                        prev_last_connected_v_yidx = np.max(prev_x_registered)
                        registered_bones[
                            prev_last_connected_v_yidx : min(prev_last_connected_v_yidx + 1, vertex_map.shape[0] - 1),
                            0,
                        ] = True

                # for v_yidx in np.where(np.sum(registered_bones, axis=1))[0]:
                #     # 横方向のボーン登録が必要なのでX登録対象ボーンまでの追加登録
                #     registered_bones[v_yidx, np.where(x_registers & full_registered_bones[v_yidx, :])] = True

            all_registered_bones[base_map_idx] = registered_bones

            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any() or not registered_bones[v_yidx, v_xidx]:
                        # 登録対象ではない場合、スルー
                        continue

                    v_yno = v_yidx + 1
                    v_xno = v_xidx + len(prev_xs) + 1

                    vkey = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices[vkey]

                    # 親は既にモデルに登録済みのものを選ぶ
                    parent_bone = None
                    for parent_v_yidx in range(v_yidx - 1, -1, -1):
                        target_vkey = vertex_map[parent_v_yidx, v_xidx]
                        if np.isnan(target_vkey).any():
                            continue

                        parent_bone = virtual_vertices[tuple(target_vkey)].map_bones.get(base_map_idx, None)
                        if parent_bone and (parent_bone.name in model.bones or parent_bone.name in tmp_all_bones):
                            # 登録されていたら終了
                            break

                    # is_regist = True
                    # substitute_bone = None
                    if not parent_bone:
                        if v_yidx > 0:
                            # 0番目以降は既に登録済みの上のボーンを採用する

                            # 親候補のX位置一覧
                            parent_xidxs = (
                                list(range(v_xidx - 1, -1, -1))
                                if v_xidx > 0
                                else list(range(v_xidx + 1, vertex_map.shape[1]))
                            )

                            for parent_v_xidx in parent_xidxs:
                                for parent_v_yidx in range(v_yidx - 1, -1, -1):
                                    target_vkey = vertex_map[parent_v_yidx, parent_v_xidx]
                                    if np.isnan(target_vkey).any():
                                        continue

                                    parent_bone = virtual_vertices[tuple(target_vkey)].map_bones.get(
                                        base_map_idx, None
                                    )
                                    if parent_bone and (
                                        parent_bone.name in model.bones or parent_bone.name in tmp_all_bones
                                    ):
                                        # 登録されていたら終了
                                        break
                                if parent_bone:
                                    break

                        if not parent_bone:
                            # 最後まで登録されている親ボーンが見つからなければ、ルート
                            if is_rigidbody_leg:
                                # 足に繋げるのであれば、左右に分ける
                                parent_bone = (
                                    left_root_bone if vv.position().x() > 0 and left_root_bone else right_root_bone
                                )
                            else:
                                parent_bone = root_bone

                    if not vv.map_bones.get(base_map_idx, None):
                        # ボーン仮登録
                        bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                        if bone_name in model.bones or bone_name in tmp_all_bones:
                            bone_name += randomname(3)
                            logger.warning("同じボーン名が既に登録されているため、末尾に乱数を追加します。 既存ボーン名: %s", bone_name)

                        bone = Bone(bone_name, bone_name, vv.position().copy(), parent_bone.index, 0, 0x0000 | 0x0002)
                        bone.local_z_vector = vv.normal().copy()

                        bone.parent_index = parent_bone.index
                        bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                        bone.local_z_vector *= MVector3D(-1, 1, -1)
                        if bone.local_z_vector == MVector3D():
                            bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, MVector3D(1, 0, 0))
                        # ローカル軸ありで回転可能
                        bone.flag |= 0x0800 | 0x0002

                        if registered_bones[v_yidx, v_xidx]:
                            # 一旦仮登録
                            vv.map_bones[base_map_idx] = bone
                            bone.index = len(model.bones) + len(tmp_all_bones)
                            tmp_all_bones[bone.name] = bone
                            tmp_all_bone_indexes[bone.index] = bone.name
                            tmp_all_bone_vvs[bone.name] = {"base_map_idx": base_map_idx, "vv": vv}

                            logger.debug(
                                f"tmp_all_bones: {bone.name}, parent: {parent_bone.name}, pos: {bone.position.to_log()}"
                            )

                    bone_cnt += 1
                    if bone_cnt > 0 and bone_cnt // 1000 > prev_bone_cnt:
                        logger.info("-- --【No.%s】ボーン生成: %s個目:終了", base_map_idx + 1, bone_cnt)
                        prev_bone_cnt = bone_cnt // 1000

            prev_xs.extend(list(range(vertex_map.shape[1])))

        logger.info("-- 親ボーンチェック")

        tmp_all_bone_parents = {}
        tmp_all_bone_targets = []
        for bi, bone_name in enumerate(tmp_all_bones.keys()):
            bone = tmp_all_bones[bone_name]

            # 自身と同じ位置のボーンを親に持つ子ボーン
            c_bones = [
                c_bone
                for c_bone in tmp_all_bones.values()
                if c_bone.index not in root_bone_indexes
                and (
                    (
                        c_bone.parent_index in tmp_all_bone_indexes
                        and tmp_all_bone_indexes[c_bone.parent_index] in tmp_all_bones
                        and tmp_all_bones[tmp_all_bone_indexes[c_bone.parent_index]].position == bone.position
                    )
                    or (
                        c_bone.parent_index in model.bone_indexes
                        and model.bone_indexes[c_bone.parent_index] in model.bones
                        and model.bones[model.bone_indexes[c_bone.parent_index]].position == bone.position
                    )
                )
            ]

            if not c_bones:
                # 子ボーンがない場合、そのまま登録
                tmp_all_bone_parents[bone.name] = tmp_all_bone_indexes[bone.parent_index]
                tmp_all_bone_targets.append(bone.name)
            else:
                c_bone_poses = {}
                for c_bone in c_bones:
                    pos = tuple(c_bone.position.data().tolist())
                    if pos not in c_bone_poses:
                        c_bone_poses[pos] = []
                    c_bone_poses[pos].append(c_bone)

                for in_c_bones in c_bone_poses.values():
                    tmp_all_bone_targets.append(in_c_bones[0].name)
                    parent_bone_name = tmp_all_bone_indexes[in_c_bones[0].parent_index]
                    tmp_all_bone_parents[in_c_bones[0].name] = parent_bone_name
                    if len(in_c_bones) > 1:
                        for c_bone in in_c_bones[1:]:
                            # 2つ以上ある場合、後のボーンは最初のボーンの親を代用する
                            tmp_all_bone_parents[c_bone.name] = parent_bone_name
                            tmp_all_bone_targets.append(c_bone.name)

                            # 自分の親は登録対象外とする
                            if c_bone.parent_index in tmp_all_bone_indexes:
                                for n, p_name in enumerate(tmp_all_bone_targets):
                                    if p_name == tmp_all_bone_indexes[c_bone.parent_index]:
                                        del tmp_all_bone_targets[n]
                                        break

            if bi > 0 and bi % 100 == 0:
                logger.info("-- -- 親ボーンチェック: %s個目:終了", bi)

        logger.info("-- -- 親ボーンチェック: %s個目:終了", bi)

        # このタイミングで中心ボーン登録
        if is_rigidbody_leg:
            if left_root_bone and left_display_name:
                model.bones[left_root_bone.name] = left_root_bone
                model.bone_indexes[left_root_bone.index] = left_root_bone.name
                model.display_slots[left_display_name].references.append((0, model.bones[left_root_bone.name].index))

            if right_root_bone and right_display_name:
                model.bones[right_root_bone.name] = right_root_bone
                model.bone_indexes[right_root_bone.index] = right_root_bone.name
                model.display_slots[right_display_name].references.append((0, model.bones[right_root_bone.name].index))
        else:
            model.bones[root_bone.name] = root_bone
            model.bone_indexes[root_bone.index] = root_bone.name
            model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

        logger.info("-- ボーン配置")

        for bi, bone_name in enumerate(sorted(tmp_all_bones.keys())):
            if bone_name not in tmp_all_bone_targets:
                if bone_name in tmp_all_bone_vvs and tmp_all_bone_vvs[bone_name]["vv"].map_bones:
                    # 同じ箇所に既にボーン定義がある場合、それを流用
                    tmp_all_bone_vvs[bone_name]["vv"].map_bones[
                        tmp_all_bone_vvs[bone_name]["base_map_idx"]
                    ] = tmp_all_bone_vvs[bone_name]["vv"].map_bones[
                        list(tmp_all_bone_vvs[bone_name]["vv"].map_bones.keys())[0]
                    ]
                if bi % 100 == 0:
                    logger.info("-- -- ボーン配置: %s個目:終了", bi)
                continue

            bone = tmp_all_bones[bone_name]
            bone.index = len(model.bones)
            model.bones[bone.name] = bone
            model.bone_indexes[bone.index] = bone_name

            if is_rigidbody_leg:
                if bone.position.x() > 0 and left_display_name:
                    model.display_slots[left_display_name].references.append((0, bone.index))
                else:
                    model.display_slots[right_display_name].references.append((0, bone.index))
            else:
                model.display_slots[display_name].references.append((0, bone.index))
            tmp_all_bone_vvs[bone.name]["vv"].map_bones[tmp_all_bone_vvs[bone.name]["base_map_idx"]] = bone

            if bone_name in tmp_all_bone_parents:
                # 親ボーンを付け替える必要がある場合
                parent_bone_name = tmp_all_bone_parents[bone_name]
                if parent_bone_name not in model.bones:
                    logger.warning(
                        "親ボーンを見つけられなかった為、中心ボーンに紐付けます: 親【%s】子【%s】",
                        parent_bone_name,
                        bone_name,
                        decoration=MLogger.DECORATION_BOX,
                    )
                    if is_rigidbody_leg:
                        parent_bone = left_root_bone if bone.position.x() > 0 and left_root_bone else right_root_bone
                    else:
                        parent_bone = root_bone
                else:
                    parent_bone = model.bones[parent_bone_name]

                # 登録対象である場合、そのまま登録
                bone.parent_index = parent_bone.index

                if parent_bone not in [root_bone, left_root_bone, right_root_bone]:
                    # 親ボーンの表示先再設定
                    parent_bone.tail_index = bone.index
                    # 親ボーンは表示ありで操作可能
                    parent_bone.flag |= 0x0008 | 0x0010 | 0x0001

            if bi > 0 and bi % 100 == 0:
                logger.info("-- -- ボーン配置: %s個目:終了", bi)

        logger.info("-- -- ボーン配置: %s個目:終了", bi)

        return (
            root_bone,
            left_root_bone,
            right_root_bone,
            virtual_vertices,
            all_registered_bones,
            all_bone_vertical_distances,
            all_bone_horizonal_distances,
            all_bone_connected,
        )

    def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
        return f"{abb_name}-{(v_yno):03d}-{(v_xno):03d}"

    def create_bust_physics(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        target_vertices: list,
        virtual_vertices: dict,
        vertex_maps: dict,
    ):
        logger.info("【%s:%s】胸物理生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        for base_map_idx, vertex_map in vertex_maps.items():
            for v_xidx in range(vertex_map.shape[1]):
                for v_yidx in range(vertex_map.shape[0]):
                    bone_key = tuple(vertex_map[v_yidx, v_xidx])
                    bone = virtual_vertices[bone_key].map_bones[base_map_idx]
                    if model.vertices.get(bone.index, None):
                        # 胸上ボーン
                        bust_top_pos = MVector3D(
                            np.average(
                                [model.bones["首"].position.data(), bone.position.data()], weights=[0.2, 0.8], axis=0
                            )
                        )
                        bust_top_pos.setX(bone.position.x())
                        bust_top_bone = Bone(
                            f"{bone.name}上",
                            f"{bone.english_name}_Above",
                            bust_top_pos,
                            bone.index,
                            0,
                            0x0001 | 0x0002 | 0x0008 | 0x0010,
                        )
                        bust_top_bone.index = len(model.bones)
                        model.bones[bust_top_bone.name] = bust_top_bone

                        # 胸下ボーン
                        bust_bottom_pos = MVector3D(
                            np.average(
                                [model.bones["上半身"].position.data(), bone.position.data()], weights=[0.3, 0.7], axis=0
                            )
                        )
                        bust_bottom_pos.setX(bone.position.x())
                        bust_bottom_bone = Bone(
                            f"{bone.name}下",
                            f"{bone.english_name}_Below",
                            bust_bottom_pos,
                            bone.index,
                            0,
                            0x0001 | 0x0002 | 0x0008 | 0x0010,
                        )
                        bust_bottom_bone.index = len(model.bones)
                        model.bones[bust_bottom_bone.name] = bust_bottom_bone

                        # 胸接続ボーン
                        bust_joint_pos = MVector3D(
                            np.average(
                                [
                                    model.bones[model.bone_indexes[bone.tail_index]].position.data(),
                                    bust_top_pos.data(),
                                ],
                                weights=[0.7, 0.3],
                                axis=0,
                            )
                        )
                        bust_joint_pos.setX(model.bones[model.bone_indexes[bone.tail_index]].position.x())
                        bust_joint_pos.setY(bust_top_pos.y())
                        bust_joint_bone = Bone(
                            f"{bone.name}接続",
                            f"{bone.english_name}_Joint",
                            bust_joint_pos,
                            bone.index,
                            0,
                            0x0001 | 0x0002 | 0x0008 | 0x0010,
                        )
                        bust_joint_bone.index = len(model.bones)
                        model.bones[bust_joint_bone.name] = bust_joint_bone

                        # 胸接続先ボーン
                        bust_joint_tail_pos = MVector3D(
                            np.average(
                                [
                                    model.bones[model.bone_indexes[bone.tail_index]].position.data(),
                                    bust_bottom_pos.data(),
                                ],
                                weights=[0.7, 0.3],
                                axis=0,
                            )
                        )
                        bust_joint_tail_pos.setX(model.bones[model.bone_indexes[bone.tail_index]].position.x())
                        bust_joint_tail_pos.setY(bust_bottom_pos.y())
                        bust_joint_tail_bone = Bone(
                            f"{bone.name}接続先",
                            f"{bone.english_name}_Joint_Tail",
                            bust_joint_tail_pos,
                            bone.index,
                            0,
                            0x0000,
                        )
                        bust_joint_tail_bone.index = len(model.bones)
                        model.bones[bust_joint_tail_bone.name] = bust_joint_tail_bone

                        bust_top_bone.tail_index = bust_joint_bone.index
                        bust_bottom_bone.tail_index = bust_joint_tail_bone.index
                        bust_joint_bone.tail_index = bust_joint_tail_bone.index

                        # ウェイトが乗ってるボーンがあれば載せ替え
                        for v in model.vertices[bone.index]:
                            if type(v.deform) is Bdef1 and v.deform.index0 == bone.index:
                                v.deform.index0 = bust_joint_bone.index
                            if type(v.deform) is Bdef2 or type(v.deform) is Sdef:
                                if v.deform.index0 == bone.index:
                                    v.deform.index0 = bust_joint_bone.index
                                if v.deform.index1 == bone.index:
                                    v.deform.index1 = bust_joint_bone.index
                            elif type(v.deform) is Bdef4:
                                if v.deform.index0 == bone.index:
                                    v.deform.index0 = bust_joint_bone.index
                                if v.deform.index1 == bone.index:
                                    v.deform.index1 = bust_joint_bone.index
                                if v.deform.index2 == bone.index:
                                    v.deform.index2 = bust_joint_bone.index
                                if v.deform.index3 == bone.index:
                                    v.deform.index3 = bust_joint_bone.index

                        bust_top_rigidbody = self.create_bust_rigidbody(param_option, bust_top_bone, bust_joint_bone)
                        bust_top_rigidbody.index = len(model.rigidbodies)
                        model.rigidbodies[bust_top_rigidbody.name] = bust_top_rigidbody

                        bust_bottom_rigidbody = self.create_bust_rigidbody(
                            param_option, bust_bottom_bone, bust_joint_tail_bone
                        )
                        bust_bottom_rigidbody.index = len(model.rigidbodies)
                        model.rigidbodies[bust_bottom_rigidbody.name] = bust_bottom_rigidbody

                        bust_joint_rigidbody = self.create_bust_rigidbody(
                            param_option, bust_joint_bone, bust_joint_tail_bone, is_joint=True
                        )
                        bust_joint_rigidbody.index = len(model.rigidbodies)
                        model.rigidbodies[bust_joint_rigidbody.name] = bust_joint_rigidbody

                        bust_top_joint = self.create_bust_joint(
                            self.get_rigidbody(model, bone.name),
                            bust_top_rigidbody,
                            bust_top_bone.position,
                            param_option["vertical_joint"],
                        )
                        bust_top_joint.index = len(model.joints)
                        model.joints[bust_top_joint.name] = bust_top_joint

                        bust_bottom_joint = self.create_bust_joint(
                            self.get_rigidbody(model, bone.name),
                            bust_bottom_rigidbody,
                            bust_bottom_bone.position,
                            param_option["vertical_joint"],
                        )
                        bust_bottom_joint.index = len(model.joints)
                        model.joints[bust_bottom_joint.name] = bust_bottom_joint

                        bust_joint_top = self.create_bust_joint(
                            bust_top_rigidbody,
                            bust_joint_rigidbody,
                            bust_joint_bone.position,
                            param_option["horizonal_joint"],
                        )
                        bust_joint_top.index = len(model.joints)
                        model.joints[bust_joint_top.name] = bust_joint_top

                        bust_joint_bottom = self.create_bust_joint(
                            bust_joint_rigidbody,
                            bust_bottom_rigidbody,
                            bust_joint_tail_bone.position,
                            param_option["diagonal_joint"],
                        )
                        bust_joint_bottom.index = len(model.joints)
                        model.joints[bust_joint_bottom.name] = bust_joint_bottom

                        logger.info("-- 胸物理: %s: 終了", bone.name)

                        break

    def create_bust_joint(
        self, a_rigidbody: RigidBody, b_rigidbody: RigidBody, joint_pos: MVector3D, param_joint: Joint
    ):
        joint_name = f"Bst|{a_rigidbody.name}|{b_rigidbody.name}"

        rotation_limit_min_radians = MVector3D(
            math.radians(param_joint.rotation_limit_min.x()),
            math.radians(param_joint.rotation_limit_min.y()),
            math.radians(param_joint.rotation_limit_min.z()),
        )
        rotation_limit_max_radians = MVector3D(
            math.radians(param_joint.rotation_limit_max.x()),
            math.radians(param_joint.rotation_limit_max.y()),
            math.radians(param_joint.rotation_limit_max.z()),
        )

        return Joint(
            joint_name,
            joint_name,
            0,
            a_rigidbody.index,
            b_rigidbody.index,
            joint_pos,
            MVector3D(),
            param_joint.translation_limit_min,
            param_joint.translation_limit_max,
            rotation_limit_min_radians,
            rotation_limit_max_radians,
            param_joint.spring_constant_translation,
            param_joint.spring_constant_rotation,
        )

    def create_bust_rigidbody(
        model: PmxModel, param_option: dict, bust_base_bone: Bone, bust_joint_bone: Bone, is_joint=False
    ):
        # 剛体情報
        param_rigidbody = param_option["rigidbody"]

        # ボーン進行方向(x)
        x_direction_pos = (bust_joint_bone.position - bust_base_bone.position).normalized()
        # ボーン進行方向に対しての横軸(y)
        y_direction_pos = MVector3D(-1, 0, 0)
        # ボーン進行方向に対しての縦軸(z)
        z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
        shape_qq = MQuaternion.rotationTo(z_direction_pos, x_direction_pos)
        shape_euler = shape_qq.toEulerAngles()
        shape_rotation_radians = MVector3D(
            math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z())
        )
        # 剛体位置
        shape_position = MVector3D(np.mean([bust_joint_bone.position, bust_base_bone.position]))
        # 剛体サイズ
        bust_length = bust_joint_bone.position.distanceToPoint(bust_base_bone.position)
        shape_size = (
            MVector3D(bust_length / 2, bust_length / 2, bust_length / 2)
            if is_joint
            else MVector3D(0.1, bust_length, 1)
        )

        rigidbody = RigidBody(
            bust_base_bone.name,
            bust_base_bone.english_name,
            bust_base_bone.index,
            param_rigidbody.collision_group,
            param_rigidbody.no_collision_group,
            0 if is_joint else 2,
            shape_size,
            shape_position,
            shape_rotation_radians,
            param_rigidbody.param.mass / 2 if is_joint else param_rigidbody.param.mass,
            param_rigidbody.param.linear_damping,
            param_rigidbody.param.angular_damping,
            param_rigidbody.param.restitution,
            param_rigidbody.param.friction,
            1,
        )
        rigidbody.shape_qq = shape_qq

        return rigidbody

    def create_bone_map(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        target_vertices: list,
        base_vertical_axis: MVector3D,
        is_root_bone=True,
    ):
        bone_grid = param_option["bone_grid"]
        bone_grid_cols = param_option["bone_grid_cols"]
        # 末端ボーンを加味するのでインクリメント
        bone_grid_rows = param_option["bone_grid_rows"] + 1

        logger.info("【%s:%s】ボーンマップ生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        logger.info("%s: ウェイトボーンの確認", material_name)

        # 閾値(とりあえず固定値)
        threshold = 0.0001

        virtual_vertices = {}

        # ウェイトボーンリスト取得
        weighted_vertex_positions = {}
        weighted_bone_pairs = []
        for n, v_idx in enumerate(model.material_vertices[material_name]):
            if v_idx not in target_vertices:
                continue

            v = model.vertex_dict[v_idx]
            v_key = v.position.to_key(threshold)
            if v_key not in virtual_vertices:
                virtual_vertices[v_key] = VirtualVertex(v_key)
            virtual_vertices[v_key].append([v], [], [])

            for weighted_bone_idx in v.deform.get_idx_list(weight=0.1):
                # ウェイトを持ってるボーンのINDEXを保持
                if weighted_bone_idx not in weighted_vertex_positions:
                    weighted_vertex_positions[weighted_bone_idx] = []
                weighted_vertex_positions[weighted_bone_idx].append(v.position.data())

            if type(v.deform) is Bdef2 or type(v.deform) is Sdef:
                if 0 < v.deform.weight0 < 1:
                    # 2つめのボーンも有効値を持っている場合、判定対象
                    key = (
                        min(v.deform.index0, v.deform.index1),
                        max(v.deform.index0, v.deform.index1),
                    )
                    if key not in weighted_bone_pairs:
                        weighted_bone_pairs.append(key)
            elif type(v.deform) is Bdef4:
                # ウェイト正規化
                total_weights = np.array([v.deform.weight0, v.deform.weight1, v.deform.weight2, v.deform.weight3])
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)

                weighted_bone_indexes = []
                if weights[0] > 0:
                    weighted_bone_indexes.append(v.deform.index0)
                if weights[1] > 0:
                    weighted_bone_indexes.append(v.deform.index1)
                if weights[2] > 0:
                    weighted_bone_indexes.append(v.deform.index2)
                if weights[3] > 0:
                    weighted_bone_indexes.append(v.deform.index3)

                for bi0, bi1 in list(combinations(weighted_bone_indexes, 2)):
                    # ボーン2つずつのペアでウェイト繋がり具合を保持する
                    key = (min(bi0, bi1), max(bi0, bi1))
                    if key not in weighted_bone_pairs:
                        weighted_bone_pairs.append(key)

            if n > 0 and n % 1000 == 0:
                logger.info("-- ウェイトボーン確認: %s個目:終了", n)

        logger.info("%s: 仮想ボーンリストの生成", material_name)
        all_bone_connected = {}
        all_registered_bones = {}
        vertex_maps = {}
        nearest_pos = MVector3D()

        if param_option["physics_type"] in [logger.transtext("布")]:
            vertex_map = np.full((bone_grid_rows + 1, bone_grid_cols, 3), (np.nan, np.nan, np.nan))
            bone_connected = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)
            registered_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

            for grid_col in range(bone_grid_cols):
                for grid_row in range(bone_grid_rows):
                    bone_name = (
                        bone_grid[grid_row][grid_col]
                        if len(bone_grid) > grid_row and len(bone_grid[grid_row]) > grid_col
                        else None
                    )
                    bone = model.bones.get(bone_name, None)
                    if not bone_name or not bone:
                        if grid_row > 0:
                            # 末端ボーンを追加
                            above_bone_name = bone_grid[min(len(bone_grid) - 1, grid_row - 1)][grid_col]
                            above_bone = model.bones.get(above_bone_name, None)
                            above_pos = above_bone.position
                            above_tail_position = (
                                model.bones[model.bone_indexes[above_bone.tail_index]].position
                                if above_bone.tail_index >= 0
                                else above_bone.position + above_bone.tail_position
                            )
                            above_tail_key = above_tail_position.to_key(threshold)

                            above_tail_bone = Bone(
                                f"{above_bone.name}先",
                                f"{above_bone.english_name}_Tail",
                                above_tail_position.copy(),
                                above_bone.index,
                                0,
                                0x0000 | 0x0002 | 0x0008 | 0x0010 | 0x0001,
                            )

                            above_tail_bone.local_x_vector = (
                                above_tail_bone.position - above_bone.position
                            ).normalized()
                            above_tail_bone.local_z_vector = MVector3D.crossProduct(
                                above_tail_bone.local_x_vector, MVector3D(1, 0, 0)
                            )
                            above_tail_bone.index = len(model.bones)
                            model.bones[above_tail_bone.name] = above_tail_bone
                            model.bone_indexes[above_tail_bone.index] = above_tail_bone.name

                            above_tail_key = above_tail_position.to_key(threshold)
                            if above_tail_key not in virtual_vertices:
                                virtual_vertices[above_tail_key] = VirtualVertex(above_tail_key)

                            virtual_vertices[above_tail_key].positions.append(above_tail_position.data())
                            virtual_vertices[above_tail_key].map_bones[0] = above_tail_bone

                            vertex_map[grid_row, grid_col] = above_tail_key
                            registered_bones[grid_row, grid_col] = True

                            above_bone.tail_index = above_tail_bone.index
                            above_bone.flag |= 0x0001 | 0x0008 | 0x0010

                            logger.info("-- 仮想末端ボーン: %s: 追加", above_tail_bone.name)

                            # -----------------------------
                            # 末端非表示ボーン
                            # 直近頂点の上下距離
                            nearest_distance = above_tail_position.distanceToPoint(above_pos)

                            # ボーン進行方向(x)
                            x_direction_pos = (above_pos - above_tail_position).normalized()
                            # ボーン進行方向に対しての横軸(y)
                            y_direction_pos = (
                                MVector3D(0, 0, -1) * x_direction_pos.x()
                                if np.isclose(abs(x_direction_pos.x()), 1)
                                else MVector3D(1, 0, 0)
                            )
                            # ボーン進行方向に対しての縦軸(z)
                            z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
                            above_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)

                            mat = MMatrix4x4()
                            mat.setToIdentity()
                            mat.translate(above_tail_position)
                            mat.rotate(above_qq)

                            # 仮想頂点の位置
                            tail_position = mat * MVector3D(
                                0,
                                -nearest_distance,
                                0,
                            )

                            tail_bone = Bone(
                                f"{above_tail_bone.name}2",
                                f"{above_tail_bone.english_name}2",
                                tail_position.copy(),
                                above_tail_bone.index,
                                0,
                                0x0000 | 0x0002,
                            )

                            tail_bone.local_x_vector = (tail_bone.position - above_tail_bone.position).normalized()
                            tail_bone.local_z_vector = MVector3D.crossProduct(
                                tail_bone.local_x_vector, MVector3D(1, 0, 0)
                            )
                            tail_bone.index = len(model.bones)
                            model.bones[tail_bone.name] = tail_bone
                            model.bone_indexes[tail_bone.index] = tail_bone.name

                            tail_key = tail_position.to_key(threshold)
                            if tail_key not in virtual_vertices:
                                virtual_vertices[tail_key] = VirtualVertex(tail_key)

                            virtual_vertices[tail_key].positions.append(tail_position.data())
                            virtual_vertices[tail_key].map_bones[0] = tail_bone

                            vertex_map[grid_row + 1, grid_col] = tail_key
                            registered_bones[grid_row + 1, grid_col] = True

                            above_tail_bone.tail_index = tail_bone.index
                            above_tail_bone.flag |= 0x0001

                            logger.info("-- 仮想末端ボーン: %s: 追加", tail_bone.name)
                        continue

                    if bone.index not in weighted_vertex_positions:
                        # ウェイトを持たないボーンの場合

                        # ボーン位置を保持
                        nearest_pos = bone.position.copy()
                        nearest_v_key = bone.position.to_key(threshold)
                        # 登録は対象外
                        registered_bones[grid_row, grid_col] = False
                    else:
                        # ウェイトを持つボーンの場合

                        # ウェイト頂点の中心を保持
                        nearest_pos = MVector3D(np.mean(weighted_vertex_positions[bone.index], axis=0))
                        nearest_v_key = nearest_pos.to_key(threshold)
                        # ウェイト頂点の中心頂点に紐付くボーンとして登録
                        registered_bones[grid_row, grid_col] = True

                    if nearest_v_key not in virtual_vertices:
                        virtual_vertices[nearest_v_key] = VirtualVertex(nearest_v_key)

                    virtual_vertices[nearest_v_key].positions.append(nearest_pos.data())
                    virtual_vertices[nearest_v_key].map_bones[0] = bone
                    vertex_map[grid_row, grid_col] = nearest_v_key

                    logger.info("-- 仮想ボーン: %s: 終了", bone.name)

            # 布は横の繋がりをチェックする
            for v_yidx in range(vertex_map.shape[0]):
                v_xidx = 0
                for v_xidx in range(0, vertex_map.shape[1] - 1):
                    if (
                        not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                        and not np.isnan(vertex_map[v_yidx, v_xidx + 1]).any()
                    ):
                        vv1 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])]
                        vv2 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])]

                        vv1_bone_index = vv1.map_bones[0].index if 0 in vv1.map_bones else -1
                        vv2_bone_index = vv2.map_bones[0].index if 0 in vv2.map_bones else -1

                        key = (min(vv1_bone_index, vv2_bone_index), max(vv1_bone_index, vv2_bone_index))
                        if key in weighted_bone_pairs:
                            # ウェイトを共有するボーンの組み合わせであった場合、接続TRUE
                            bone_connected[v_yidx, v_xidx] = True

                if v_xidx > 0:
                    v_xidx += 1
                if not np.isnan(vertex_map[v_yidx, v_xidx]).any() and not np.isnan(vertex_map[v_yidx, 0]).any():
                    # 輪を描いたのも繋がっているかチェック
                    vv1 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])]
                    vv2 = virtual_vertices[tuple(vertex_map[v_yidx, 0])]

                    vv1_bone_index = vv1.map_bones[0].index if 0 in vv1.map_bones else -1
                    vv2_bone_index = vv2.map_bones[0].index if 0 in vv2.map_bones else -1

                    key = (min(vv1_bone_index, vv2_bone_index), max(vv1_bone_index, vv2_bone_index))
                    if key in weighted_bone_pairs:
                        # ウェイトを共有するボーンの組み合わせであった場合、接続TRUE
                        bone_connected[v_yidx, v_xidx] = True

            all_bone_connected[0] = bone_connected
            all_registered_bones[0] = registered_bones
            vertex_maps[len(vertex_maps)] = vertex_map
        else:
            # 布以外は一列ものとして別登録
            for grid_col in range(bone_grid_cols):
                valid_bone_grid_rows = [n for n in range(bone_grid_rows) if bone_grid[n][grid_col]]
                vertex_map = np.full((len(valid_bone_grid_rows), 1, 3), (np.nan, np.nan, np.nan))
                # 横との接続は一切なし
                bone_connected = np.zeros((len(valid_bone_grid_rows), 1), dtype=np.int)
                registered_bones = np.zeros((len(valid_bone_grid_rows), 1), dtype=np.int)

                for grid_row in valid_bone_grid_rows:
                    bone_name = bone_grid[grid_row][grid_col]
                    bone = model.bones.get(bone_name, None)
                    if not bone_name or not bone:
                        continue

                    if bone.index not in weighted_vertex_positions:
                        # ウェイトを持たないボーンの場合

                        # ボーン位置を保持
                        nearest_pos = bone.position.copy()
                        nearest_v_key = bone.position.to_key(threshold)
                        # 登録は対象外
                        registered_bones[grid_row, 0] = False
                    else:
                        # ウェイトを持つボーンの場合

                        # ウェイト頂点の中心を保持
                        nearest_pos = MVector3D(np.mean(weighted_vertex_positions[bone.index], axis=0))
                        nearest_v_key = nearest_pos.to_key(threshold)
                        registered_bones[grid_row, 0] = True

                    if nearest_v_key not in virtual_vertices:
                        virtual_vertices[nearest_v_key] = VirtualVertex(nearest_v_key)

                    virtual_vertices[nearest_v_key].positions.append(nearest_pos.data())
                    virtual_vertices[nearest_v_key].map_bones[grid_col] = bone
                    vertex_map[grid_row, 0] = nearest_v_key

                    logger.info("-- 仮想ボーン: %s: 終了", bone.name)

                all_bone_connected[grid_col] = bone_connected
                all_registered_bones[grid_col] = registered_bones
                vertex_maps[len(vertex_maps)] = vertex_map

        vertex_positions = {}
        top_bone_positions = []
        for vertex_map in vertex_maps.values():
            for v_yidx in range(vertex_map.shape[0]):
                for vkey in vertex_map[v_yidx, :]:
                    if np.isnan(vkey).any() or tuple(vkey) not in virtual_vertices:
                        continue
                    if v_yidx == 0:
                        top_bone_positions.append(virtual_vertices[tuple(vkey)].position().data())
                    vertex_positions[tuple(vkey)] = virtual_vertices[tuple(vkey)].position().data()

        # 中心ボーン
        root_bone = None
        if is_root_bone:
            display_name, root_bone = self.create_root_bone(
                model,
                param_option,
                material_name,
                MVector3D(np.mean(top_bone_positions, axis=0)),
                base_vertical_axis,
                param_option["parent_bone_name"],
            )

            model.bones[root_bone.name] = root_bone
            model.bone_indexes[root_bone.index] = root_bone.name
            model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))
        else:
            root_bone = model.bones[param_option["parent_bone_name"]]

        # 各頂点の位置との差分から距離を測る
        v_distances = np.linalg.norm(
            (np.array(list(vertex_positions.values())) - root_bone.position.data()), ord=2, axis=1
        )
        # 親ボーンに最も近い頂点
        nearest_vertex_idx = list(vertex_positions.keys())[np.argmin(v_distances)]
        # 親ボーンに最も遠い頂点
        farest_vertex_idx = list(vertex_positions.keys())[np.argmax(v_distances)]
        # 材質全体の傾き
        material_direction = (
            MVector3D(vertex_positions[farest_vertex_idx] - vertex_positions[nearest_vertex_idx])
            .abs()
            .normalized()
            .data()[np.where(np.abs(base_vertical_axis.data()))]
        )[0]
        logger.info("%s: 材質頂点の傾き算出: %s", material_name, round(material_direction, 5))
        is_material_horizonal = np.isclose(material_direction, 0.0)

        return (
            virtual_vertices,
            vertex_maps,
            all_registered_bones,
            all_bone_connected,
            root_bone,
            is_material_horizonal,
        )

    def create_vertex_map(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        target_vertices: list,
        base_vertical_axis: MVector3D,
        base_reverse_axis: MVector3D,
        target_idx: int,
    ):
        logger.info("【%s:%s】頂点マップ生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

        # 裏面頂点リスト
        back_vertices = []
        # 残頂点リスト
        remaining_vertices = {}
        axis_idx = np.where(np.abs(base_vertical_axis.data()))[0][0]

        parent_bone = model.bones[param_option["parent_bone_name"]]
        # 親ボーンの傾き
        parent_direction = parent_bone.tail_position.normalized()
        if parent_bone.tail_index >= 0 and parent_bone.tail_index in model.bone_indexes:
            parent_direction = (
                (model.bones[model.bone_indexes[parent_bone.tail_index]].position) - parent_bone.position
            ).normalized()
        if param_option["direction"] in [logger.transtext("上"), logger.transtext("下")]:
            parent_direction = base_vertical_axis
            logger.info("%s: 親ボーンの傾き: %s", material_name, parent_direction.to_log())

        # 一旦全体の位置を把握
        n = 0
        vertex_positions = {}
        for index_idx in model.material_indices[material_name]:
            for v0_idx, v1_idx in zip(
                model.indices[index_idx],
                model.indices[index_idx][1:] + [model.indices[index_idx][0]],
            ):
                if v0_idx not in target_vertices or v1_idx not in target_vertices:
                    continue
                vertex_positions[v0_idx] = model.vertex_dict[v0_idx].position.data()

            n += 1
            if n > 0 and n % 1000 == 0:
                logger.info("-- 全体メッシュ確認: %s個目:終了", n)

        if not vertex_positions:
            logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None, None, None, None, None, None, None

        material_mean_pos = MVector3D(np.mean(list(vertex_positions.values()), axis=0))
        logger.info("%s: 材質頂点の中心点算出: %s", material_name, material_mean_pos.to_log())

        # 各頂点の位置との差分から距離を測る
        v_distances = np.linalg.norm(
            (np.array(list(vertex_positions.values())) - parent_bone.position.data()), ord=2, axis=1
        )
        # 親ボーンに最も近い頂点
        nearest_vertex_idx = list(vertex_positions.keys())[np.argmin(v_distances)]
        # 親ボーンに最も遠い頂点
        farest_vertex_idx = list(vertex_positions.keys())[np.argmax(v_distances)]
        # 材質全体の傾き
        material_direction = (
            (model.vertex_dict[farest_vertex_idx].position - model.vertex_dict[nearest_vertex_idx].position)
            .abs()
            .normalized()
            .data()[np.where(np.abs(base_vertical_axis.data()))]
        )[0]
        logger.info("%s: 材質頂点の傾き算出: %s", material_name, round(material_direction, 5))
        is_material_horizonal = material_direction < 0.1

        # 頂点間の距離
        # https://blog.shikoan.com/distance-without-for-loop/
        all_vertex_diffs = np.expand_dims(np.array(list(vertex_positions.values())), axis=1) - np.expand_dims(
            np.array(list(vertex_positions.values())), axis=0
        )
        all_vertex_distances = np.sqrt(np.sum(all_vertex_diffs**2, axis=-1))
        # 頂点同士の距離から閾値生成
        threshold = np.min(all_vertex_distances[all_vertex_distances > 0]) * 0.5

        logger.info("%s: 材質頂点の閾値算出: %s", material_name, round(threshold, 5))

        logger.info("%s: 仮想頂点リストの生成", material_name)

        virtual_vertices = {}
        index_surface_normals = {}
        edge_pair_lkeys = {}
        all_line_pairs = {}
        all_indexes_vertices = []
        all_same_pair_edges = {}
        n = 0
        for index_idx in model.material_indices[material_name]:
            # 頂点の組み合わせから面INDEXを引く
            if (
                model.indices[index_idx][0] not in target_vertices
                or model.indices[index_idx][1] not in target_vertices
                or model.indices[index_idx][2] not in target_vertices
            ):
                # 3つ揃ってない場合、スルー
                continue

            v0_idx = model.indices[index_idx][0]
            v1_idx = model.indices[index_idx][1]
            v2_idx = model.indices[index_idx][2]

            v0 = model.vertex_dict[v0_idx]
            v1 = model.vertex_dict[v1_idx]
            v2 = model.vertex_dict[v2_idx]

            v0_key = v0.position.to_key(threshold)
            v1_key = v1.position.to_key(threshold)
            v2_key = v2.position.to_key(threshold)

            for (l1, l2) in combinations([v0_key, v1_key, v2_key], 2):
                if l1 not in all_line_pairs:
                    all_line_pairs[l1] = []
                if l2 not in all_line_pairs[l1]:
                    all_line_pairs[l1].append(l2)

                if l2 not in all_line_pairs:
                    all_line_pairs[l2] = []
                if l1 not in all_line_pairs[l2]:
                    all_line_pairs[l2].append(l1)
            all_indexes_vertices.append(tuple(list(sorted([v0_key, v1_key, v2_key]))))

            # 一旦ルートボーンにウェイトを一括置換
            v0.deform = Bdef1(parent_bone.index)

            # 面垂線
            vv1 = v1.position - v0.position
            vv2 = v2.position - v1.position
            surface_vector = MVector3D.crossProduct(vv1, vv2)
            surface_normal = surface_vector.normalized()

            # 面の中心
            mean_pos = MVector3D(np.mean([v0.position.data(), v1.position.data(), v2.position.data()], axis=0))
            # 面垂線と軸ベクトルとの内積
            direction_dot = MVector3D.dotProduct(surface_normal, parent_direction)

            if np.isclose(material_direction, 0):
                # 水平の場合、軸の向きだけ考える
                intersect = surface_normal.data()[target_idx]
            else:
                # ボーンから面中心への向き（評価軸を殺して垂直にする）を親の面法線とする
                if param_option["direction"] in [logger.transtext("上"), logger.transtext("下")]:
                    material_normal = (mean_pos - parent_bone.position) * base_reverse_axis
                else:
                    material_normal = MVector3D.crossProduct(mean_pos, mean_pos + base_vertical_axis)

                # 面中心を面法線方向に伸ばした垂線を線分とみなす
                mean_line_segment = mean_pos + surface_normal
                surface_line_segment = mean_pos + surface_normal * 1000

                v0_dot = MVector3D.dotProduct(material_normal, mean_line_segment)
                surface_dot = MVector3D.dotProduct(material_normal, surface_line_segment)

                logger.test(
                    f"material_normal[{material_normal.to_log()}], mean_line_segment[{mean_line_segment.to_log()}], surface_line_segment[{surface_line_segment.to_log()}], v0_dot[{round(v0_dot, 3)}], surface_dot[{round(surface_dot, 3)}]"
                )
                # (v1・n) * (v2・n) <= 0ならば線分は平面と衝突を起こしている
                intersect = v0_dot * surface_dot

            # 面法線と同じ向き場合かつ面垂線の向きが軸ベクトルに近くない場合、辺キー生成（表面のみを対象とする）
            # プリーツは厚みがないとみなす
            if (intersect > 0 and direction_dot < 0.5) or param_option["special_shape"] == logger.transtext("全て表面"):
                for vv0_key, vv1_key, vv2_key, vv0 in [
                    (v0_key, v1_key, v2_key, v0),
                    (v1_key, v2_key, v0_key, v1),
                    (v2_key, v0_key, v1_key, v2),
                ]:
                    lkey = (min(vv0_key, vv1_key), max(vv0_key, vv1_key))
                    if lkey not in edge_pair_lkeys:
                        edge_pair_lkeys[lkey] = []
                        all_same_pair_edges[lkey] = []
                    if index_idx not in edge_pair_lkeys[lkey]:
                        edge_pair_lkeys[lkey].append(index_idx)
                    all_same_pair_edges[lkey].append(vv2_key)

                    if lkey not in index_surface_normals:
                        index_surface_normals[lkey] = []
                    index_surface_normals[lkey].append(surface_normal)

                    # 仮想頂点登録（該当頂点対象）
                    if vv0_key not in virtual_vertices:
                        virtual_vertices[vv0_key] = VirtualVertex(vv0_key)
                    virtual_vertices[vv0_key].append([vv0], [vv1_key, vv2_key], [index_idx])

                    # 残頂点リストにまずは登録
                    if vv0_key not in remaining_vertices:
                        remaining_vertices[vv0_key] = virtual_vertices[vv0_key]

                logger.test(
                    f"☆表 index[{index_idx}], v0[{v0.index}:{v0_key}], v1[{v1.index}:{v1_key}], v2[{v2.index}:{v2_key}], i[{round(intersect, 5)}], dot[{round(direction_dot, 4)}], mn[{mean_pos.to_log()}], sn[{surface_normal.to_log()}], pa[{parent_bone.position.to_log()}]"
                )
            else:
                # 裏面に登録
                back_vertices.append(v0_idx)
                back_vertices.append(v1_idx)
                back_vertices.append(v2_idx)
                logger.test(
                    f"★裏 index[{index_idx}], v0[{v0.index}:{v0_key}], v1[{v1.index}:{v1_key}], v2[{v2.index}:{v2_key}], i[{round(intersect, 5)}], dot[{round(direction_dot, 4)}], mn[{mean_pos.to_log()}], sn[{surface_normal.to_log()}], pa[{parent_bone.position.to_log()}]"
                )

            n += 1

            if n > 0 and n % 100 == 0:
                logger.info("-- メッシュ確認: %s個目:終了", n)

        if not virtual_vertices:
            logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None, None, None, None, None, None, None

        if not edge_pair_lkeys:
            logger.warning("対象範囲にエッジが見つけられなかった為、処理を終了します。\n面が表裏反転してないかご確認ください。", decoration=MLogger.DECORATION_BOX)
            return None, None, None, None, None, None, None

        logger.test("--------------------------")

        edge_line_pairs = {}
        for n, ((min_vkey, max_vkey), line_iidxs) in enumerate(edge_pair_lkeys.items()):
            surface_normals = index_surface_normals[(min_vkey, max_vkey)]
            is_pair = False
            if len(line_iidxs) == 1:
                is_pair = True

                logger.test(
                    f"○ min_vkey: [{min_vkey}({virtual_vertices[min_vkey].vidxs()})], max_vkey: [{max_vkey}({virtual_vertices[max_vkey].vidxs()})], line_iidxs: [{line_iidxs}], only-one-index"
                )
            else:
                surface_dots = [MVector3D.dotProduct(surface_normals[0], sn) for sn in surface_normals]
                surface_dot_diffs = np.abs(np.array(surface_dots) - surface_dots[0])
                if np.where(surface_dot_diffs < 0.2)[0].shape[0] == 1:
                    logger.test(
                        f"△ min_vkey: [{min_vkey}({virtual_vertices[min_vkey].vidxs()})], max_vkey: [{max_vkey}({virtual_vertices[max_vkey].vidxs()})], line_iidxs: [{line_iidxs}], surface_dots[{[round(sd, 4) for sd in surface_dots]}], surface_diffs[{[round(sd, 4) for sd in surface_dot_diffs]}], surface-one-index"
                    )
                else:
                    logger.test(
                        f"× min_vkey: [{min_vkey}({virtual_vertices[min_vkey].vidxs()})], max_vkey: [{max_vkey}({virtual_vertices[max_vkey].vidxs()})], line_iidxs: [{line_iidxs}], surface_dots[{[round(sd, 4) for sd in surface_dots]}], surface_diffs[{[round(sd, 4) for sd in surface_dot_diffs]}], multi-index"
                    )

            if is_pair:
                if min_vkey not in virtual_vertices or max_vkey not in virtual_vertices:
                    continue

                if min_vkey not in edge_line_pairs:
                    edge_line_pairs[min_vkey] = []
                if max_vkey not in edge_line_pairs:
                    edge_line_pairs[max_vkey] = []

                edge_line_pairs[min_vkey].append(max_vkey)
                edge_line_pairs[max_vkey].append(min_vkey)

            if n > 0 and n % 200 == 0:
                logger.info("-- 辺確認: %s個目:終了", n)

        if logger.is_debug_level():
            logger.debug("--------------------------")
            logger.debug("エッジペアリスト")
            for v_key, pair_vkeys in edge_line_pairs.items():
                logger.debug(
                    f"key[{v_key}:{virtual_vertices[v_key].vidxs()}], pair[{pair_vkeys}:{[virtual_vertices[pair_vkey].vidxs() for pair_vkey in pair_vkeys]}]"
                )

        if param_option["special_shape"] == logger.transtext("面欠け"):
            logger.debug("仮想面の確認")

            virtual_edges = []
            for n, (v0_key, pair_vertices) in enumerate(all_line_pairs.items()):
                for (v1_key, v2_key) in combinations(pair_vertices, 2):
                    if not (v0_key in edge_line_pairs and v1_key in edge_line_pairs and v2_key in edge_line_pairs):
                        # 全部エッジになければスルー
                        continue

                    ikey = tuple(list(sorted([v0_key, v1_key, v2_key])))
                    if ikey in all_indexes_vertices:
                        # 既に面のエッジの組み合わせである場合、スルー
                        continue

                    edge_existed = {}
                    for (vv0_key, vv1_key) in combinations(set([v0_key, v1_key, v2_key]), 2):
                        lkey = (min(vv0_key, vv1_key), max(vv0_key, vv1_key))
                        edge_existed[lkey] = lkey in all_same_pair_edges

                    if 1 < Counter(edge_existed.values())[False]:
                        # 仮想エッジが1件より多い場合スルー
                        logger.debug(
                            "** ×仮想エッジスルー: [%s:%s, %s:%s, %s:%s][%s]",
                            v0_key,
                            virtual_vertices[v0_key].vidxs(),
                            v1_key,
                            virtual_vertices[v1_key].vidxs(),
                            v2_key,
                            virtual_vertices[v2_key].vidxs(),
                            edge_existed,
                        )
                        continue

                    logger.debug("----------------------")
                    logger.debug(
                        "** ○仮想エッジ1件: [%s:%s, %s:%s, %s:%s][%s]",
                        v0_key,
                        virtual_vertices[v0_key].vidxs(),
                        v1_key,
                        virtual_vertices[v1_key].vidxs(),
                        v2_key,
                        virtual_vertices[v2_key].vidxs(),
                        edge_existed,
                    )

                    # 仮想エッジ
                    virtual_edge_keys = [e for e, v in edge_existed.items() if v == False]
                    if not virtual_edge_keys:
                        continue
                    virtual_edge_keys = virtual_edge_keys[0]
                    ve1_vec = virtual_vertices[virtual_edge_keys[0]].position()
                    ve2_vec = virtual_vertices[virtual_edge_keys[1]].position()
                    ve_line = MSegment(ve1_vec, ve2_vec)

                    is_both_connected = False
                    for vkey in [v0_key, v1_key, v2_key]:
                        # 仮想エッジの両方に繋がっている場合、内積が大きく違う事（同じ辺でないこと）をチェック
                        dot = MVector3D.dotProduct(
                            (virtual_vertices[vkey].position() - ve1_vec).normalized(),
                            (virtual_vertices[vkey].position() - ve2_vec).normalized(),
                        )
                        if abs(dot) > 0.8:
                            is_both_connected = True
                            logger.debug(
                                "** ×同一辺重複: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][p: [%s:%s]][%s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vkey,
                                virtual_vertices[vkey].vidxs(),
                                dot,
                            )
                            break
                        else:
                            logger.debug(
                                "** ○同一辺重複なし: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][p: [%s:%s]][%s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vkey,
                                virtual_vertices[vkey].vidxs(),
                                dot,
                            )

                    if is_both_connected:
                        continue

                    area_threshold = (
                        np.mean(
                            [
                                virtual_vertices[v0_key]
                                .position()
                                .distanceToPoint(virtual_vertices[v1_key].position()),
                                virtual_vertices[v0_key]
                                .position()
                                .distanceToPoint(virtual_vertices[v2_key].position()),
                                virtual_vertices[v1_key]
                                .position()
                                .distanceToPoint(virtual_vertices[v2_key].position()),
                            ]
                        )
                        * 0.3
                    )

                    # 頂点の組合せの中に他の頂点が含まれているか
                    is_inner_vkey = False
                    for vvkey in edge_line_pairs.keys():
                        if not (
                            vvkey in virtual_vertices[virtual_edge_keys[0]].connected_vvs
                            and vvkey in virtual_vertices[virtual_edge_keys[1]].connected_vvs
                        ):
                            # エッジキーが接続先にない場合、スルー
                            continue

                        # エッジキーが仮想エッジの両方から繋がっている場合、距離チェック
                        min_length, h, t = calc_point_segment_dist(MPoint(virtual_vertices[vvkey].position()), ve_line)
                        if 0 < min_length < area_threshold:
                            is_inner_vkey = True
                            logger.debug(
                                "** ×頂点内頂点重複: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][p: [%s:%s]][%s < %s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vvkey,
                                virtual_vertices[vvkey].vidxs(),
                                min_length,
                                area_threshold,
                            )
                            break
                        else:
                            logger.debug(
                                "** 頂点重複なし: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][p: [%s:%s]][%s < %s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vvkey,
                                virtual_vertices[vvkey].vidxs(),
                                min_length,
                                area_threshold,
                            )

                    if is_inner_vkey:
                        continue

                    relation_vkeys = [v0_key, v1_key, v2_key]
                    for vk in [v0_key, v1_key, v2_key]:
                        relation_vkeys.extend(virtual_vertices[vk].connected_vvs)
                    relation_vkeys = set(relation_vkeys)

                    is_intersect = False
                    for (vv0_key, vv1_key) in all_same_pair_edges.keys():
                        if not (vv0_key in relation_vkeys and vv1_key in relation_vkeys):
                            # 関係する頂点でない場合スルー
                            logger.test(
                                "** ×交差無関係スルー: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][vv: [%s:%s, %s:%s]]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vv0_key,
                                virtual_vertices[vv0_key].vidxs(),
                                vv1_key,
                                virtual_vertices[vv1_key].vidxs(),
                            )
                            continue

                        if len({vv0_key, vv1_key} - {v0_key, v1_key, v2_key}) == 0:
                            # 全部仮想面の頂点の場合、見なくていいのでスルー
                            logger.debug(
                                "** ×仮想面スルー: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][vv: [%s:%s, %s:%s]]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vv0_key,
                                virtual_vertices[vv0_key].vidxs(),
                                vv1_key,
                                virtual_vertices[vv1_key].vidxs(),
                            )
                            continue

                        if len({vv0_key, vv1_key} - {virtual_edge_keys[0], virtual_edge_keys[1]}) == 1:
                            # 同じ頂点が残っている場合、交差は見なくていいのでスルー
                            logger.debug(
                                "** ×交差交点スルー: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][vv: [%s:%s, %s:%s]]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vv0_key,
                                virtual_vertices[vv0_key].vidxs(),
                                vv1_key,
                                virtual_vertices[vv1_key].vidxs(),
                            )
                            continue

                        vv1_vec = virtual_vertices[vv0_key].position()
                        vv2_vec = virtual_vertices[vv1_key].position()
                        vv_line = MSegment(vv1_vec, vv2_vec)
                        min_length, p1, p2, t1, t2 = calc_segment_segment_dist(vv_line, ve_line)
                        if 0 < min_length < area_threshold * 1.5 and t1 and t2:
                            # 線分があって、閾値より小さい場合、交差していると見なす
                            is_intersect = True
                            logger.debug(
                                "** ×交差あり: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][vv: [%s:%s, %s:%s]][%s < %s, %s, %s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vv0_key,
                                virtual_vertices[vv0_key].vidxs(),
                                vv1_key,
                                virtual_vertices[vv1_key].vidxs(),
                                min_length,
                                area_threshold,
                                round(t1, 5),
                                round(t2, 5),
                            )
                            break
                        else:
                            logger.debug(
                                "** ○交差なし: [%s:%s, %s:%s, %s:%s][ve: [%s:%s, %s:%s]][vv: [%s:%s, %s:%s]][%s < %s, %s, %s]",
                                v0_key,
                                virtual_vertices[v0_key].vidxs(),
                                v1_key,
                                virtual_vertices[v1_key].vidxs(),
                                v2_key,
                                virtual_vertices[v2_key].vidxs(),
                                virtual_edge_keys[0],
                                virtual_vertices[virtual_edge_keys[0]].vidxs(),
                                virtual_edge_keys[1],
                                virtual_vertices[virtual_edge_keys[1]].vidxs(),
                                vv0_key,
                                virtual_vertices[vv0_key].vidxs(),
                                vv1_key,
                                virtual_vertices[vv1_key].vidxs(),
                                min_length,
                                area_threshold,
                                round(t1, 5),
                                round(t2, 5),
                            )

                    if is_intersect:
                        # 既存のエッジと交差していたらスルー
                        continue

                    logger.debug(
                        "* 仮想面: [%s:%s, %s:%s, %s:%s]",
                        v0_key,
                        virtual_vertices[v0_key].vidxs(),
                        v1_key,
                        virtual_vertices[v1_key].vidxs(),
                        v2_key,
                        virtual_vertices[v2_key].vidxs(),
                    )
                    logger.info(
                        "-- %s: 仮想面追加: [%s, %s, %s]",
                        material_name,
                        virtual_vertices[v0_key].vidxs(),
                        virtual_vertices[v1_key].vidxs(),
                        virtual_vertices[v2_key].vidxs(),
                    )

                    # 頂点を繋げた辺がエッジペアに含まれてない場合、仮想面を追加する
                    for (vv0_key, vv1_key) in combinations(set([v0_key, v1_key, v2_key]), 2):
                        lkey = (min(vv0_key, vv1_key), max(vv0_key, vv1_key))

                        if vv0_key in virtual_edge_keys and vv1_key in virtual_edge_keys and lkey not in virtual_edges:
                            # 仮想エッジの場合、追加
                            if vv0_key not in edge_line_pairs:
                                edge_line_pairs[vv0_key] = []
                            if vv1_key not in edge_line_pairs:
                                edge_line_pairs[vv1_key] = []

                            edge_line_pairs[vv0_key].append(vv1_key)
                            edge_line_pairs[vv1_key].append(vv0_key)

                            virtual_edges.append(lkey)
                        else:
                            # 仮想エッジではない場合、エッジから除外
                            for vvv0_key, vvv1_key in [(vv0_key, vv1_key), (vv1_key, vv0_key)]:
                                if vvv0_key in edge_line_pairs:
                                    for n, vk in enumerate(edge_line_pairs[vvv0_key]):
                                        if vk == vvv1_key:
                                            del edge_line_pairs[vvv0_key][n]
                                            break
                                    if not edge_line_pairs[vvv0_key]:
                                        del edge_line_pairs[vvv0_key]

                    # 接続先も追加
                    for vvkey in [v0_key, v1_key, v2_key]:
                        for rvvkey in set([v0_key, v1_key, v2_key]) - {vvkey}:
                            if rvvkey not in virtual_vertices[vvkey].connected_vvs:
                                virtual_vertices[vvkey].connected_vvs.append(rvvkey)

                if n > 0 and n % 20 == 0:
                    logger.info("-- 仮想面確認: %s個目:終了", n)

            logger.debug(
                "仮想エッジ: %s",
                ", ".join(
                    [
                        "[{0}:{1}, {2}:{3}]".format(
                            vv0_key,
                            virtual_vertices[vv0_key].vidxs(),
                            vv1_key,
                            virtual_vertices[vv1_key].vidxs(),
                        )
                        for vv0_key, vv1_key in virtual_edges
                    ]
                ),
            )

        logger.info("%s: エッジの抽出準備", material_name)

        # エッジを繋いでいく
        tmp_edge_lines = []
        edge_vkeys = []
        n = 0
        remain_start_vkeys = list(edge_line_pairs.keys())
        remain_existed_edges = list(edge_line_pairs.keys())
        while remain_existed_edges:
            registered_edge_line_pairs = {}
            _, tmp_edge_lines, edge_vkeys, registered_edge_line_pairs = self.get_edge_lines(
                edge_line_pairs,
                registered_edge_line_pairs,
                None,
                tmp_edge_lines,
                edge_vkeys,
                virtual_vertices,
                param_option,
                0,
                n,
            )
            remain_start_vkeys = [elp for elp in edge_line_pairs.keys() if edge_line_pairs[elp]]
            remain_existed_edges = set([vk for vk in remain_start_vkeys if vk in edge_line_pairs])
            n += 1
            logger.info("-- エッジ検出: %s個目:終了", n)

        all_edge_lines = []
        for n, edge_lines in enumerate(tmp_edge_lines):
            if 0 < len(edge_lines) < 3:
                logger.info(
                    "-- %s: 検出エッジ（件数が少ないため対象外）: %s",
                    material_name,
                    [f"{ekey}:{virtual_vertices[ekey].vidxs()}" for ekey in edge_lines],
                )
            elif edge_lines:
                logger.info(
                    "-- %s: 検出エッジ: %s",
                    material_name,
                    [f"{ekey}:{virtual_vertices[ekey].vidxs()}" for ekey in edge_lines],
                )
                all_edge_lines.append(edge_lines)

        if not all_edge_lines:
            logger.warning(
                "エッジが検出できなかったため、処理を中断します。\n裏面のある材質で「すべて表面」を選択してないか、確認してください。", decoration=MLogger.DECORATION_BOX
            )
            return None, None, None, None, None, None, None

        logger.info("%s: エッジの抽出", material_name)

        # topは全部並列
        horizonal_top_edge_keys = []
        if is_material_horizonal:
            # 並列の場合、近い方をルートとする
            all_root_key_poses = np.array([virtual_vertices[el[0]].position().data() for el in all_edge_lines])
            all_root_key_idx = np.argmin(np.linalg.norm(all_root_key_poses - material_mean_pos.data(), ord=2, axis=1))
            all_root_key = all_edge_lines[all_root_key_idx][0]
        else:
            # 始点をルートとする（下方向の場合、センターの真後ろ側）
            all_root_key = all_edge_lines[0][0]

        # 根元頂点CSVが指定されている場合、対象頂点リスト生成
        if param_option["top_vertices_csv"]:
            top_target_vertices = read_vertices_from_file(
                param_option["top_vertices_csv"], model, None if param_option["top_vertices_csv"] else material_name
            )
            if not top_target_vertices:
                logger.warning("根元頂点CSVが正常に読み込めなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
                return None, None, None, None, None, None, None

            # 根元頂点キーリスト生成
            top_edge_keys = {}
            for vidx in top_target_vertices:
                v = model.vertex_dict[vidx]

                if v.index not in target_vertices:
                    continue

                # 仮想頂点登録（該当頂点対象）
                vkey = v.position.to_key(threshold)
                if vkey not in virtual_vertices:
                    virtual_vertices[vkey] = VirtualVertex(vkey)
                virtual_vertices[vkey].append([v], [], [])

                top_edge_keys[vkey] = vkey

            horizonal_top_edge_keys, _ = self.get_edge_lines_from_csv(
                all_line_pairs,
                top_edge_keys,
                None,
                [],
                virtual_vertices,
                param_option,
                0,
            )

            if not horizonal_top_edge_keys:
                logger.warning(
                    "根元頂点CSVに指定された頂点が物理材質もしくは物理頂点に含まれていなかった為、処理を終了します。\n物理対象頂点CSVは根元頂点も含めて指定してください。",
                    decoration=MLogger.DECORATION_BOX,
                )
                return None, None, None, None, None, None, None
        else:
            # 根元頂点CSVが指定されていない場合
            # 最もキーが上の一点から大体水平に繋がっている辺を上部エッジとする
            # 既存のラインと逆方向のラインを試す
            for target_edge_lines, is_reverse in [
                (all_edge_lines, False),
                ([list(reversed(ael)) for ael in all_edge_lines], True),
            ]:
                for n, edge_lines in enumerate(target_edge_lines):
                    if all_root_key not in edge_lines:
                        # そもそもルートキーが含まれてなければスルー
                        continue

                    root_key_idx = [n for n, el in enumerate(edge_lines) if el == all_root_key][0]

                    if edge_lines[0] == edge_lines[-1]:
                        del edge_lines[-1]

                    for m, (prev_edge_key, now_edge_key, next_edge_key) in enumerate(
                        zip(
                            edge_lines[(root_key_idx - 1) :] + edge_lines[: (root_key_idx - 1)],
                            edge_lines[(root_key_idx):] + edge_lines[:(root_key_idx)],
                            edge_lines[(root_key_idx + 1) :] + edge_lines[: (root_key_idx + 1)],
                        )
                    ):
                        if now_edge_key in horizonal_top_edge_keys:
                            continue

                        if m == 0:
                            if is_reverse:
                                # 反転は0番目は既に入ってるので now のみ
                                horizonal_top_edge_keys.insert(0, now_edge_key)
                            else:
                                horizonal_top_edge_keys.append(prev_edge_key)
                                horizonal_top_edge_keys.append(now_edge_key)
                        elif m > 0:
                            if is_reverse:
                                horizonal_top_edge_keys.insert(0, now_edge_key)
                            else:
                                horizonal_top_edge_keys.append(now_edge_key)

                        prev_edge_pos = virtual_vertices[prev_edge_key].position()
                        now_edge_pos = virtual_vertices[now_edge_key].position()
                        next_edge_pos = virtual_vertices[next_edge_key].position()

                        # エッジの内積
                        dot = MVector3D.dotProduct(
                            (next_edge_pos - now_edge_pos).normalized(),
                            (now_edge_pos - prev_edge_pos).normalized(),
                        )

                        # エッジの評価軸の変動
                        distance = abs(
                            abs(next_edge_pos.data()[target_idx] - now_edge_pos.data()[target_idx])
                            - abs(now_edge_pos.data()[target_idx] - prev_edge_pos.data()[target_idx])
                        )

                        logger.debug(
                            "now: %s, dot: %s, distance: %s, threshold: %s",
                            virtual_vertices[now_edge_key].vidxs(),
                            round(dot, 3),
                            round(distance, 3),
                            round(threshold, 3),
                        )

                        if dot < 0.5 and (is_material_horizonal or distance > threshold):
                            break

            logger.debug(f"horizonal[{horizonal_top_edge_keys}]")

            if not horizonal_top_edge_keys:
                logger.warning(
                    "物理方向に対して上部エッジが見つけられなかった為、処理を終了します。\nVRoid製スカートの場合、上部のベルト部分が含まれていないかご確認ください。",
                    decoration=MLogger.DECORATION_BOX,
                )
                return None, None, None, None, None, None, None

        logger.info(
            "-- %s: 上部エッジ %s",
            material_name,
            [(ek, virtual_vertices[ek].vidxs()) for ek in horizonal_top_edge_keys],
        )

        top_edge_poses = []
        for key in horizonal_top_edge_keys:
            top_edge_poses.append(virtual_vertices[key].position().data())

        # bottomはエッジが分かれてたら分ける
        all_bottom_edge_keys = []

        # 頂点マップ上部の位置
        top_vv_poses = np.array([virtual_vertices[tk].position().data() for tk in horizonal_top_edge_keys])

        # 離れた評価軸
        axis_root_val = base_vertical_axis.data()[axis_idx] * 100

        for n, edge_lines in enumerate(all_edge_lines):
            bottom_edge_lines = [el for el in edge_lines if el not in horizonal_top_edge_keys]

            if not bottom_edge_lines:
                # 全部上部エッジの場合、スルー
                continue

            if bottom_edge_lines[0] == bottom_edge_lines[-1]:
                del bottom_edge_lines[-1]

            target_bottom_lines = []
            for v_key in bottom_edge_lines:
                vv = virtual_vertices[v_key]

                # 仮想頂点の評価軸位置
                vv_axis_val = vv.position().data()[axis_idx]

                # 頂点マップ上部との距離
                top_distances = np.linalg.norm((top_vv_poses - vv.position().data()), ord=2, axis=1)
                nearest_top_vidxs = virtual_vertices[horizonal_top_edge_keys[np.argmin(top_distances)]].vidxs()
                nearest_top_pos = top_vv_poses[np.argmin(top_distances)]
                # 頂点マップ上部直近頂点の評価軸位置
                nearest_top_axis_val = nearest_top_pos[axis_idx]

                if (axis_root_val - nearest_top_axis_val) <= (axis_root_val - vv_axis_val):
                    # 評価軸にTOPより遠い（スカートの場合、TOPより下）の場合、対象
                    logger.debug(
                        f"○下部エッジ対象: target [{vv.vidxs()}], axis_root_val [{round(axis_root_val, 3)}], nearest_top_vidxs[{nearest_top_vidxs}], nearest_top_axis_val[{round(nearest_top_axis_val, 3)}], vv[{vv.vidxs()}], vv_axis_val[{round(vv_axis_val, 3)}]"
                    )
                    target_bottom_lines.append(v_key)

            if target_bottom_lines:
                all_bottom_edge_keys.append(target_bottom_lines)

        if not all_bottom_edge_keys:
            logger.warning(
                "物理方向に対して下部エッジが見つけられなかった為、処理を終了します。\n根元頂点CSVで物理の開始位置を指定すると下部エッジが見つけられるようになります。",
                decoration=MLogger.DECORATION_BOX,
            )
            return None, None, None, None, None, None, None

        for ei, eks in enumerate(all_bottom_edge_keys):
            logger.info(
                "-- %s: 下部エッジ[%s] %s",
                material_name,
                f"{(ei + 1):02d}",
                [(ek, virtual_vertices[ek].vidxs()) for ek in eks],
            )

        top_edge_mean_pos = MVector3D(np.mean(top_edge_poses, axis=0))
        top_edge_start_pos = top_edge_mean_pos + MVector3D(0, 0, 20)

        top_degrees = {}
        for top_pos in top_edge_poses:
            top_degree0, top_degree1 = self.calc_arc_degree(
                top_edge_start_pos,
                top_edge_mean_pos,
                MVector3D(top_pos),
                base_vertical_axis,
                base_reverse_axis,
            )
            top_degrees[top_degree0] = MVector3D(top_pos)
            top_degrees[top_degree1] = MVector3D(top_pos)
            logger.debug(
                "TOP vidx[%s], pos[%s], degrees[%s, %s]",
                virtual_vertices[MVector3D(top_pos).to_key(threshold)].vidxs(),
                MVector3D(top_pos).to_log(),
                round(top_degree0, 3),
                round(top_degree1, 3),
            )

        bottom_edge_poses = []
        for bottom_edge_keys in all_bottom_edge_keys:
            for key in bottom_edge_keys:
                bottom_edge_poses.append(virtual_vertices[key].position().data())

        # 下端の中央は上部に合わせる
        bottom_edge_mean_pos = top_edge_mean_pos.copy()
        bottom_edge_start_pos = bottom_edge_mean_pos + MVector3D(0, 0, 20)

        # できるだけ評価軸は離して登録する
        if param_option["direction"] in [logger.transtext("下")]:
            top_edge_mean_pos.setY(np.max(top_edge_poses, axis=0)[1])
            bottom_edge_mean_pos.setY(np.min(bottom_edge_poses, axis=0)[1])
        elif param_option["direction"] in [logger.transtext("上")]:
            top_edge_mean_pos.setY(np.min(top_edge_poses, axis=0)[1])
            bottom_edge_mean_pos.setY(np.max(bottom_edge_poses, axis=0)[1])
        elif param_option["direction"] in [logger.transtext("左")]:
            top_edge_mean_pos.setX(np.min(top_edge_poses, axis=0)[0])
            bottom_edge_mean_pos.setX(np.max(bottom_edge_poses, axis=0)[0])
        elif param_option["direction"] in [logger.transtext("右")]:
            top_edge_mean_pos.setX(np.max(top_edge_poses, axis=0)[0])
            bottom_edge_mean_pos.setX(np.min(bottom_edge_poses, axis=0)[0])

        if param_option["direction"] in [logger.transtext("上"), logger.transtext("下")]:
            top_x_radius = np.max(np.abs(np.array(top_edge_poses)[:, 0] - top_edge_mean_pos.data()[0]))
            top_y_radius = 1
            top_z_radius = np.max(np.abs(np.array(top_edge_poses)[:, 2] - top_edge_mean_pos.data()[2]))
            bottom_x_radius = np.max(np.abs(np.array(bottom_edge_poses)[:, 0] - bottom_edge_mean_pos.data()[0]))
            bottom_y_radius = 1
            bottom_z_radius = np.max(np.abs(np.array(bottom_edge_poses)[:, 2] - bottom_edge_mean_pos.data()[2]))
        else:
            top_x_radius = 1
            top_y_radius = np.max(np.abs(np.array(top_edge_poses)[:, 1] - top_edge_mean_pos.data()[1]))
            top_z_radius = np.max(np.abs(np.array(top_edge_poses)[:, 2] - top_edge_mean_pos.data()[2]))
            bottom_x_radius = 1
            bottom_y_radius = np.max(np.abs(np.array(bottom_edge_poses)[:, 1] - bottom_edge_mean_pos.data()[1]))
            bottom_z_radius = np.max(np.abs(np.array(bottom_edge_poses)[:, 2] - bottom_edge_mean_pos.data()[2]))

        # 評価軸の平均値
        top_mean_val = np.mean(np.array(top_edge_poses)[:, target_idx])
        bottom_mean_val = np.mean(np.array(bottom_edge_poses)[:, target_idx])

        logger.info(
            "%s: エッジ上部 中央位置[%s] 半径[x: %s, y: %s, z: %s]",
            material_name,
            top_edge_mean_pos.to_log(),
            round(top_x_radius, 4),
            round(top_y_radius, 4),
            round(top_z_radius, 4),
        )
        logger.info(
            "%s: エッジ下部 中央位置[%s] 半径[x: %s, y: %s, z: %s]",
            material_name,
            bottom_edge_mean_pos.to_log(),
            round(bottom_x_radius, 4),
            round(bottom_y_radius, 4),
            round(bottom_z_radius, 4),
        )

        registered_vkeys = []
        vertex_maps = {}
        logger.info("-----------")

        for bi, bottom_edge_keys in enumerate(all_bottom_edge_keys):
            all_vkeys = []

            if param_option["route_estimate_type"] == logger.transtext("リング"):
                # リング取得
                all_rings = []

                # TOPキーの中央値を閾値とする
                hky = np.median([hkey[1] for hkey in horizonal_top_edge_keys])

                remaining_vkeys = dict(
                    list(
                        sorted(
                            [(k, k) for k in virtual_vertices.keys() if k not in horizonal_top_edge_keys],
                            key=lambda x: (abs(x[0][0]), -x[0][2], -x[0][1]),
                        )
                    )
                )
                # 前のキーとして上のリングを選ぶ
                prev_rings = horizonal_top_edge_keys
                all_rings.append(prev_rings)

                remain_cnt = len(virtual_vertices) * 2
                while remaining_vkeys and remain_cnt > 0:
                    logger.debug("-----------")
                    remain_cnt -= 1
                    # 横方向に繋いだリング
                    rings = self.create_ring(
                        virtual_vertices, remaining_vkeys, base_vertical_axis, None, prev_rings, []
                    )

                    all_rings.append(rings)

                    for rkey in rings:
                        if rkey in remaining_vkeys:
                            del remaining_vkeys[rkey]

                    prev_rings = rings

                for ai, rings in enumerate(all_rings):
                    # 横のリングを縦に並べ直す
                    for vi, v in enumerate(rings):
                        while vi >= len(all_vkeys):
                            all_vkeys.append([])
                        while ai >= len(all_vkeys[vi]):
                            all_vkeys[vi].append([])
                        all_vkeys[vi][ai] = v
            else:
                for ti, bottom_key in enumerate(bottom_edge_keys):
                    bottom_target_pos = virtual_vertices[bottom_key].position()

                    if param_option["route_estimate_type"] == logger.transtext("縮尺"):
                        # 中心から見た処理対象仮想頂点の位置
                        bottom_local_pos = (bottom_target_pos - bottom_edge_mean_pos) * base_reverse_axis
                        bottom_radius_vec = bottom_local_pos.copy()
                        bottom_radius_vec.abs()
                        bottom_radius_vec.one()
                        # 上部の理想位置をざっくり比率から求めておく
                        top_target_pos = top_edge_mean_pos + (
                            bottom_local_pos
                            # 全体の縮尺
                            * (
                                np.array([top_x_radius, top_y_radius, top_z_radius])
                                / np.array([bottom_x_radius, bottom_y_radius, bottom_z_radius])
                            )
                            # 下端エッジの上の方を小さめに判定
                            * abs(
                                abs(top_mean_val - bottom_target_pos.data()[target_idx])
                                / abs(top_mean_val - bottom_mean_val)
                            )
                        )
                        top_distances = np.linalg.norm(
                            (np.array(top_edge_poses) - top_target_pos.data()), ord=2, axis=1
                        )
                        top_nearest_pos = MVector3D(np.array(top_edge_poses)[np.argmin(top_distances)])
                        logger.debug(
                            f"◆上部縮尺推定: bottom_vidxs: [{virtual_vertices[bottom_key].vidxs()}], top_vidxs: [{virtual_vertices[top_nearest_pos.to_key(threshold)].vidxs()}, top_target_pos: [{top_target_pos.to_log()}], top_nearest_pos: [{top_nearest_pos.to_log()}]"
                        )
                    elif param_option["route_estimate_type"] == logger.transtext("角度"):
                        # 下部の角度に類似した上部角度を選ぶ
                        bottom_edge_start_pos.setY(bottom_target_pos.y())
                        bottom_edge_mean_pos.setY(bottom_target_pos.y())
                        bottom_degree0, bottom_degree1 = self.calc_arc_degree(
                            bottom_edge_start_pos,
                            bottom_edge_mean_pos,
                            bottom_target_pos,
                            base_vertical_axis,
                            base_reverse_axis,
                        )

                        degree_diffs0 = np.abs(np.array(list(top_degrees.keys())) - bottom_degree0)
                        degree_diffs1 = np.abs(np.array(list(top_degrees.keys())) - bottom_degree1)

                        if np.min(degree_diffs0) <= np.min(degree_diffs1):
                            bottom_degree = bottom_degree0
                            top_target_pos = np.array(list(top_degrees.values()))[np.argmin(degree_diffs0)]
                            top_degree = np.array(list(top_degrees.keys()))[np.argmin(degree_diffs0)]
                        else:
                            bottom_degree = bottom_degree1
                            top_target_pos = np.array(list(top_degrees.values()))[np.argmin(degree_diffs1)]
                            top_degree = np.array(list(top_degrees.keys()))[np.argmin(degree_diffs1)]

                        logger.debug(
                            f"◆上部角度推定: bottom_vidxs: [{virtual_vertices[bottom_key].vidxs()}], top_vidxs: [{virtual_vertices[top_target_pos.to_key(threshold)].vidxs()}], bottom_degree: [{round(bottom_degree, 3)}], nearest_top_degree: [{round(top_degree, 3)}], top_pos: [{top_target_pos.to_log()}]"
                        )
                    else:
                        # 反対方向にまっすぐ伸ばす
                        top_target_pos = bottom_target_pos + (base_vertical_axis * 3 * -1)

                    vkeys, vscores = self.create_vertex_line_map(
                        bottom_key,
                        top_target_pos,
                        bottom_key,
                        horizonal_top_edge_keys,
                        virtual_vertices,
                        (base_vertical_axis * -1),
                        [bottom_key],
                        [],
                        param_option,
                        [],
                    )
                    score = 0
                    if vscores:
                        if param_option["special_shape"] == logger.transtext("全て表面"):
                            # プリーツはスコアの重み付平均を取る
                            score = np.average(
                                vscores, weights=list(reversed((np.arange(1, len(vscores) + 1) ** 2).tolist()))
                            )
                        else:
                            # スコア中央値を取る
                            score = np.mean(vscores)

                    logger.info(
                        "%s: 頂点ルート走査[%s]: 始端: %s -> 終端: %s, スコア: %s",
                        material_name,
                        f"{(ti + 1):03d}",
                        virtual_vertices[vkeys[0]].vidxs() if vkeys else "NG",
                        virtual_vertices[vkeys[-1]].vidxs() if vkeys else "NG",
                        round(score, 4) if vscores else "-",
                    )
                    if len(vkeys) > 1:
                        all_vkeys.append(vkeys)

            logger.info("-----------")
            logger.info("%s: 頂点マップ[%s]: 登録対象チェック", material_name, f"{(bi + 1):03d}")

            # XYの最大と最小の抽出
            xu = np.unique([i for i, vk in enumerate(all_vkeys)])
            # Yは全マップの中央値を確保する
            yu = np.unique([i for vks in all_vkeys for i, vk in enumerate(vks)])

            # 存在しない頂点INDEXで二次元配列初期化
            tmp_vertex_map = np.full((len(yu) * 2, len(xu), 3), (np.nan, np.nan, np.nan))
            # tmp_score_map = np.full(len(xu), 0.0, dtype=np.float)
            registered_vertices = []

            for x, vkeys in enumerate(all_vkeys):
                is_regists = [True for y in vkeys]
                for px, pvkeys in enumerate(all_vkeys):
                    if x == px:
                        continue
                    for y, vkey in enumerate(vkeys):
                        if vkey in pvkeys or vkey in registered_vkeys:
                            # 他で登録されているキーがある場合は登録対象外
                            is_regists[y] = False

                if not np.count_nonzero(is_regists):
                    # すべてのキーが他のラインで登録されている場合、登録対象外の場合、スルー
                    continue

                # 根元から順にマップに埋めていく
                for y, vkey in enumerate(vkeys):
                    vv = virtual_vertices[vkey]
                    if not vv.vidxs():
                        continue

                    logger.test(f"x: {x}, y: {y}, vv: {vkey}, vidxs: {vv.vidxs()}")

                    tmp_vertex_map[y, x] = vkey
                    registered_vertices.append(vkey)

                logger.test("-------")

            logger.info("%s: 頂点マップ[%s]: 不要軸削除", material_name, f"{(bi + 1):03d}")

            # X軸方向の削除
            remove_xidxs = []
            for v_xidx in range(tmp_vertex_map.shape[1]):
                if np.isnan(tmp_vertex_map[:, v_xidx]).all():
                    # 全部nanの場合、削除対象
                    remove_xidxs.append(v_xidx)

            tmp_vertex_map = np.delete(tmp_vertex_map, remove_xidxs, axis=1)

            # Y軸方向の削除
            remove_yidxs = []
            active_yidxs = []
            for v_yidx in range(tmp_vertex_map.shape[0]):
                if np.isnan(tmp_vertex_map[v_yidx, :]).all():
                    remove_yidxs.append(v_yidx)
                else:
                    # どこか有効な値があれば残す
                    active_yidxs.append(v_yidx)

            # 根元の1つは必ず残して仮想頂点を作る
            if active_yidxs:
                remove_yidxs = (
                    np.array(remove_yidxs)[np.where(np.array(remove_yidxs) < min(active_yidxs))[0]].tolist()
                    + np.array(remove_yidxs)[np.where(np.array(remove_yidxs) > max(active_yidxs) + 1)[0]].tolist()
                )

            tmp_vertex_map = np.delete(tmp_vertex_map, remove_yidxs, axis=0)

            if not tmp_vertex_map.any():
                continue

            logger.info("%s: 頂点マップ[%s]: マップ生成", material_name, f"{(bi + 1):03d}")

            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((tmp_vertex_map.shape[0], tmp_vertex_map.shape[1], 3), (np.nan, np.nan, np.nan))
            vertex_display_map = np.full((tmp_vertex_map.shape[0], tmp_vertex_map.shape[1]), "None  ")

            for x in range(tmp_vertex_map.shape[1]):
                for y, vkey in enumerate(tmp_vertex_map[:, x]):
                    if np.isnan(vkey).any():
                        continue
                    vkey = tuple(vkey)
                    vv = virtual_vertices[vkey]
                    if not vv.vidxs():
                        continue

                    logger.test(f"x: {x}, y: {y}, vv: {vkey}, vidxs: {vv.vidxs()}")

                    vertex_map[y, x] = vkey
                    vertex_display_map[y, x] = ":".join([str(v) for v in vv.vidxs()])

                logger.test("-------")

            if vertex_map.any():
                target_vertex_maps = {}
                target_vertex_display_maps = {}
                target_vertex_maps[len(target_vertex_maps)] = vertex_map
                target_vertex_display_maps[len(target_vertex_display_maps)] = vertex_display_map

                for target_vertex_map, target_vertex_display_map in zip(
                    target_vertex_maps.values(), target_vertex_display_maps.values()
                ):
                    # X軸方向の削除
                    remove_xidxs = []
                    for v_xidx in range(target_vertex_map.shape[1]):
                        if np.isnan(target_vertex_map[:, v_xidx]).all():
                            # 全部nanの場合、削除対象
                            remove_xidxs.append(v_xidx)

                    target_vertex_map = np.delete(target_vertex_map, remove_xidxs, axis=1)
                    target_vertex_display_map = np.delete(target_vertex_display_map, remove_xidxs, axis=1)

                    # Y軸方向の削除
                    remove_yidxs = []
                    active_yidxs = []
                    for v_yidx in range(target_vertex_map.shape[0]):
                        if np.isnan(target_vertex_map[v_yidx, :]).all():
                            remove_yidxs.append(v_yidx)
                        else:
                            # どこか有効な値があれば残す
                            active_yidxs.append(v_yidx)

                    # 描画は実際の頂点のみとする
                    target_vertex_display_map = np.delete(target_vertex_display_map, remove_yidxs, axis=0)

                    # 実際は根元の1つは必ず残して仮想頂点を作る
                    remove_yidxs = (
                        np.array(remove_yidxs)[np.where(np.array(remove_yidxs) < min(active_yidxs))[0]].tolist()
                        + np.array(remove_yidxs)[np.where(np.array(remove_yidxs) > max(active_yidxs) + 1)[0]].tolist()
                    )

                    target_vertex_map = np.delete(target_vertex_map, remove_yidxs, axis=0)

                    # オフセット分ずらす
                    horizonal_bone_offset = param_option["horizonal_bone_offset"]
                    target_vertex_map = np.roll(target_vertex_map, -horizonal_bone_offset, axis=1)

                    logger.info("%s: 頂点マップ[%s]: 仮想頂点生成", material_name, f"{(bi + 1):03d}")

                    for v_yidx in range(target_vertex_map.shape[0]):
                        for v_xidx in range(target_vertex_map.shape[1]):
                            if np.isnan(target_vertex_map[v_yidx, v_xidx]).any():
                                # ない場合、仮想頂点を設定する
                                nearest_v_yidx = (
                                    np.where(~np.isnan(target_vertex_map[:, v_xidx]))[0][
                                        np.argmin(
                                            np.abs(np.where(~np.isnan(target_vertex_map[:, v_xidx]))[0] - v_yidx)
                                        )
                                    ]
                                    if np.where(~np.isnan(target_vertex_map[:, v_xidx]))[0].any()
                                    else 0
                                )
                                nearest_v_xidx = (
                                    np.where(~np.isnan(target_vertex_map[nearest_v_yidx, :, 0]))[0][
                                        np.argmin(
                                            np.abs(
                                                np.where(~np.isnan(target_vertex_map[nearest_v_yidx, :, 0]))[0]
                                                - v_xidx
                                            )
                                        )
                                    ]
                                    if np.where(~np.isnan(target_vertex_map[nearest_v_yidx, :, 0]))[0].any()
                                    else 0
                                )
                                nearest_above_v_yidx = (
                                    np.where(~np.isnan(target_vertex_map[:nearest_v_yidx, nearest_v_xidx]))[0][
                                        np.argmin(
                                            np.abs(
                                                np.where(
                                                    ~np.isnan(target_vertex_map[:nearest_v_yidx, nearest_v_xidx])
                                                )[0]
                                                - nearest_v_yidx
                                            )
                                        )
                                    ]
                                    if np.where(~np.isnan(target_vertex_map[:nearest_v_yidx, nearest_v_xidx]))[0].any()
                                    else 0
                                )
                                above_yidx = (
                                    np.where(~np.isnan(target_vertex_map[:v_yidx, v_xidx]))[0][
                                        np.argmin(
                                            np.abs(np.where(~np.isnan(target_vertex_map[:v_yidx, v_xidx]))[0] - v_yidx)
                                        )
                                    ]
                                    if np.where(~np.isnan(target_vertex_map[:v_yidx, v_xidx]))[0].any()
                                    else 0
                                )
                                above_above_yidx = (
                                    np.where(~np.isnan(target_vertex_map[:above_yidx, v_xidx]))[0][
                                        np.argmin(
                                            np.abs(
                                                np.where(~np.isnan(target_vertex_map[:above_yidx, v_xidx]))[0]
                                                - above_yidx
                                            )
                                        )
                                    ]
                                    if np.where(~np.isnan(target_vertex_map[:above_yidx, v_xidx]))[0].any()
                                    else 0
                                )

                                if (
                                    np.isnan(target_vertex_map[nearest_above_v_yidx, nearest_v_xidx]).any()
                                    or np.isnan(target_vertex_map[nearest_v_yidx, nearest_v_xidx]).any()
                                    or np.isnan(target_vertex_map[above_yidx, v_xidx]).any()
                                    or np.isnan(target_vertex_map[above_above_yidx, v_xidx]).any()
                                ):
                                    continue

                                nearest_vv = virtual_vertices[tuple(target_vertex_map[nearest_v_yidx, nearest_v_xidx])]
                                nearest_above_vv = virtual_vertices[
                                    tuple(target_vertex_map[nearest_above_v_yidx, nearest_v_xidx])
                                ]
                                above_vv = virtual_vertices[tuple(target_vertex_map[above_yidx, v_xidx])]
                                above_above_vv = virtual_vertices[tuple(target_vertex_map[above_above_yidx, v_xidx])]

                                # 直近頂点の上下距離
                                nearest_distance = nearest_vv.position().distanceToPoint(nearest_above_vv.position())

                                # ボーン進行方向(x)
                                x_direction_pos = (above_above_vv.position() - above_vv.position()).normalized()
                                # ボーン進行方向に対しての横軸(y)
                                y_direction_pos = (
                                    MVector3D(0, 0, -1) * x_direction_pos.x()
                                    if np.isclose(abs(x_direction_pos.x()), 1)
                                    else MVector3D(1, 0, 0)
                                )
                                # ボーン進行方向に対しての縦軸(z)
                                z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
                                above_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)

                                mat = MMatrix4x4()
                                mat.setToIdentity()
                                mat.translate(above_vv.position())
                                mat.rotate(above_qq)

                                # 仮想頂点の位置
                                target_position = mat * MVector3D(
                                    0,
                                    -nearest_distance,
                                    0,
                                )
                                target_key = target_position.to_key(threshold)
                                if target_key not in virtual_vertices:
                                    virtual_vertices[target_key] = VirtualVertex(target_key)
                                virtual_vertices[target_key].positions.append(target_position.data())
                                target_vertex_map[v_yidx, v_xidx] = target_key

                    # 最上部が同じキーで揃ってるINDEX
                    same_top_keys = np.where(np.sum(np.abs(np.diff(target_vertex_map[:1], axis=1)), axis=2) == 0)
                    separated_vertex_maps = []
                    separated_vertex_display_maps = []
                    if same_top_keys[1].any():
                        prev_v_xidx = 0
                        for v_xidx in same_top_keys[1]:
                            separated_vertex_maps.append(target_vertex_map[:, prev_v_xidx : (v_xidx + 1)])
                            separated_vertex_display_maps.append(
                                target_vertex_display_map[:, prev_v_xidx : (v_xidx + 1)]
                            )
                            prev_v_xidx = v_xidx + 1
                        separated_vertex_maps.append(target_vertex_map[:, prev_v_xidx:])
                        separated_vertex_display_maps.append(target_vertex_display_map[:, prev_v_xidx:])
                    else:
                        separated_vertex_maps.append(target_vertex_map)
                        separated_vertex_display_maps.append(target_vertex_display_map)

                    for (sv_map, svd_map) in zip(separated_vertex_maps, separated_vertex_display_maps):
                        vertex_maps[len(vertex_maps)] = sv_map

                        logger.info("-----------------------")
                        logger.info("%s: 頂点マップ[%s]: マップ生成 ------", material_name, f"{len(vertex_maps):03d}")
                        logger.info(
                            "\n".join([", ".join(svd_map[vx, :]) for vx in range(svd_map.shape[0])]),
                            translate=False,
                        )

            logger.info("%s: 頂点マップ[%s]: 終了 ---------", material_name, f"{(bi + 1):03d}")

        logger.debug("-----------------------")

        return (
            vertex_maps,
            virtual_vertices,
            remaining_vertices,
            back_vertices,
            threshold,
            is_material_horizonal,
            horizonal_top_edge_keys,
        )

    def create_ring(
        self,
        virtual_vertices: dict,
        remaining_vkeys: list,
        base_vertical_axis: MVector3D,
        vkey: tuple,
        prev_rings: list,
        rings: list,
    ):
        axis_idx = np.where(np.abs(base_vertical_axis.data()))[0][0]

        if not vkey:
            if 2 > len(prev_rings):
                return rings

            # vkey が決まってない場合、始点なのでprevから繋がってる頂点を選ぶ
            prev_start_vv: VirtualVertex = virtual_vertices[prev_rings[0]]
            prev_next_vv: VirtualVertex = virtual_vertices[prev_rings[1]]

            # 進行方向(x)
            x_direction_pos = (prev_next_vv.position() - prev_start_vv.position()).normalized()
            # 進行方向に対しての縦軸(y)
            y_direction_pos = MVector3D(0, -1, 0)
            # ボーン進行方向に対しての縦軸(y)
            if np.isclose(abs(x_direction_pos.y()), 1):
                # ちょうど垂直で真下に向かってる場合、縦軸を別に設定する
                y_direction_pos = MVector3D(1, 0, 0)
            # 進行方向に対しての横軸(z)
            z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos).normalized()

            connected_dots = {}
            for connected_vkey in prev_start_vv.connected_vvs:
                # 内積
                if connected_vkey in remaining_vkeys and (
                    (connected_vkey in prev_next_vv.connected_vvs)
                    or True in [prev_rings[0] in virtual_vertices[connected_vkey].connected_vvs]
                ):
                    # 自分と繋がってる頂点が隣とも繋がってる（＼）か、走査頂点と繋がってる頂点が自分と繋がっている（／）場合のみ対象
                    connected_dots[connected_vkey] = MVector3D.dotProduct(
                        z_direction_pos,
                        (virtual_vertices[connected_vkey].position() - prev_start_vv.position()).normalized(),
                    )

            if not connected_dots:
                return rings

            logger.debug(
                "*** vertical-dots: %s",
                [(k, virtual_vertices[k].vidxs(), round(v, 3)) for k, v in connected_dots.items()],
            )
            # 内積が一番大きいものを縦方向とみなす
            vkey = list(connected_dots.keys())[np.argmax(list(connected_dots.values()))]

        # 登録
        rings.append(vkey)

        vv: VirtualVertex = virtual_vertices[vkey]
        connected_vecs = {}
        connected_dots = {}
        for connected_vkey in vv.connected_vvs:
            # 残頂点に含まれていて、かつ上のリングと繋がっているキーのみ対象
            if (
                connected_vkey in remaining_vkeys
                and len(rings) < len(prev_rings)
                and connected_vkey in virtual_vertices[prev_rings[len(rings)]].connected_vvs
            ):
                # 評価軸の絶対値
                connected_vec = vv.position() - virtual_vertices[connected_vkey].position()
                connected_vecs[connected_vkey] = np.abs((connected_vec.data())[axis_idx])
                if len(rings) > 1:
                    # 内積
                    connected_dots[connected_vkey] = MVector3D.dotProduct(
                        (vv.position() - virtual_vertices[rings[-2]].position()).normalized(),
                        (virtual_vertices[connected_vkey].position() - vv.position()).normalized(),
                    )
        logger.debug(
            "*** next-dots: %s", [(k, virtual_vertices[k].vidxs(), round(v, 3)) for k, v in connected_dots.items()]
        )

        if not connected_vecs:
            return rings

        if len(rings) < 2:
            # 評価軸が一番小さいものが横方向とみなす
            next_vkey = list(connected_vecs.keys())[np.argmin(list(connected_vecs.values()))]

            logger.debug(
                "** ring vkey[%s], next[%s], length[%s]",
                vv.vidxs(),
                virtual_vertices[next_vkey].vidxs(),
                round(np.max(list(connected_vecs.values())), 3),
            )
        else:
            dots = np.array(list(connected_dots.values()))
            logger.debug("*** dots: %s", np.round(dots, decimals=3))
            # 内積が一番大きいものを横方向とみなす
            next_vkey = list(connected_dots.keys())[np.argmax(dots)]

            logger.debug(
                "** ring vkey[%s], next[%s], dot[%s]",
                vv.vidxs(),
                virtual_vertices[next_vkey].vidxs(),
                round(np.max(dots), 3),
            )

        if next_vkey in rings:
            # 既に組み込まれてる方に進む場合、終了
            return rings

        return self.create_ring(virtual_vertices, remaining_vkeys, base_vertical_axis, next_vkey, prev_rings, rings)

    def create_vertex_line_map(
        self,
        bottom_key: tuple,
        top_pos: MVector3D,
        from_key: tuple,
        top_edge_keys: list,
        virtual_vertices: dict,
        base_vertical_reverse_axis: MVector3D,
        vkeys: list,
        vscores: list,
        param_option: dict,
        registered_vkeys: list,
        loop=0,
    ):

        if loop > 500:
            return None, None

        from_vv = virtual_vertices[from_key]
        from_pos = from_vv.position()

        bottom_vv = virtual_vertices[bottom_key]

        # ボーン進行方向(x)
        top_x_pos = (top_pos - from_pos).normalized()
        # ボーン進行方向に対しての縦軸(y)
        top_y_pos = MVector3D(1, 0, 0)
        if np.isclose(abs(top_x_pos.x()), 1):
            # ちょうど水平で真横に向かってる場合、縦軸を別に設定する
            top_y_pos = MVector3D(0, 0, -1)
        # ボーン進行方向に対しての横軸(z)
        top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
        top_qq = MQuaternion.fromDirection(top_z_pos, top_x_pos)
        logger.debug(
            f" - bottom({bottom_vv.vidxs()}): top[{top_pos.to_log()}], x[{top_x_pos.to_log()}], y[{top_y_pos.to_log()}], z[{top_z_pos.to_log()}]"
        )

        mat = MMatrix4x4()
        mat.setToIdentity()
        mat.translate(from_pos)
        mat.rotate(top_qq)

        local_top_vpos = (mat.inverted() * top_pos).normalized()

        scores = []
        for n, to_key in enumerate(from_vv.connected_vvs):
            if to_key not in virtual_vertices:
                # 存在しないキーは無視
                scores.append(0)
                logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_key}], 対象外")
                continue

            if to_key in registered_vkeys:
                # 登録済みのは無視（違う列は参照しない）
                scores.append(0)
                logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_key}], 登録済み")
                continue

            to_vv = virtual_vertices[to_key]
            to_pos = to_vv.position()

            local_next_vpos = (mat.inverted() * to_pos).normalized()
            direction_dot = MVector3D.dotProduct(local_top_vpos, local_next_vpos)

            if to_key in vkeys:
                # 到達済みのベクトルには行かせない
                scores.append(0)
                logger.debug(
                    f" - ×get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], 到達済み"
                )
                continue

            if direction_dot < 0.2:
                # 反対方向のベクトルには行かせない
                scores.append(0)
                logger.debug(
                    f" - ×get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], 反対方向"
                )
                continue

            # ローカル軸の向きが出来るだけTOPの向きに沿っている
            yaw_score = 1 - (MVector3D(local_top_vpos.x(), 0, 0) - MVector3D(local_next_vpos.x(), 0, 0)).length()
            pitch_score = 1 - (MVector3D(local_top_vpos.y(), 0, 0) - MVector3D(local_next_vpos.y(), 0, 0)).length()
            roll_score = 1 - (MVector3D(local_top_vpos.z(), 0, 0) - MVector3D(local_next_vpos.z(), 0, 0)).length()

            if param_option["route_search_type"] == logger.transtext("前頂点優先"):
                # 前頂点との内積差を考慮する場合
                prev_dot = (
                    MVector3D.dotProduct(
                        (virtual_vertices[vkeys[0]].position() - virtual_vertices[vkeys[1]].position()).normalized(),
                        (to_pos - virtual_vertices[vkeys[0]].position()).normalized(),
                    )
                    if len(vkeys) > 1
                    else 1
                )
            else:
                # 根元頂点の向きのみ参照する場合
                prev_dot = 1

            score = (yaw_score) * (pitch_score**3) * (roll_score**2)

            scores.append((score**2) * (direction_dot**3) * (prev_dot**5))

            logger.debug(
                f" - ○get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], local_top_vpos[{local_top_vpos.to_log()}], local_next_vpos[{local_next_vpos.to_log()}]"
                + f", direction_dot[{round(direction_dot, 5)}], prev_dot[{round(prev_dot, 5)}], total_score: [{round(score * direction_dot * prev_dot, 5)}], score: [{round(score, 5)}]"
                + f", yaw_score: {round(yaw_score, 5)}, pitch_score: {round(pitch_score, 5)}, roll_score: {round(roll_score, 5)}"
            )

        if np.count_nonzero(scores) == 0:
            # スコアが付けられなくなったら終了
            return vkeys, vscores

        # 最もスコアの高いINDEXを採用
        logger.debug("scores: %s", scores)
        nearest_idx = np.argmax(scores)
        vertical_key = from_vv.connected_vvs[nearest_idx]

        # 前の辺との内積差を考慮する（プリーツライン選択用）
        prev_diff_dot = (
            MVector3D.dotProduct(
                (virtual_vertices[vkeys[0]].position() - virtual_vertices[vkeys[1]].position()).normalized(),
                (virtual_vertices[vertical_key].position() - virtual_vertices[vkeys[0]].position()).normalized(),
            )
            if len(vkeys) > 1 and param_option["special_shape"] == logger.transtext("全て表面")
            else 1
        )

        logger.debug(
            f"◇direction: from: [{virtual_vertices[from_key].vidxs()}], to: [{virtual_vertices[vertical_key].vidxs()}], prev_diff_dot[{round(prev_diff_dot, 4)}]"
        )

        vkeys.insert(0, vertical_key)
        vscores.insert(0, np.max(scores) * prev_diff_dot)

        if vertical_key in top_edge_keys:
            # 上端に辿り着いたら終了
            return vkeys, vscores

        # 上部を求め直す
        if param_option["route_estimate_type"] == logger.transtext("軸方向") or param_option[
            "route_search_type"
        ] == logger.transtext("前頂点優先"):
            # 進行方向に合わせて進行方向を検出する
            # ボーン進行方向(x)
            top_x_pos = (
                virtual_vertices[vertical_key].position() - virtual_vertices[from_key].position()
            ).normalized()
            # ボーン進行方向に対しての縦軸(y)
            top_y_pos = MVector3D(1, 0, 0)
            if np.isclose(abs(top_x_pos.x()), 1):
                # ちょうど水平で真横に向かってる場合、縦軸を別に設定する
                top_y_pos = MVector3D(0, 0, -1)
            # ボーン進行方向に対しての横軸(z)
            top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
            top_qq = MQuaternion.fromDirection(top_z_pos, top_x_pos)

            mat = MMatrix4x4()
            mat.setToIdentity()
            mat.translate(virtual_vertices[from_key].position())
            mat.rotate(top_qq)
            vertical_pos = mat.inverted() * virtual_vertices[vertical_key].position()

            mat = MMatrix4x4()
            mat.setToIdentity()
            mat.translate(virtual_vertices[vertical_key].position())
            mat.rotate(top_qq)
            top_pos = mat * MVector3D(0, vertical_pos.y(), 0)

        return self.create_vertex_line_map(
            bottom_key,
            top_pos,
            vertical_key,
            top_edge_keys,
            virtual_vertices,
            base_vertical_reverse_axis,
            vkeys,
            vscores,
            param_option,
            registered_vkeys,
            loop + 1,
        )

    def calc_arc_degree(
        self,
        start_pos: MVector3D,
        mean_pos: MVector3D,
        target_pos: MVector3D,
        base_vertical_axis: MVector3D,
        base_reverse_axis: MVector3D,
    ):
        start_normal_pos = ((start_pos - mean_pos) * base_reverse_axis).normalized()
        target_normal_pos = ((target_pos - mean_pos) * base_reverse_axis).normalized()
        qq = MQuaternion.rotationTo(start_normal_pos, target_normal_pos)
        degree = qq.toDegreeSign(base_vertical_axis)
        if np.isclose(MVector3D.dotProduct(start_normal_pos, target_normal_pos), -1):
            # ほぼ真後ろを向いてる場合、固定で180度を入れておく
            degree = 180

        return degree, degree + 360

    def get_edge_lines_from_csv(
        self,
        all_line_pairs: dict,
        top_edge_keys: dict,
        start_vkey: tuple,
        edge_lines: list,
        virtual_vertices: dict,
        param_option: dict,
        loop: int,
        is_reverse: bool = False,
    ):
        if not top_edge_keys or loop > 500:
            return edge_lines, loop

        if not start_vkey:
            if param_option["direction"] == logger.transtext("上"):
                # X(中央揃え) - Y(昇順) - Z(降順)
                sorted_edge_line_pairs = sorted(list(top_edge_keys.keys()), key=lambda x: (abs(x[0]), x[1], -x[2]))
            elif param_option["direction"] == logger.transtext("右"):
                # Y(降順) - X(降順) - Z(降順)
                sorted_edge_line_pairs = sorted(list(top_edge_keys.keys()), key=lambda x: (-x[1], -x[0], -x[2]))
            elif param_option["direction"] == logger.transtext("左"):
                # Y(降順) - X(昇順) - Z(降順)
                sorted_edge_line_pairs = sorted(list(top_edge_keys.keys()), key=lambda x: (-x[1], x[0], -x[2]))
            else:
                # 下: X(中央揃え) - Y(降順) - Z(降順)
                sorted_edge_line_pairs = sorted(list(top_edge_keys.keys()), key=lambda x: (abs(x[0]), -x[1], -x[2]))
            start_vkey = sorted_edge_line_pairs[0]
            edge_lines.append(start_vkey)

        if start_vkey not in all_line_pairs:
            return edge_lines, loop

        if start_vkey in top_edge_keys:
            del top_edge_keys[start_vkey]

        remain_next_vkeys = [
            vkey for vkey in all_line_pairs[start_vkey] if vkey not in edge_lines and vkey in top_edge_keys
        ]

        if not remain_next_vkeys:
            if not top_edge_keys:
                return edge_lines, loop
            # まだTOPキーが残っている場合、反対方向に進む
            is_reverse = True
            remain_next_vkeys = [
                vkey for vkey in all_line_pairs[edge_lines[0]] if vkey not in edge_lines and vkey in top_edge_keys
            ]
            if not remain_next_vkeys:
                # 前が繋がってない場合、無視する
                return edge_lines, loop

        if len(edge_lines) == 1:
            # 初回はルールに沿って次を抽出する
            if param_option["direction"] == logger.transtext("上"):
                # X(中央揃え) - Z(降順) - Y(昇順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (abs(x[0]), -x[2], x[1]))[0]
            elif param_option["direction"] == logger.transtext("右"):
                # Y(降順) - Z(降順) - X(降順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (-x[1], -x[2], -x[0]))[0]
            elif param_option["direction"] == logger.transtext("左"):
                # Y(降順) - Z(降順) - X(昇順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (-x[1], -x[2], x[0]))[0]
            else:
                # 下: X(中央揃え) - Z(降順) - Y(降順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (abs(x[0]), -x[2], -x[1]))[0]
        else:
            # 2回目以降は出来るだけ内積が大きいのを選ぶ
            next_dots = []
            for vkey in remain_next_vkeys:
                next_dots.append(
                    MVector3D.dotProduct(
                        (
                            virtual_vertices[edge_lines[-1]].position() - virtual_vertices[edge_lines[-2]].position()
                        ).normalized(),
                        (virtual_vertices[vkey].position() - virtual_vertices[edge_lines[-1]].position()).normalized(),
                    )
                )
            next_vkey = remain_next_vkeys[np.argmax(next_dots)]

        if is_reverse:
            edge_lines.insert(0, next_vkey)
        else:
            edge_lines.append(next_vkey)

        return self.get_edge_lines_from_csv(
            all_line_pairs, top_edge_keys, next_vkey, edge_lines, virtual_vertices, param_option, loop + 1, is_reverse
        )

    def get_edge_lines(
        self,
        edge_line_pairs: dict,
        registered_edge_line_pairs: dict,
        start_vkey: tuple,
        edge_lines: list,
        edge_vkeys: list,
        virtual_vertices: dict,
        param_option: dict,
        loop: int,
        n: int,
    ):
        remain_start_vkeys = [elp for elp in edge_line_pairs.keys() if edge_line_pairs[elp]]

        if not remain_start_vkeys or loop > 500:
            return start_vkey, edge_lines, edge_vkeys, registered_edge_line_pairs

        if not start_vkey:
            if param_option["direction"] == logger.transtext("上"):
                # X(中央揃え) - Y(昇順) - Z(降順)
                sorted_edge_line_pairs = sorted(remain_start_vkeys, key=lambda x: (abs(x[0]), x[1], -x[2]))
            elif param_option["direction"] == logger.transtext("右"):
                # Y(降順) - X(降順) - Z(降順)
                sorted_edge_line_pairs = sorted(remain_start_vkeys, key=lambda x: (-x[1], -x[0], -x[2]))
            elif param_option["direction"] == logger.transtext("左"):
                # Y(降順) - X(昇順) - Z(降順)
                sorted_edge_line_pairs = sorted(remain_start_vkeys, key=lambda x: (-x[1], x[0], -x[2]))
            else:
                # 下: X(中央揃え) - Y(降順) - Z(降順)
                sorted_edge_line_pairs = sorted(remain_start_vkeys, key=lambda x: (abs(x[0]), -x[1], -x[2]))
            start_vkey = sorted_edge_line_pairs[0]
            edge_lines.append([start_vkey])

        remain_next_vkeys = list(
            set(
                [
                    nk
                    for nk in edge_line_pairs[start_vkey]
                    if (start_vkey, nk) not in edge_vkeys and (nk, start_vkey) not in edge_vkeys
                ]
            )
        )

        if not remain_next_vkeys:
            if start_vkey and start_vkey in edge_line_pairs:
                if start_vkey not in registered_edge_line_pairs:
                    registered_edge_line_pairs[start_vkey] = []
                for vk in registered_edge_line_pairs[start_vkey]:
                    if vk not in registered_edge_line_pairs[start_vkey]:
                        registered_edge_line_pairs[start_vkey].append(vk)

                del edge_line_pairs[start_vkey]

            return None, edge_lines, edge_vkeys, registered_edge_line_pairs

        if len(edge_lines[-1]) == 1:
            # 初回はルールに沿って次を抽出する
            if param_option["direction"] == logger.transtext("上"):
                # X(中央揃え) - Z(降順) - Y(昇順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (abs(x[0]), -x[2], x[1]))[0]
            elif param_option["direction"] == logger.transtext("右"):
                # Y(降順) - Z(降順) - X(降順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (-x[1], -x[2], -x[0]))[0]
            elif param_option["direction"] == logger.transtext("左"):
                # Y(降順) - Z(降順) - X(昇順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (-x[1], -x[2], x[0]))[0]
            else:
                # 下: X(中央揃え) - Z(降順) - Y(降順)
                next_vkey = sorted(remain_next_vkeys, key=lambda x: (abs(x[0]), -x[2], -x[1]))[0]
        else:
            # 2回目以降は出来るだけ内積が大きいのを選ぶ
            next_dots = []
            for vkey in remain_next_vkeys:
                next_dots.append(
                    MVector3D.dotProduct(
                        (
                            virtual_vertices[edge_lines[-1][-1]].position()
                            - virtual_vertices[edge_lines[-1][-2]].position()
                        ).normalized(),
                        (
                            virtual_vertices[vkey].position() - virtual_vertices[edge_lines[-1][-1]].position()
                        ).normalized(),
                    )
                )
            next_vkey = remain_next_vkeys[np.argmax(next_dots)]

        if start_vkey:
            if start_vkey and (start_vkey, next_vkey) not in edge_vkeys:
                # まだ組み合わせが登録されていない場合のみ登録
                edge_lines[-1].append(next_vkey)
                edge_vkeys.append((start_vkey, next_vkey))

            if start_vkey not in registered_edge_line_pairs:
                registered_edge_line_pairs[start_vkey] = []
            if next_vkey not in registered_edge_line_pairs[start_vkey]:
                registered_edge_line_pairs[start_vkey].append(next_vkey)

            if next_vkey not in registered_edge_line_pairs:
                registered_edge_line_pairs[next_vkey] = []
            if start_vkey not in registered_edge_line_pairs[next_vkey]:
                registered_edge_line_pairs[next_vkey].append(start_vkey)

            for n, nk in enumerate(edge_line_pairs[start_vkey]):
                if nk == next_vkey:
                    del edge_line_pairs[start_vkey][n]

            for n, sk in enumerate(edge_line_pairs[next_vkey]):
                if sk == start_vkey:
                    del edge_line_pairs[next_vkey][n]

            if start_vkey in edge_line_pairs and not edge_line_pairs[start_vkey]:
                if start_vkey not in registered_edge_line_pairs:
                    registered_edge_line_pairs[start_vkey] = []
                for vk in registered_edge_line_pairs[start_vkey]:
                    if vk not in registered_edge_line_pairs[start_vkey]:
                        registered_edge_line_pairs[start_vkey].append(vk)

                del edge_line_pairs[start_vkey]

            _, edge_lines, edge_vkeys, registered_edge_line_pairs = self.get_edge_lines(
                edge_line_pairs,
                registered_edge_line_pairs,
                next_vkey,
                edge_lines,
                edge_vkeys,
                virtual_vertices,
                param_option,
                loop + 1,
                n,
            )

        return None, edge_lines, edge_vkeys, registered_edge_line_pairs

    def get_rigidbody(self, model: PmxModel, bone_name: str):
        if bone_name not in model.bones:
            return None

        for rigidbody in model.rigidbodies.values():
            if rigidbody.bone_index == model.bones[bone_name].index:
                return rigidbody

        return None

    def get_block_vidxs(
        self,
        v_yidx: int,
        v_xidx: int,
        vertex_maps: dict,
        all_registered_bones: dict,
        all_bone_connected: dict,
        base_map_idx: int,
        is_weight=False,
        is_center=False,
    ):
        registered_bones = all_registered_bones[base_map_idx]
        vertex_map = vertex_maps[base_map_idx]

        target_v_yidx = (
            np.max(np.where(registered_bones[: (v_yidx + 1), v_xidx]))
            if registered_bones[: (v_yidx + 1), v_xidx].any()
            else np.max(np.where(registered_bones[: (v_yidx + 1), : (v_xidx + 1)])[0])
            if registered_bones[: (v_yidx + 1), : (v_xidx + 1)].any()
            else registered_bones.shape[0] - 1
        )
        target_v_xidx = (
            np.max(np.where(registered_bones[target_v_yidx, : (v_xidx + 1)]))
            if registered_bones[target_v_yidx, : (v_xidx + 1)].any()
            else registered_bones.shape[1] - 1
        )

        # 一旦は登録対象のボーンINDEXをベースに最大INDEXを取得する

        registered_max_v_xidx = (
            np.max(np.where(registered_bones[target_v_yidx, :]))
            if registered_bones[target_v_yidx, :].any()
            and np.max(np.where(registered_bones[target_v_yidx, :])) >= v_xidx
            else np.max(np.where(registered_bones[: (target_v_yidx + 1), :]))
            if np.where(registered_bones[: (target_v_yidx + 1), :])[1].any()
            and np.max(np.where(registered_bones[: (target_v_yidx + 1), :])) >= v_xidx
            else registered_bones.shape[1] - 1
        )

        registered_max_v_yidx = (
            np.max(np.where(registered_bones[:, target_v_xidx]))
            if np.where(registered_bones[:, target_v_xidx])[0].any()
            else np.max(np.where(registered_bones[:, : (target_v_xidx + 1)])[0])
            if registered_bones[:, : (target_v_xidx + 1)].any()
            else registered_bones.shape[0] - 1
        )

        prev_xidx = 0
        prev_map_idx = base_map_idx
        prev_connected = False
        if is_center:
            # 中央配置の場合、端っこ
            prev_xidx = 0
            prev_connected = True
        elif v_xidx == 0:
            prev_map_idx = (
                base_map_idx
                if len(all_bone_connected) == 1
                else base_map_idx - 1
                if base_map_idx > 0
                else len(all_bone_connected) - 1
            )
            prev_xidx = all_bone_connected[prev_map_idx].shape[1] - 1
            if len(all_bone_connected) == 1 and all_bone_connected[base_map_idx][target_v_yidx, -1]:
                # 最後が先頭と繋がっている場合、最後と繋ぐ
                prev_xidx = registered_max_v_xidx
                prev_map_idx = base_map_idx
                prev_connected = True
            elif (
                base_map_idx > 0
                and all_bone_connected[list(all_bone_connected.keys())[base_map_idx - 1]].shape[1] > 0
                and all_bone_connected[list(all_bone_connected.keys())[base_map_idx - 1]].shape[0] > target_v_yidx
                and all_bone_connected[list(all_bone_connected.keys())[base_map_idx - 1]][target_v_yidx, -1].any()
            ):
                prev_map_idx = list(all_bone_connected.keys())[base_map_idx - 1]
                prev_connected = True
                # 最後が先頭と繋がっている場合、最後と繋ぐ
                if (
                    tuple(vertex_maps[prev_map_idx][target_v_yidx, -1])
                    == tuple(vertex_maps[base_map_idx][target_v_yidx, v_xidx])
                    and all_bone_connected[prev_map_idx].shape[1] > 1
                    and all_bone_connected[prev_map_idx][target_v_yidx, -2].any()
                ):
                    # 前のボーンが同じ仮想頂点であり、かつそのもうひとつ前と繋がっている場合
                    prev_xidx = (
                        np.max(np.where(all_registered_bones[prev_map_idx][target_v_yidx, :-1]))
                        if prev_map_idx in all_registered_bones
                        and all_registered_bones[prev_map_idx][target_v_yidx, :-1].any()
                        else np.max(np.where(all_registered_bones[prev_map_idx][: (target_v_yidx + 1), :-1])[1])
                        if prev_map_idx in all_registered_bones
                        and np.where(all_registered_bones[prev_map_idx][: (target_v_yidx + 1), :-1])[1].any()
                        else 0
                    )
                else:
                    # 前のボーンの仮想頂点が自分と違う場合、そのまま前のを採用
                    prev_xidx = (
                        np.max(np.where(all_registered_bones[prev_map_idx][target_v_yidx, :]))
                        if prev_map_idx in all_registered_bones
                        and all_registered_bones[prev_map_idx][target_v_yidx, :].any()
                        else np.max(np.where(all_registered_bones[prev_map_idx][: (target_v_yidx + 1), :])[1])
                        if prev_map_idx in all_registered_bones
                        and np.where(all_registered_bones[prev_map_idx][: (target_v_yidx + 1), :])[1].any()
                        else 0
                    )
        else:
            # 剛体・ジョイントの1番目以降は、自分より前で、同じ列にボーンが登録されている最も近いの
            prev_xidx = max(
                set(np.where(registered_bones[target_v_yidx, :v_xidx])[0].tolist())
                | set(np.where(all_bone_connected[base_map_idx][target_v_yidx, :v_xidx] == 0)[0].tolist())
                | {0}
            )
            prev_connected = True if all_bone_connected[base_map_idx][v_yidx, prev_xidx] else False

        next_xidx = registered_max_v_xidx
        next_map_idx = base_map_idx
        next_connected = False
        if is_center:
            # 中央配置の場合、端っこ
            next_xidx = registered_bones.shape[1] - 1
            next_connected = True
        elif v_xidx >= registered_max_v_xidx:
            next_map_idx = (
                base_map_idx
                if len(all_bone_connected) == 1
                else base_map_idx + 1
                if base_map_idx < len(all_bone_connected) - 1
                else base_map_idx
            )
            next_xidx = 0
            if len(all_bone_connected) == 1 and all_bone_connected[base_map_idx][v_yidx, registered_max_v_xidx:].any():
                # 最後が先頭と繋がっている場合(最後の有効ボーンから最初までがどこか繋がっている場合）、最後と繋ぐ（マップが1つの場合）
                next_connected = True
                next_map_idx = 0
            elif (
                base_map_idx < len(all_bone_connected) - 1
                and all_bone_connected[base_map_idx][v_yidx, registered_max_v_xidx:].any()
            ):
                # マップが複数、かつ最後ではない場合（次の先頭と繋ぐ）
                next_connected = True
            elif (
                base_map_idx == len(all_bone_connected) - 1
                and all_bone_connected[base_map_idx][v_yidx, registered_max_v_xidx:].any()
            ):
                # マップが複数かつ最後である場合（最初の先頭と繋ぐ）
                next_connected = True
                next_map_idx = 0
        else:
            # 剛体・ジョイントのmaxより前は、自分より前で、縦列のいずれかにボーンが登録されている最も近いの
            next_xidx = min(
                set((np.where(registered_bones[target_v_yidx, (v_xidx + 1) :])[0] + (v_xidx + 1)).tolist())
                | set(
                    (
                        np.where(
                            all_bone_connected[base_map_idx][
                                min(all_bone_connected[base_map_idx].shape[0] - 2, target_v_yidx), (v_xidx + 1) :
                            ]
                            == 0
                        )[0]
                        + (v_xidx + 1)
                    ).tolist()
                )
                | {registered_bones.shape[1] - 1}
            )
            # 自身から隣が繋がっているか
            next_connected = True if all_bone_connected[base_map_idx][v_yidx, v_xidx] else False

        above_yidx = 0
        if v_yidx > 0:
            # 横列上方のいずれかに登録されている場合
            above_yidx = (
                np.max(np.where(registered_bones[:v_yidx, target_v_xidx])[0])
                if np.where(registered_bones[:v_yidx, target_v_xidx])[0].any()
                else 0
            )

        # 横列下方のいずれかに登録されている場合
        below_yidx = (
            np.min(np.where(registered_bones[(v_yidx + 1) :, target_v_xidx])[0]) + (v_yidx + 1)
            if np.where(registered_bones[(v_yidx + 1) :, target_v_xidx])[0].any()
            else registered_max_v_yidx
        )

        max_v_xidx = np.max(np.where(vertex_map[v_yidx, :, :])[0])
        max_v_yidx = np.max(np.where(vertex_map[:, v_xidx, :])[0])

        return (
            prev_map_idx,
            prev_xidx,
            prev_connected,
            next_map_idx,
            next_xidx,
            next_connected,
            above_yidx,
            below_yidx,
            target_v_yidx,
            target_v_xidx,
            registered_max_v_yidx,
            registered_max_v_xidx,
            max_v_yidx,
            max_v_xidx,
        )


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def randomname(n) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# http://marupeke296.com/COL_3D_No27_CapsuleCapsule.html
# 点と直線の最短距離
# p : 点
# l : 直線
# h : 点から下ろした垂線の足（戻り値）
# t :ベクトル係数（戻り値）
# 戻り値: 最短距離
def calc_point_line_dist(point: MPoint, line: MLine):
    v_length_square = line.vector_real.lengthSquared()
    t = 0.0
    if v_length_square > 0.0:
        t = MVector3D.dotProduct(line.vector, (point.point - line.point.point).normalized()) / v_length_square

    h = line.point.point + line.vector * t
    length = (h - point.point).length()
    return length, h, t


# ∠p1p2p3は鋭角?
def is_sharp_angle(p1: MPoint, p2: MPoint, p3: MPoint):
    return MVector3D.dotProduct((p1.point - p2.point).normalized(), (p3.point - p2.point).normalized()) > 0


# 点と線分の最短距離
# p : 点
# seg : 線分
# h : 最短距離となる端点（戻り値）
# t : 端点位置（ t < 0: 始点の外, 0 <= t <= 1: 線分内, t > 1: 終点の外 ）
# 戻り値: 最短距離
def calc_point_segment_dist(point: MPoint, segment: MSegment):

    end_point = MPoint(segment.vector_end)

    # 垂線の長さ、垂線の足の座標及びtを算出
    min_length, h, t = calc_point_line_dist(point, MLine(segment.point, end_point.point - segment.point.point))

    if not is_sharp_angle(point, segment.point, end_point):
        # 始点側の外側
        min_length = (segment.point.point - point.point).length()
        h = segment.point.point
    elif not is_sharp_angle(point, end_point, segment.point):
        # 終点側の外側
        min_length = (end_point.point - point.point).length()
        h = end_point.point

    return min_length, h, t


# 2直線の最短距離
# l1 : L1
# l2 : L2
# p1 : L1側の垂線の足（戻り値）
# p2 : L2側の垂線の足（戻り値）
# t1 : L1側のベクトル係数（戻り値）
# t2 : L2側のベクトル係数（戻り値）
# 戻り値: 最短距離
def calc_line_line_dist(l1: MLine, l2: MLine):

    # 2直線が平行？
    if np.isclose(MVector3D.dotProduct(l1.vector, l2.vector), 1):
        # 点P11と直線L2の最短距離の問題に帰着
        min_length, p2, t2 = calc_point_line_dist(l1.point, l2)
        return min_length, l1.point.point, p2, 0.0, t2

    # 2直線はねじれ関係
    DV1V2 = MVector3D.dotProduct(l1.vector_real, l2.vector_real)
    DV1V1 = l1.vector_real.lengthSquared()
    DV2V2 = l2.vector_real.lengthSquared()
    P21P11 = (l1.point.point - l2.point.point).normalized()

    if (DV1V1 * DV2V2 - DV1V2 * DV1V2) == 0:
        min_length, p2, t2 = calc_point_line_dist(l1.point, l2)
        return min_length, l1.point.point, p2, 0.0, t2

    t1 = (
        DV1V2 * MVector3D.dotProduct(l2.vector_real, P21P11) - DV2V2 * MVector3D.dotProduct(l1.vector_real, P21P11)
    ) / (DV1V1 * DV2V2 - DV1V2 * DV1V2)
    p1 = l1.get_point(t1)
    t2 = MVector3D.dotProduct(l2.vector_real, (p1 - l2.point.point)) / DV2V2
    p2 = l2.get_point(t2)
    length = (p2 - p1).length()
    length2 = (l1.point.point - l2.point.point).length()

    if length > length2:
        # 仮対応
        return length2, l1.point.point, l2.point.point, 0.5, 0.5

    return length, p1, p2, t1, t2


# 2線分の最短距離
# s1 : S1(線分1)
# s2 : S2(線分2)
# p1 : S1側の垂線の足（戻り値）
# p2 : S2側の垂線の足（戻り値）
# t1 : S1側のベクトル係数（戻り値）
# t2 : S2側のベクトル係数（戻り値）
# 戻り値: 最短距離
def calc_segment_segment_dist(s1: MSegment, s2: MSegment):

    # S1が縮退している？
    if s1.vector_real.lengthSquared() < 0.00001:
        # S2も縮退？
        if s2.vector_real.lengthSquared() < 0.00001:
            # 点と点の距離の問題に帰着
            return (s2.point.point - s1.point.point).length(), s1.point.point, s2.point.point, 0, 0
        else:
            # S1の始点とS2の最短問題に帰着
            min_length, p2, t2 = calc_point_segment_dist(s1.point, s2)
            return min_length, s1.point.point, p2, 0.0, max(0, min(1, t2))

    # S2が縮退している？
    elif s2.vector_real.lengthSquared() < 0.00001:
        # S2の始点とS1の最短問題に帰着
        min_length, p1, t1 = calc_point_segment_dist(s2.point, s1)
        return min_length, p1, s2.point.point, max(0, min(1, t1)), 0.0

    # 線分同士 ------

    # 2線分が平行だったら垂線の端点の一つをP1に仮決定
    if np.isclose(MVector3D.dotProduct(s1.vector, s2.vector), 1):
        min_length, p2, t2 = calc_point_segment_dist(s1.point, s2)
        t1 = 0
        p1 = s1.point
        if 0 <= t2 <= 1:
            return min_length, p1, p2, t1, t2
    else:
        # 線分はねじれの関係
        # 2直線間の最短距離を求めて仮のt1,t2を求める
        min_length, p1, p2, t1, t2 = calc_line_line_dist(s1, s2)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return min_length, p1, p2, t1, t2

    # 垂線の足が外にある事が判明
    # S1側のt1を0～1の間にクランプして垂線を降ろす
    t1 = max(0, min(1, t1))
    p1 = s1.get_point(t1)
    min_length, p2, t2 = calc_point_segment_dist(MPoint(p1), s2)
    if 0 <= t2 <= 1:
        return min_length, p1, p2, t1, t2

    # S2側が外だったのでS2側をクランプ、S1に垂線を降ろす
    t2 = max(0, min(1, t2))
    p2 = s2.get_point(t2)
    min_length, p1, t1 = calc_point_segment_dist(MPoint(p2), s1)
    if 0 <= t1 <= 1:
        return min_length, p1, p2, t1, t2

    # 双方の端点が最短と判明
    t1 = max(0, min(1, t1))
    p1 = s1.get_point(t1)
    return (p2 - p1).length(), p1, p2, t1, t2


# カプセル同士の衝突判定
# c1 : S1(線分1)
# c2 : S2(線分2)
# 戻り値: 衝突していたらtrue
def is_col_capsule_capsule(c1: MCapsule, c2: MCapsule):
    distance, p1, p2, t1, t2 = calc_segment_segment_dist(c1.segment, c2.segment)
    return distance <= c1.radius + c2.radius


# # https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
# def calc_intersect(vP0: MVector3D, vP1: MVector3D, vQ0: MVector3D, vQ1: MVector3D) -> MVector3D:
#     P0 = (vQ0 - vP0).data().reshape(1, 3)
#     P1 = (vQ1 - vP1).data().reshape(1, 3)

#     """P0 and P1 are NxD arrays defining N lines.
#     D is the dimension of the space. This function
#     returns the least squares intersection of the N
#     lines from the system given by eq. 13 in
#     http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
#     """
#     # generate all line direction vectors
#     n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized

#     # generate the array of all projectors
#     projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
#     # see fig. 1

#     # generate R matrix and q vector
#     R = projs.sum(axis=0)
#     q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)

#     # solve the least squares problem for the
#     # intersection point p: Rp = q
#     p = np.linalg.lstsq(R, q, rcond=None)[0]

#     return MVector3D(p[0][0], p[1][0], p[2][0]) + vP0


# # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
# def calc_intersect(P0: MVector3D, P1: MVector3D, Q0: MVector3D, Q1: MVector3D) -> MVector3D:
#     u = P1 - P0

#     dot = MVector3D.dotProduct(Q1, u)
#     if np.isclose(dot, 0):
#         return None

#     # The factor of the point between p0 -> p1 (0 - 1)
#     # if 'fac' is between (0 - 1) the point intersects with the segment.
#     # Otherwise:
#     #  < 0.0: behind p0.
#     #  > 1.0: infront of p1.
#     w = P0 - Q0
#     fac = -MVector3D.dotProduct(Q1, w) / dot
#     x = u * fac

#     return x, fac


# https://stackoverflow.com/questions/34602761/intersecting-3d-lines
def calc_intersect(P0: MVector3D, P1: MVector3D, Q0: MVector3D, Q1: MVector3D) -> tuple:
    # Direction vectors
    DP = (P1 - P0).normalized()
    DQ = (Q1 - Q0).normalized()

    # start difference vector
    PQ = (Q0 - P0).normalized()

    # Find values
    a = MVector3D.dotProduct(DP, DP)
    b = MVector3D.dotProduct(DP, DQ)
    c = MVector3D.dotProduct(DQ, DQ)
    d = MVector3D.dotProduct(DP, PQ)
    e = MVector3D.dotProduct(DQ, PQ)

    # Find discriminant
    DD = a * c - b * b

    # segments are parallel, and consider special case of (partial) coincidence
    if np.isclose(DD, 0):
        return MVector3D(), -1, -1, -1

    # Find parameters for the closest points on lines
    tt = (b * e - c * d) / DD
    uu = (a * e - b * d) / DD

    Pt = P0 + (DP * tt)
    Qu = Q0 + (DQ * uu)

    length = Pt.distanceToPoint(Qu)

    # # If any parameter is out of range 0..1, then segments don't intersect
    # if not (0 <= tt <= 1 and 0 <= uu <= 1):
    #     return Pt, tt, uu, np.iinfo(np.int32).max

    return Pt, tt, uu, length


def read_vertices_from_file(file_path: str, model: PmxModel, material_name=None):
    target_vertices = []

    # 材質が指定されている場合、範囲を限定(指定されてなければ全頂点対象)
    material_vertices = model.material_vertices[material_name] if material_name else list(model.vertex_dict.keys())

    for encoding in ["cp932", "utf-8"]:
        # PmxEditor0.2.7.3以降は出力ファイルがUTF-8なので、両方試す
        # フォーマットは汎用固定とする
        try:
            with open(file_path, encoding=encoding, mode="r") as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーを読み飛ばす
                for row in reader:
                    if len(row) > 1 and int(row[1]) in material_vertices:
                        target_vertices.append(int(row[1]))
            # 無事頂点を読み終えたら終了
            break
        except Exception:
            continue

    return target_vertices


# area of polygon poly
def poly_area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]], [1, b[1], b[2]], [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]], [b[0], 1, b[2]], [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2) ** 0.5
    return (x / magnitude, y / magnitude, z / magnitude)


SEMI_STANDARD_BONE_NAMES = [
    "全ての親",
    "センター",
    "グルーブ",
    "腰",
    "下半身",
    "上半身",
    "上半身2",
    "上半身3",
    "首",
    "頭",
    "両目",
    "左目",
    "右目",
    "左胸",
    "左胸先",
    "右胸",
    "右胸先",
    "左肩P",
    "左肩",
    "左肩C",
    "左腕",
    "左腕捩",
    "左腕捩1",
    "左腕捩2",
    "左腕捩3",
    "左ひじ",
    "左手捩",
    "左手捩1",
    "左手捩2",
    "左手捩3",
    "左手首",
    "左親指０",
    "左親指１",
    "左親指２",
    "左親指先",
    "左人指１",
    "左人指２",
    "左人指３",
    "左人指先",
    "左中指１",
    "左中指２",
    "左中指３",
    "左中指先",
    "左薬指１",
    "左薬指２",
    "左薬指３",
    "左薬指先",
    "左小指１",
    "左小指２",
    "左小指３",
    "左小指先",
    "右肩P",
    "右肩",
    "右肩C",
    "右腕",
    "右腕捩",
    "右腕捩1",
    "右腕捩2",
    "右腕捩3",
    "右ひじ",
    "右手捩",
    "右手捩1",
    "右手捩2",
    "右手捩3",
    "右手首",
    "右親指０",
    "右親指１",
    "右親指２",
    "右親指先",
    "右人指１",
    "右人指２",
    "右人指３",
    "右人指先",
    "右中指１",
    "右中指２",
    "右中指３",
    "右中指先",
    "右薬指１",
    "右薬指２",
    "右薬指３",
    "右薬指先",
    "右小指１",
    "右小指２",
    "右小指３",
    "右小指先",
    "腰キャンセル左",
    "左足",
    "左ひざ",
    "左足首",
    "左つま先",
    "左足IK親",
    "左足ＩＫ",
    "左つま先ＩＫ",
    "腰キャンセル右",
    "右足",
    "右ひざ",
    "右足首",
    "右つま先",
    "右足IK親",
    "右足ＩＫ",
    "右つま先ＩＫ",
    "左足D",
    "左ひざD",
    "左足首D",
    "左足先EX",
    "右足D",
    "右ひざD",
    "右足首D",
    "右足先EX",
]
