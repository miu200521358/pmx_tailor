# -*- coding: utf-8 -*-
#
from cmath import isnan
import logging
import os
import traceback
import numpy as np
import itertools
import math
import copy
import bezier
import csv
import random
import string
from collections import Counter

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
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4
from utils.MLogger import MLogger
from utils.MException import SizingException, MKilledException
import utils.MBezierUtils as MBezierUtils

logger = MLogger(__name__, level=1)


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
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('密集度')}: {param_option['threshold']}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('細かさ')}: {param_option['fineness']}"
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('質量')}: {param_option['mass']}"
                service_data_txt = (
                    f"{service_data_txt}\n　　{logger.transtext('柔らかさ')}: {param_option['air_resistance']}"
                )
                service_data_txt = (
                    f"{service_data_txt}\n　　{logger.transtext('張り')}: {param_option['shape_maintenance']}"
                )

            logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

            model = self.options.pmx_model
            model.comment += f"\r\n\r\n{logger.transtext('物理')}: PmxTailor"

            # 保持ボーンは全設定を確認する
            saved_bone_names = self.get_saved_bone_names(model)

            for pidx, param_option in enumerate(self.options.param_options):
                if not self.create_physics(model, param_option, saved_bone_names):
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

    def get_saved_bone_names(self, model: PmxModel):
        # TODO
        return []

    def create_physics(self, model: PmxModel, param_option: dict, saved_bone_names: list):
        model.comment += f"\r\n{logger.transtext('材質')}: {param_option['material_name']} --------------"
        model.comment += f"\r\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"
        model.comment += f", {logger.transtext('細かさ')}: {param_option['fineness']}"
        model.comment += f", {logger.transtext('質量')}: {param_option['mass']}"
        model.comment += f", {logger.transtext('柔らかさ')}: {param_option['air_resistance']}"
        model.comment += f", {logger.transtext('張り')}: {param_option['shape_maintenance']}"

        material_name = param_option["material_name"]

        # 頂点CSVが指定されている場合、対象頂点リスト生成
        if param_option["vertices_csv"]:
            target_vertices = []
            try:
                with open(param_option["vertices_csv"], encoding="cp932", mode="r") as f:
                    reader = csv.reader(f)
                    next(reader)  # ヘッダーを読み飛ばす
                    for row in reader:
                        if len(row) > 1 and int(row[1]) in model.material_vertices[material_name]:
                            target_vertices.append(int(row[1]))
            except Exception:
                logger.warning("頂点CSVが正常に読み込めなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
                return None, None
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
            virtual_vertices, vertex_maps, all_regist_bones, all_bone_connected, root_bone = self.create_bone_map(
                model, param_option, material_name, target_vertices, is_root_bone=False
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
                virtual_vertices, vertex_maps, all_regist_bones, all_bone_connected, root_bone = self.create_bone_map(
                    model, param_option, material_name, target_vertices
                )
            else:
                vertex_maps, virtual_vertices, remaining_vertices, back_vertices = self.create_vertex_map(
                    model,
                    param_option,
                    material_name,
                    target_vertices,
                    base_vertical_axis,
                    base_reverse_axis,
                    target_idx,
                )

                if not vertex_maps:
                    return False

                # 各頂点の有効INDEX数が最も多いものをベースとする
                map_cnt = []
                for vertex_map in vertex_maps:
                    map_cnt.append(np.count_nonzero(~np.isnan(vertex_map)) / 3)

                if len(map_cnt) == 0:
                    logger.warning("有効な頂点マップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
                    return False

                vertex_map_orders = [k for k in np.argsort(-np.array(map_cnt)) if map_cnt[k] > np.max(map_cnt) * 0.5]

                (
                    root_bone,
                    virtual_vertices,
                    all_regist_bones,
                    all_bone_vertical_distances,
                    all_bone_horizonal_distances,
                    all_bone_connected,
                ) = self.create_bone(
                    model, param_option, material_name, virtual_vertices, vertex_maps, vertex_map_orders
                )

                remaining_vertices = self.create_weight(
                    model,
                    param_option,
                    material_name,
                    virtual_vertices,
                    vertex_maps,
                    all_regist_bones,
                    all_bone_vertical_distances,
                    all_bone_horizonal_distances,
                    all_bone_connected,
                    remaining_vertices,
                )

                # 残ウェイト
                self.create_remaining_weight(model, param_option, material_name, virtual_vertices, remaining_vertices)

                # 裏ウェイト
                self.create_back_weight(model, param_option, material_name, virtual_vertices, back_vertices)

            root_rigidbody = self.create_rigidbody(
                model,
                param_option,
                material_name,
                virtual_vertices,
                vertex_maps,
                all_regist_bones,
                all_bone_connected,
                root_bone,
                base_reverse_axis,
            )

            self.create_joint(
                model,
                param_option,
                material_name,
                virtual_vertices,
                vertex_maps,
                all_regist_bones,
                all_bone_connected,
                root_rigidbody,
            )

        return True

    def create_joint(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        all_regist_bones: dict,
        all_bone_connected: dict,
        root_rigidbody: RigidBody,
    ):
        logger.info("【%s】ジョイント生成", material_name, decoration=MLogger.DECORATION_LINE)

        # ジョイント生成
        created_joints = {}
        prev_joint_cnt = 0

        for base_map_idx, regist_bones in all_regist_bones.items():
            logger.info("--【No.%s】ジョイント生成", base_map_idx + 1)

            vertex_map = vertex_maps[base_map_idx]

            # 上下はY軸比較, 左右はX軸比較
            target_idx = 1 if param_option["direction"] in ["上", "下"] else 0
            target_direction = 1 if param_option["direction"] in ["上", "右"] else -1

            # キーは比較対象＋向きで昇順
            vv_keys = sorted(np.unique(vertex_map[np.where(regist_bones)][:, target_idx]) * target_direction)

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

            # 逆ジョイント情報
            (
                reverse_limit_min_mov_xs,
                reverse_limit_min_mov_ys,
                reverse_limit_min_mov_zs,
                reverse_limit_max_mov_xs,
                reverse_limit_max_mov_ys,
                reverse_limit_max_mov_zs,
                reverse_limit_min_rot_xs,
                reverse_limit_min_rot_ys,
                reverse_limit_min_rot_zs,
                reverse_limit_max_rot_xs,
                reverse_limit_max_rot_ys,
                reverse_limit_max_rot_zs,
                reverse_spring_constant_mov_xs,
                reverse_spring_constant_mov_ys,
                reverse_spring_constant_mov_zs,
                reverse_spring_constant_rot_xs,
                reverse_spring_constant_rot_ys,
                reverse_spring_constant_rot_zs,
            ) = self.create_joint_param(
                param_option["reverse_joint"], vv_keys, param_option["reverse_joint_coefficient"]
            )

            for v_yidx, v_xidx in zip(np.where(regist_bones)[0], np.where(regist_bones)[1]):
                bone_key = tuple(vertex_map[v_yidx, v_xidx])
                vv = virtual_vertices.get(bone_key, None)

                if not vv:
                    logger.warning("ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s", bone_key)
                    continue

                if not vv.map_rigidbodies.get(base_map_idx, None):
                    # 剛体はくっついてない場合があるので、その場合はワーニングは出さずにスルー
                    continue

                bone_y_idx = np.where(vv_keys == bone_key[target_idx] * target_direction)[0]
                bone_y_idx = bone_y_idx[0] if bone_y_idx else 0

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
                    max_v_yidx,
                    max_v_xidx,
                ) = self.get_block_vidxs(
                    v_yidx, v_xidx, vertex_maps, all_regist_bones, all_bone_connected, base_map_idx
                )

                prev_above_vv = virtual_vertices.get(tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx]), None)
                now_prev_vv = virtual_vertices.get(tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx]), None)
                now_above_vv = virtual_vertices.get(tuple(vertex_map[above_yidx, v_xidx]), None)
                now_below_vv = virtual_vertices.get(tuple(vertex_map[below_yidx, v_xidx]), None)

                if param_option["vertical_joint"] and now_above_vv:
                    # 縦ジョイント
                    if v_yidx == 0:
                        a_rigidbody = root_rigidbody
                        b_rigidbody = vv.map_rigidbodies[base_map_idx]

                        joint_pos = vv.map_bones[base_map_idx].position

                        # 剛体進行方向(x) 中心剛体との角度は反映させない
                        tail_vv = now_below_vv
                    else:
                        a_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)
                        b_rigidbody = vv.map_rigidbodies.get(base_map_idx, None)

                        # 剛体が重なる箇所の交点
                        above_mat = MMatrix4x4()
                        above_point = MVector3D()
                        if a_rigidbody:
                            above_mat.setToIdentity()
                            above_mat.translate(a_rigidbody.shape_position)
                            above_mat.rotate(a_rigidbody.shape_qq)
                            above_point = above_mat * MVector3D(-a_rigidbody.shape_size.y(), 0, 0)

                        now_mat = MMatrix4x4()
                        now_mat.setToIdentity()
                        now_mat.translate(b_rigidbody.shape_position)
                        now_mat.rotate(b_rigidbody.shape_qq)
                        now_point = now_mat * MVector3D(b_rigidbody.shape_size.y(), 0, 0)

                        joint_pos = (above_point + now_point) / 2

                        tail_vv = now_above_vv

                    if (
                        not a_rigidbody
                        or not b_rigidbody
                        or (a_rigidbody.name == b_rigidbody.name)
                        or a_rigidbody.index < 0
                        or b_rigidbody.index < 0
                    ):
                        logger.warning(
                            "縦ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                            vv.map_bones[base_map_idx].name if vv.map_bones.get(base_map_idx, None) else vv.vidxs(),
                        )
                    else:
                        # 剛体進行方向(x)
                        x_direction_pos = (
                            vv.map_bones[base_map_idx].position - tail_vv.map_bones[base_map_idx].position
                        ).normalized()
                        # 剛体進行方向に対しての縦軸(y)
                        y_direction_pos = (
                            (vv.normal().normalized() + tail_vv.normal().normalized()) / 2
                        ).normalized() * -1
                        joint_qq = MQuaternion.fromDirection(y_direction_pos, x_direction_pos)

                        joint_key, joint = self.build_joint(
                            "↓",
                            0,
                            bone_y_idx,
                            a_rigidbody,
                            b_rigidbody,
                            joint_pos,
                            joint_qq,
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
                        )
                        created_joints[joint_key] = joint

                        if param_option["reverse_joint"]:
                            # 逆ジョイント
                            if v_yidx == 0:
                                a_rigidbody = vv.map_rigidbodies[base_map_idx]
                                b_rigidbody = root_rigidbody

                                # 剛体進行方向(x) 中心剛体との角度は反映させない
                                tail_vv = now_below_vv
                            else:
                                a_rigidbody = vv.map_rigidbodies.get(base_map_idx, None)
                                b_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)

                                tail_vv = now_above_vv

                            # 剛体進行方向(x)
                            x_direction_pos = (
                                tail_vv.map_bones[base_map_idx].position - vv.map_bones[base_map_idx].position
                            ).normalized()
                            # 剛体進行方向に対しての縦軸(y)
                            y_direction_pos = (
                                (vv.normal().normalized() + tail_vv.normal().normalized()) / 2
                            ).normalized() * -1
                            joint_qq = MQuaternion.fromDirection(y_direction_pos, x_direction_pos)

                            joint_key, joint = self.build_joint(
                                "↑",
                                1,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                reverse_limit_min_mov_xs,
                                reverse_limit_min_mov_ys,
                                reverse_limit_min_mov_zs,
                                reverse_limit_max_mov_xs,
                                reverse_limit_max_mov_ys,
                                reverse_limit_max_mov_zs,
                                reverse_limit_min_rot_xs,
                                reverse_limit_min_rot_ys,
                                reverse_limit_min_rot_zs,
                                reverse_limit_max_rot_xs,
                                reverse_limit_max_rot_ys,
                                reverse_limit_max_rot_zs,
                                reverse_spring_constant_mov_xs,
                                reverse_spring_constant_mov_ys,
                                reverse_spring_constant_mov_zs,
                                reverse_spring_constant_rot_xs,
                                reverse_spring_constant_rot_ys,
                                reverse_spring_constant_rot_zs,
                            )
                            created_joints[joint_key] = joint

                        # バランサー剛体が必要な場合
                        if param_option["rigidbody_balancer"]:
                            a_rigidbody = vv.map_rigidbodies[base_map_idx]
                            b_rigidbody = vv.map_balance_rigidbodies[base_map_idx]

                            if not (
                                not a_rigidbody
                                or not b_rigidbody
                                or (a_rigidbody.name == b_rigidbody.name)
                                or a_rigidbody.index < 0
                                or b_rigidbody.index < 0
                            ):
                                joint_axis_up = (
                                    now_below_vv.map_bones[base_map_idx].position - vv.map_bones[base_map_idx].position
                                ).normalized()
                                joint_axis = (
                                    vv.map_balance_rigidbodies[base_map_idx].shape_position
                                    - vv.map_bones[base_map_idx].position
                                ).normalized()
                                joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                                joint_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)

                                joint_key, joint = self.build_joint(
                                    "B",
                                    8,
                                    0,
                                    a_rigidbody,
                                    b_rigidbody,
                                    a_rigidbody.shape_position.copy(),
                                    MQuaternion(),
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
                                created_joints[joint_key] = joint

                                if now_below_vv.map_balance_rigidbodies.get(base_map_idx, None):
                                    # バランサー補助剛体
                                    joint_key, joint = self.build_joint(
                                        "BS",
                                        9,
                                        0,
                                        vv.map_balance_rigidbodies[base_map_idx],
                                        now_below_vv.map_balance_rigidbodies[base_map_idx],
                                        MVector3D(),
                                        MQuaternion(),
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
                                    created_joints[joint_key] = joint

                if param_option["horizonal_joint"] and prev_connected and now_prev_vv:
                    # 横ジョイント
                    a_rigidbody = now_prev_vv.map_rigidbodies.get(prev_map_idx, None)
                    b_rigidbody = vv.map_rigidbodies.get(base_map_idx, None)

                    if (
                        not a_rigidbody
                        or not b_rigidbody
                        or (a_rigidbody.name == b_rigidbody.name)
                        or a_rigidbody.index < 0
                        or b_rigidbody.index < 0
                    ):
                        logger.warning(
                            "横ジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                            vv.map_bones[base_map_idx].name if vv.map_bones.get(base_map_idx, None) else vv.vidxs(),
                        )
                    else:
                        # 剛体が重なる箇所の交点
                        now_mat = MMatrix4x4()
                        now_mat.setToIdentity()
                        now_mat.translate(a_rigidbody.shape_position)
                        now_mat.rotate(a_rigidbody.shape_qq)
                        now_point = now_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                        next_mat = MMatrix4x4()
                        next_mat.setToIdentity()
                        next_mat.translate(b_rigidbody.shape_position)
                        next_mat.rotate(b_rigidbody.shape_qq)
                        now_next_point = next_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                        joint_pos = (now_point + now_next_point) / 2
                        joint_qq = MQuaternion.slerp(a_rigidbody.shape_qq, b_rigidbody.shape_qq, 0.5)

                        joint_key, joint = self.build_joint(
                            "→",
                            2,
                            bone_y_idx,
                            a_rigidbody,
                            b_rigidbody,
                            joint_pos,
                            joint_qq,
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
                        )
                        created_joints[joint_key] = joint

                        if param_option["reverse_joint"]:
                            # 逆ジョイント
                            a_rigidbody = vv.map_rigidbodies.get(base_map_idx, None)
                            b_rigidbody = now_prev_vv.map_rigidbodies.get(prev_map_idx, None)

                            joint_key, joint = self.build_joint(
                                "←",
                                3,
                                bone_y_idx,
                                a_rigidbody,
                                b_rigidbody,
                                joint_pos,
                                joint_qq,
                                reverse_limit_min_mov_xs,
                                reverse_limit_min_mov_ys,
                                reverse_limit_min_mov_zs,
                                reverse_limit_max_mov_xs,
                                reverse_limit_max_mov_ys,
                                reverse_limit_max_mov_zs,
                                reverse_limit_min_rot_xs,
                                reverse_limit_min_rot_ys,
                                reverse_limit_min_rot_zs,
                                reverse_limit_max_rot_xs,
                                reverse_limit_max_rot_ys,
                                reverse_limit_max_rot_zs,
                                reverse_spring_constant_mov_xs,
                                reverse_spring_constant_mov_ys,
                                reverse_spring_constant_mov_zs,
                                reverse_spring_constant_rot_xs,
                                reverse_spring_constant_rot_ys,
                                reverse_spring_constant_rot_zs,
                            )
                            created_joints[joint_key] = joint

                if param_option["diagonal_joint"] and prev_connected and prev_above_vv:
                    # 斜めジョイント
                    a_rigidbody = prev_above_vv.map_rigidbodies.get(prev_map_idx, None)
                    b_rigidbody = vv.map_rigidbodies.get(base_map_idx, None)

                    if (
                        not a_rigidbody
                        or not b_rigidbody
                        or (a_rigidbody.name == b_rigidbody.name)
                        or a_rigidbody.index < 0
                        or b_rigidbody.index < 0
                        or v_yidx == above_yidx
                    ):
                        if v_yidx != above_yidx:
                            logger.warning(
                                "斜めジョイント生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                                vv.map_bones[base_map_idx].name
                                if vv.map_bones.get(base_map_idx, None)
                                else vv.vidxs(),
                            )
                    else:
                        # 剛体が重なる箇所の交点
                        now_mat = MMatrix4x4()
                        now_mat.setToIdentity()
                        now_mat.translate(a_rigidbody.shape_position)
                        now_mat.rotate(a_rigidbody.shape_qq)
                        now_point = now_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                        next_mat = MMatrix4x4()
                        next_mat.setToIdentity()
                        next_mat.translate(b_rigidbody.shape_position)
                        next_mat.rotate(b_rigidbody.shape_qq)
                        now_next_point = next_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                        joint_pos = (now_point + now_next_point) / 2
                        joint_qq = MQuaternion.slerp(a_rigidbody.shape_qq, b_rigidbody.shape_qq, 0.5)

                        joint_key, joint = self.build_joint(
                            "＼",
                            4,
                            bone_y_idx,
                            a_rigidbody,
                            b_rigidbody,
                            joint_pos,
                            joint_qq,
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
                        )
                        created_joints[joint_key] = joint

                    a_rigidbody = now_prev_vv.map_rigidbodies.get(prev_map_idx, None)
                    b_rigidbody = now_above_vv.map_rigidbodies.get(base_map_idx, None)

                    if not (
                        not a_rigidbody
                        or not b_rigidbody
                        or (a_rigidbody.name == b_rigidbody.name)
                        or a_rigidbody.index < 0
                        or b_rigidbody.index < 0
                        or v_yidx == above_yidx
                    ):
                        # 剛体が重なる箇所の交点
                        now_mat = MMatrix4x4()
                        now_mat.setToIdentity()
                        now_mat.translate(a_rigidbody.shape_position)
                        now_mat.rotate(a_rigidbody.shape_qq)
                        now_point = now_mat * MVector3D(a_rigidbody.shape_size.x(), 0, 0)

                        next_mat = MMatrix4x4()
                        next_mat.setToIdentity()
                        next_mat.translate(b_rigidbody.shape_position)
                        next_mat.rotate(b_rigidbody.shape_qq)
                        now_next_point = next_mat * MVector3D(-b_rigidbody.shape_size.x(), 0, 0)

                        joint_pos = (now_point + now_next_point) / 2
                        joint_qq = MQuaternion.slerp(a_rigidbody.shape_qq, b_rigidbody.shape_qq, 0.5)

                        joint_key, joint = self.build_joint(
                            "／",
                            5,
                            bone_y_idx,
                            a_rigidbody,
                            b_rigidbody,
                            joint_pos,
                            joint_qq,
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
                        )
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
            logger.debug(f"joint: {joint}")

    def build_joint(
        self,
        direction_mark: str,
        direction_idx: int,
        bone_y_idx: int,
        a_rigidbody: RigidBody,
        b_rigidbody: RigidBody,
        joint_pos: MVector3D,
        joint_qq: MQuaternion,
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
    ):
        joint_name = f"{direction_mark}|{a_rigidbody.name}|{b_rigidbody.name}"
        joint_key = f"{direction_idx}:{a_rigidbody.index:09d}:{b_rigidbody.index:09d}"

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
                limit_min_mov_xs[bone_y_idx],
                limit_min_mov_ys[bone_y_idx],
                limit_min_mov_zs[bone_y_idx],
            ),
            MVector3D(
                limit_max_mov_xs[bone_y_idx],
                limit_max_mov_ys[bone_y_idx],
                limit_max_mov_zs[bone_y_idx],
            ),
            MVector3D(
                math.radians(limit_min_rot_xs[bone_y_idx]),
                math.radians(limit_min_rot_ys[bone_y_idx]),
                math.radians(limit_min_rot_zs[bone_y_idx]),
            ),
            MVector3D(
                math.radians(limit_max_rot_xs[bone_y_idx]),
                math.radians(limit_max_rot_ys[bone_y_idx]),
                math.radians(limit_max_rot_zs[bone_y_idx]),
            ),
            MVector3D(
                spring_constant_mov_xs[bone_y_idx],
                spring_constant_mov_ys[bone_y_idx],
                spring_constant_mov_zs[bone_y_idx],
            ),
            MVector3D(
                spring_constant_rot_xs[bone_y_idx],
                spring_constant_rot_ys[bone_y_idx],
                spring_constant_rot_zs[bone_y_idx],
            ),
        )
        return joint_key, joint

    def create_joint_param(self, param_joint: Joint, vv_keys: np.ndarray, coefficient: float):
        max_vy = len(vv_keys)
        middle_vy = max_vy * 0.3
        min_vy = 0
        xs = np.arange(min_vy, max_vy, step=1)

        limit_min_mov_xs = 0
        limit_min_mov_ys = 0
        limit_min_mov_zs = 0
        limit_max_mov_xs = 0
        limit_max_mov_ys = 0
        limit_max_mov_zs = 0
        limit_min_rot_xs = 0
        limit_min_rot_ys = 0
        limit_min_rot_zs = 0
        limit_max_rot_xs = 0
        limit_max_rot_ys = 0
        limit_max_rot_zs = 0
        spring_constant_mov_xs = 0
        spring_constant_mov_ys = 0
        spring_constant_mov_zs = 0
        spring_constant_rot_xs = 0
        spring_constant_rot_ys = 0
        spring_constant_rot_zs = 0

        if param_joint:
            # 縦ジョイント
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
        all_regist_bones: dict,
        all_bone_connected: dict,
        root_bone: Bone,
        base_reverse_axis: MVector3D,
    ):
        logger.info("【%s】剛体生成", material_name, decoration=MLogger.DECORATION_LINE)

        # 剛体生成
        created_rigidbodies = {}
        created_rigidbody_vvs = {}
        # 剛体の質量
        created_rigidbody_masses = {}
        created_rigidbody_linear_dampinges = {}
        created_rigidbody_angular_dampinges = {}
        prev_rigidbody_cnt = 0

        # 剛体情報
        param_rigidbody = param_option["rigidbody"]
        # 剛体係数
        coefficient = param_option["rigidbody_coefficient"]
        # 剛体形状
        rigidbody_shape_type = param_option["rigidbody_shape_type"]

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
                parent_bone_rigidbody.shape_position,
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

        for base_map_idx, regist_bones in all_regist_bones.items():
            logger.info("--【No.%s】剛体生成", base_map_idx + 1)

            vertex_map = vertex_maps[base_map_idx]

            # 厚みの判定

            # 上下はY軸比較, 左右はX軸比較
            target_idx = 1 if param_option["direction"] in ["上", "下"] else 0
            target_direction = 1 if param_option["direction"] in ["上", "右"] else -1

            # キーは比較対象＋向きで昇順
            vv_keys = sorted(np.unique(vertex_map[np.where(regist_bones[:-1, :])][:, target_idx]) * target_direction)
            # 全体の比較対象の距離
            vv_distances = sorted(
                vertex_map[np.where(~np.isnan(vertex_map))].reshape(
                    int(np.where(~np.isnan(vertex_map))[0].shape[0] / 3), 3
                ),
                key=lambda x: x[target_idx],
            )
            distance = (
                (
                    virtual_vertices[tuple(vv_distances[0])].position()
                    - virtual_vertices[tuple(vv_distances[-1])].position()
                ).abs()
            ).data()[target_idx]
            # 厚みは比較キーの数分だけ作る
            rigidbody_limit_thicks = np.linspace(0.15, distance / 3 * 0.15, len(vv_keys))

            for v_yidx, v_xidx in zip(np.where(regist_bones[:-1, :])[0], np.where(regist_bones[:-1, :])[1]):
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
                    max_v_yidx,
                    max_v_xidx,
                ) = self.get_block_vidxs(
                    v_yidx, v_xidx, vertex_maps, all_regist_bones, all_bone_connected, base_map_idx
                )

                prev_above_vv = virtual_vertices.get(tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx]), None)
                prev_now_vv = virtual_vertices.get(tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx]), None)
                prev_below_vv = virtual_vertices.get(tuple(vertex_maps[prev_map_idx][below_yidx, prev_xidx]), None)
                now_above_vv = virtual_vertices.get(tuple(vertex_map[v_yidx, v_xidx]), None)
                now_below_vv = virtual_vertices.get(tuple(vertex_map[below_yidx, v_xidx]), None)
                next_above_vv = virtual_vertices.get(tuple(vertex_maps[next_map_idx][v_yidx, next_xidx]), None)
                next_now_vv = virtual_vertices.get(tuple(vertex_maps[next_map_idx][v_yidx, next_xidx]), None)
                next_below_vv = virtual_vertices.get(tuple(vertex_maps[next_map_idx][below_yidx, next_xidx]), None)

                if not (
                    prev_now_vv
                    and prev_above_vv
                    and prev_below_vv
                    and now_above_vv
                    and now_below_vv
                    and next_above_vv
                    and next_now_vv
                    and next_below_vv
                ):
                    logger.warning(
                        "剛体生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                        vv.map_bones[base_map_idx].name if vv.map_bones.get(base_map_idx, None) else vv.vidxs(),
                    )
                    continue

                prev_above_bone = prev_above_vv.map_bones.get(prev_map_idx, None)
                prev_now_bone = prev_now_vv.map_bones.get(prev_map_idx, None)
                prev_below_bone = prev_below_vv.map_bones.get(prev_map_idx, None)
                now_above_bone = now_above_vv.map_bones.get(base_map_idx, None)
                now_below_bone = now_below_vv.map_bones.get(base_map_idx, None)
                next_above_bone = next_above_vv.map_bones.get(next_map_idx, None)
                next_now_bone = next_now_vv.map_bones.get(next_map_idx, None)
                next_below_bone = next_below_vv.map_bones.get(next_map_idx, None)

                if not (now_above_bone and now_below_bone and now_above_bone.index in model.vertices):
                    logger.warning(
                        "剛体生成に必要な情報が取得できなかった為、スルーします。 処理対象: %s",
                        vv.map_bones[base_map_idx].name if vv.map_bones.get(base_map_idx, None) else vv.vidxs(),
                    )
                    continue

                # 剛体の厚みINDEX
                rigidbody_y_idx = np.where(vv_keys == rigidbody_bone_key[target_idx] * target_direction)[0]
                if not rigidbody_y_idx:
                    rigidbody_y_idx = 0

                v_poses = [(v.position * base_reverse_axis).data() for v in model.vertices[now_above_bone.index]]

                # 前と繋がっている場合、前のボーンの頂点を追加する
                if prev_connected and prev_now_bone and prev_now_bone.index in model.vertices:
                    v_poses += [(v.position * base_reverse_axis).data() for v in model.vertices[prev_now_bone.index]]

                # 次と繋がっている場合、次のボーンの頂点を追加する
                if next_connected and next_now_bone and next_now_bone.index in model.vertices:
                    v_poses += [(v.position * base_reverse_axis).data() for v in model.vertices[next_now_bone.index]]

                # 重複は除外
                v_poses = np.unique(v_poses, axis=0)

                v_combs = np.array(list(itertools.product(v_poses, repeat=2)))
                x_size = np.max(np.linalg.norm(v_combs[:, 0] - v_combs[:, 1], ord=2, axis=1))
                y_size = np.linalg.norm(
                    now_above_bone.position.data() - now_below_bone.position.data(),
                    ord=2,
                )

                if rigidbody_shape_type == 0:
                    # 球剛体の場合
                    ball_size = max(0.25, np.mean([x_size, y_size]) * 0.5)
                    shape_size = MVector3D(ball_size, ball_size, ball_size)

                elif rigidbody_shape_type == 1:
                    # 箱剛体の場合
                    shape_size = MVector3D(
                        x_size * 0.27,
                        max(0.25, (y_size * 0.5)),
                        rigidbody_limit_thicks[rigidbody_y_idx],
                    )

                else:
                    # カプセル剛体の場合
                    shape_size = MVector3D(
                        x_size * 0.2,
                        max(0.25, (y_size * 0.85)),
                        rigidbody_limit_thicks[rigidbody_y_idx],
                    )

                if prev_connected and prev_now_bone and prev_below_bone and not next_connected:
                    # 後が繋がってない場合、前との中間とする
                    mean_position = MVector3D(
                        np.mean(
                            [
                                now_above_bone.position.data(),
                                now_below_bone.position.data(),
                                prev_now_bone.position.data(),
                                prev_below_bone.position.data(),
                            ],
                            axis=0,
                        )
                    )
                elif (
                    not prev_connected
                    and next_connected
                    and next_now_bone
                    or (next_connected and next_now_bone and param_option["special_shape"] == logger.transtext("プリーツ"))
                ):
                    # 前が繋がってない場合、次との中間とする
                    mean_position = MVector3D(
                        np.mean(
                            [
                                now_above_bone.position.data(),
                                now_below_bone.position.data(),
                                next_now_bone.position.data(),
                                next_below_bone.position.data(),
                            ],
                            axis=0,
                        )
                    )
                else:
                    # それ以外は自身の中間
                    mean_position = MVector3D(
                        np.mean(
                            [
                                now_above_bone.position.data(),
                                now_below_bone.position.data(),
                            ],
                            axis=0,
                        )
                    )

                # ボーン進行方向(x)
                x_direction_pos = (now_above_bone.position - now_below_bone.position).normalized()
                # ボーン進行方向に対しての横軸(y)
                if (
                    next_above_bone
                    and now_above_bone
                    and ((prev_above_bone and prev_above_bone != now_above_bone) or next_above_bone != now_above_bone)
                ):
                    y_direction_pos = (
                        next_above_bone.position
                        - (prev_above_bone.position if prev_above_bone else now_above_bone.position)
                    ).normalized()
                else:
                    y_direction_pos = MVector3D(1, 0, 0)
                # ボーン進行方向に対しての縦軸(z)
                z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
                shape_qq = MQuaternion.fromDirection(z_direction_pos * -1, x_direction_pos)
                shape_euler = shape_qq.toEulerAngles()
                shape_rotation_radians = MVector3D(
                    math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z())
                )

                mat = MMatrix4x4()
                mat.setToIdentity()
                mat.translate(mean_position)
                mat.rotate(shape_qq)

                # 身体の方にちょっと寄せる
                shape_position = mat * MVector3D(0, 0, 0.07)

                # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
                mode = 2 if 0 == v_yidx else 1
                mass = param_rigidbody.param.mass * shape_size.x() * shape_size.y() * shape_size.z()
                linear_damping = (
                    param_rigidbody.param.linear_damping * shape_size.x() * shape_size.y() * shape_size.z()
                )
                angular_damping = (
                    param_rigidbody.param.angular_damping * shape_size.x() * shape_size.y() * shape_size.z()
                )

                vv.map_rigidbodies[base_map_idx] = RigidBody(
                    vv.map_bones[base_map_idx].name,
                    vv.map_bones[base_map_idx].name,
                    vv.map_bones[base_map_idx].index,
                    param_rigidbody.collision_group,
                    param_rigidbody.no_collision_group,
                    rigidbody_shape_type,
                    shape_size,
                    shape_position,
                    shape_rotation_radians,
                    mass,
                    linear_damping,
                    angular_damping,
                    param_rigidbody.param.restitution,
                    param_rigidbody.param.friction,
                    mode,
                )
                vv.map_rigidbodies[base_map_idx].shape_qq = shape_qq

                # 別途保持しておく
                if vv.map_rigidbodies[base_map_idx].name not in created_rigidbodies:
                    if base_map_idx not in created_rigidbody_vvs:
                        created_rigidbody_vvs[base_map_idx] = {}
                    if v_xidx not in created_rigidbody_vvs[base_map_idx]:
                        created_rigidbody_vvs[base_map_idx][v_xidx] = {}
                    created_rigidbody_vvs[base_map_idx][v_xidx][v_yidx] = vv
                    created_rigidbodies[vv.map_rigidbodies[base_map_idx].name] = vv.map_rigidbodies[base_map_idx]
                    created_rigidbody_masses[vv.map_rigidbodies[base_map_idx].name] = mass
                    created_rigidbody_linear_dampinges[vv.map_rigidbodies[base_map_idx].name] = linear_damping
                    created_rigidbody_angular_dampinges[vv.map_rigidbodies[base_map_idx].name] = angular_damping
                else:
                    # 既に保持済みの剛体である場合、前のを参照する
                    vv.map_rigidbodies[base_map_idx] = created_rigidbodies[vv.map_rigidbodies[base_map_idx].name]

                if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 50 > prev_rigidbody_cnt:
                    logger.info("-- -- 【No.%s】剛体: %s個目:終了", base_map_idx + 1, len(created_rigidbodies))
                    prev_rigidbody_cnt = len(created_rigidbodies) // 50

        min_mass = 0
        min_linear_damping = 0
        min_angular_damping = 0

        max_mass = 0
        max_linear_damping = 0
        max_angular_damping = 0

        if len(created_rigidbody_masses.values()) > 0:
            min_mass = np.min(list(created_rigidbody_masses.values()))
            min_linear_damping = np.min(list(created_rigidbody_linear_dampinges.values()))
            min_angular_damping = np.min(list(created_rigidbody_angular_dampinges.values()))

            max_mass = np.max(list(created_rigidbody_masses.values()))
            max_linear_damping = np.max(list(created_rigidbody_linear_dampinges.values()))
            max_angular_damping = np.max(list(created_rigidbody_angular_dampinges.values()))

        for rigidbody_name in sorted(created_rigidbodies.keys()):
            # 剛体を登録
            rigidbody = created_rigidbodies[rigidbody_name]
            rigidbody.index = len(model.rigidbodies)

            # 質量と減衰は面積に応じた値に変換
            if min_mass != max_mass:
                rigidbody.param.mass = calc_ratio(
                    rigidbody.param.mass,
                    max_mass,
                    min_mass,
                    param_rigidbody.param.mass,
                    param_rigidbody.param.mass * coefficient,
                )
            if min_linear_damping != max_linear_damping:
                rigidbody.param.linear_damping = calc_ratio(
                    rigidbody.param.linear_damping,
                    max_linear_damping,
                    min_linear_damping,
                    param_rigidbody.param.linear_damping,
                    min(0.9999999, param_rigidbody.param.linear_damping * coefficient),
                )
            if min_angular_damping != max_angular_damping:
                rigidbody.param.angular_damping = calc_ratio(
                    rigidbody.param.angular_damping,
                    max_angular_damping,
                    min_angular_damping,
                    param_rigidbody.param.angular_damping,
                    min(0.9999999, param_rigidbody.param.angular_damping * coefficient),
                )

            if rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                rigidbody.name += randomname(3)

            model.rigidbodies[rigidbody.name] = rigidbody
            logger.debug(f"rigidbody: {rigidbody}")

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
                        org_tail_position = org_bone.tail_position
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
                logger.debug(f"rigidbody: {rigidbody}")

            logger.info("-- バランサー剛体: %s個目:終了", len(created_rigidbodies))

        return root_rigidbody

    def create_back_weight(
        self, model: PmxModel, param_option: dict, material_name: str, virtual_vertices: dict, back_vertices: list
    ):
        if param_option["back_material_name"]:
            # 表面で残った裏頂点と裏材質で指定されている頂点を全部対象とする
            back_vertices += list(model.material_vertices[param_option["back_material_name"]])

        if not back_vertices:
            return

        logger.info("【%s】裏ウェイト生成", material_name, decoration=MLogger.DECORATION_LINE)

        weight_cnt = 0
        prev_weight_cnt = 0

        front_vertices = {}
        for vv in virtual_vertices.values():
            for v in vv.real_vertices:
                front_vertices[v.index] = v.position.data()

        for vertex_idx in back_vertices:
            bv = model.vertex_dict[vertex_idx]

            # 各頂点の位置との差分から距離を測る
            bv_distances = np.linalg.norm(
                (np.array(list(front_vertices.values())) - bv.position.data()), ord=2, axis=1
            )

            # 直近頂点INDEXのウェイトを転写
            copy_front_vertex_idx = list(front_vertices.keys())[np.argmin(bv_distances)]
            bv.deform = copy.deepcopy(model.vertex_dict[copy_front_vertex_idx].deform)

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 200 > prev_weight_cnt:
                logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)
                prev_weight_cnt = weight_cnt // 200

        logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)

    def create_remaining_weight(
        self, model: PmxModel, param_option: dict, material_name: str, virtual_vertices: dict, remaining_vertices: dict
    ):
        logger.info("【%s】残ウェイト生成", material_name, decoration=MLogger.DECORATION_LINE)

        vertex_cnt = 0
        prev_vertex_cnt = 0

        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]

        # 塗り終わった頂点リスト
        weighted_vkeys = list(set(list(virtual_vertices.keys())) - set(list(remaining_vertices.keys())))

        weighted_poses = {}
        for vkey in weighted_vkeys:
            vv = virtual_vertices[vkey]
            weighted_poses[vkey] = vv.position().data()

        for vkey, vv in remaining_vertices.items():
            if not vv.vidxs():
                continue

            # ウェイト済み頂点のうち、最も近いのを抽出
            weighted_diff_distances = np.linalg.norm(
                np.array(list(weighted_poses.values())) - vv.position().data(), ord=2, axis=1
            )
            nearest_vkey = list(weighted_poses.keys())[np.argmin(weighted_diff_distances)]
            nearest_vv = virtual_vertices[nearest_vkey]
            nearest_deform = nearest_vv.deform

            if type(nearest_deform) is Bdef1:
                logger.debug(
                    f"remaining1 nearest_vv: {nearest_vv.vidxs()}, weight_names: [{model.bone_indexes[nearest_deform.index0]}], total_weights: [1]"
                )

                for rv in vv.real_vertices:
                    rv.deform = Bdef1(nearest_deform.index0)

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

                logger.debug(
                    f"remaining2 nearest_vv: {vv.vidxs()}, weight_names: [{weight_names}], total_weights: [{total_weights}]"
                )

                if np.count_nonzero(weights) == 1:
                    vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                elif np.count_nonzero(weights) == 2:
                    vv.deform = Bdef2(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        weights[weight_idxs[-1]],
                    )
                else:
                    vv.deform = Bdef4(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        model.bones[weight_names[weight_idxs[-3]]].index,
                        model.bones[weight_names[weight_idxs[-4]]].index,
                        weights[weight_idxs[-1]],
                        weights[weight_idxs[-2]],
                        weights[weight_idxs[-3]],
                        weights[weight_idxs[-4]],
                    )

                for rv in vv.real_vertices:
                    rv.deform = vv.deform

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

                logger.debug(
                    f"remaining4 nearest_vv: {vv.vidxs()}, weight_names: [{weight_names}], total_weights: [{total_weights}]"
                )

                if np.count_nonzero(weights) == 1:
                    vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                elif np.count_nonzero(weights) == 2:
                    vv.deform = Bdef2(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        weights[weight_idxs[-1]],
                    )
                else:
                    vv.deform = Bdef4(
                        model.bones[weight_names[weight_idxs[-1]]].index,
                        model.bones[weight_names[weight_idxs[-2]]].index,
                        model.bones[weight_names[weight_idxs[-3]]].index,
                        model.bones[weight_names[weight_idxs[-4]]].index,
                        weights[weight_idxs[-1]],
                        weights[weight_idxs[-2]],
                        weights[weight_idxs[-3]],
                        weights[weight_idxs[-4]],
                    )

                for rv in vv.real_vertices:
                    rv.deform = vv.deform

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
        vertex_maps: dict,
        all_regist_bones: dict,
        all_bone_vertical_distances: dict,
        all_bone_horizonal_distances: dict,
        all_bone_connected: dict,
        remaining_vertices: dict,
    ):
        logger.info("【%s】ウェイト生成", material_name, decoration=MLogger.DECORATION_LINE)

        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]

        for base_map_idx, regist_bones in all_regist_bones.items():
            logger.info("--【No.%s】ウェイト分布判定", base_map_idx + 1)

            vertex_map = vertex_maps[base_map_idx]

            # ウェイト分布
            prev_weight_cnt = 0
            weight_cnt = 0

            for v_xidx in range(regist_bones.shape[1]):
                target_v_xidx = v_xidx

                for v_yidx in range(regist_bones.shape[0]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        continue

                    vkey = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices[vkey]

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
                        max_v_yidx,
                        max_v_xidx,
                    ) = self.get_block_vidxs(
                        v_yidx, v_xidx, vertex_maps, all_regist_bones, all_bone_connected, base_map_idx
                    )

                    if regist_bones[v_yidx, v_xidx] and v_yidx < max_v_yidx:
                        # 同じ仮想頂点上に登録されているボーンが複数ある場合、均等に割る
                        weight_bone_idxs = list(
                            set(
                                [
                                    mbone.index
                                    for mbone in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].map_bones.values()
                                    if mbone
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
                        elif np.count_nonzero(weight_bone_idxs) == 2:
                            vv.deform = Bdef2(
                                weight_bone_idxs[0],
                                weight_bone_idxs[1],
                                deform_weights[0],
                            )
                        else:
                            # 3つの場合にうまくいかないので、後ろに追加しておく
                            deform_weights += [0 for _ in range(4)]
                            weight_bone_idxs += [parent_bone.index for _ in range(4)]

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

                        # 頂点位置にボーンが登録されている場合、BDEF1登録対象
                        for rv in vv.real_vertices:
                            rv.deform = vv.deform

                            # 逆登録
                            for weight_bone_idx in weight_bone_idxs:
                                if weight_bone_idx not in model.vertices:
                                    model.vertices[weight_bone_idx] = []
                                model.vertices[weight_bone_idx].append(rv)

                        # 登録対象の場合、残対象から削除
                        if vkey in remaining_vertices:
                            del remaining_vertices[vkey]

                    elif np.where(regist_bones[v_yidx, :])[0].shape[0] > 1 and v_yidx < max_v_yidx:
                        # 同じY位置にボーンがある場合、横のBDEF2登録対象
                        if v_xidx < regist_bones.shape[1] - 1 and regist_bones[v_yidx, (v_xidx + 1) :].any():
                            regist_next_xidx = next_xidx
                        else:
                            regist_next_xidx = 0

                        if (
                            not all_bone_horizonal_distances[base_map_idx].any()
                            or np.isnan(vertex_maps[prev_map_idx][v_yidx, prev_xidx]).any()
                            or np.isnan(vertex_maps[next_map_idx][v_yidx, regist_next_xidx]).any()
                            or not virtual_vertices[tuple(vertex_maps[prev_map_idx][v_yidx, prev_xidx])].map_bones.get(
                                prev_map_idx, None
                            )
                            or not virtual_vertices[
                                tuple(vertex_maps[next_map_idx][v_yidx, regist_next_xidx])
                            ].map_bones.get(next_map_idx, None)
                        ):
                            continue

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
                            virtual_vertices[tuple(vertex_maps[next_map_idx][v_yidx, regist_next_xidx])]
                            .map_bones[next_map_idx]
                            .index
                        )

                        vv.deform = Bdef2(
                            weight_bone_idx_0,
                            weight_bone_idx_1,
                            1 - prev_weight,
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

                        # 登録対象の場合、残対象から削除
                        if vkey in remaining_vertices:
                            del remaining_vertices[vkey]

                    elif np.where(regist_bones[:, v_xidx])[0].shape[0] > 1:
                        # 同じX位置にボーンがある場合、縦のBDEF2登録対象
                        if (
                            below_yidx == max_v_yidx
                            and not np.isnan(vertex_map[above_yidx, v_xidx]).any()
                            and virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])].map_bones.get(
                                base_map_idx, None
                            )
                        ):
                            weight_bone_idx_0 = (
                                virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])].map_bones[base_map_idx].index
                            )

                            vv.deform = Bdef1(weight_bone_idx_0)

                            # 末端がある場合、上のボーンでBDEF1
                            for rv in vv.real_vertices:
                                rv.deform = vv.deform

                                # 逆登録
                                model.vertices[weight_bone_idx_0].append(rv)

                            # 登録対象の場合、残対象から削除
                            if vkey in remaining_vertices:
                                del remaining_vertices[vkey]
                        else:
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
                                np.sum(
                                    all_bone_vertical_distances[base_map_idx][
                                        (above_yidx + 1) : (v_yidx + 1), (v_xidx - 1)
                                    ]
                                )
                                / np.sum(
                                    all_bone_vertical_distances[base_map_idx][
                                        (above_yidx + 1) : (below_yidx + 1), (v_xidx - 1)
                                    ]
                                )
                            )

                            weight_bone_idx_0 = (
                                virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])].map_bones[base_map_idx].index
                            )
                            weight_bone_idx_1 = (
                                virtual_vertices[tuple(vertex_map[below_yidx, v_xidx])].map_bones[base_map_idx].index
                            )

                            vv.deform = Bdef2(
                                weight_bone_idx_0,
                                weight_bone_idx_1,
                                1 - above_weight,
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

                            # 登録対象の場合、残対象から削除
                            if vkey in remaining_vertices:
                                del remaining_vertices[vkey]
                    else:
                        if next_connected and next_xidx == 0:
                            # 最後の頂点の場合、とりあえず次の距離を対象とする
                            next_xidx = vertex_map.shape[1] - 1
                            target_next_xidx = 0
                        else:
                            target_next_xidx = next_xidx

                        if (
                            not all_bone_vertical_distances[base_map_idx].any()
                            or not all_bone_horizonal_distances[base_map_idx].any()
                            or np.isnan(vertex_maps[prev_map_idx][above_yidx, prev_xidx]).any()
                            or np.isnan(vertex_maps[next_map_idx][above_yidx, target_next_xidx]).any()
                            or np.isnan(vertex_maps[prev_map_idx][below_yidx, prev_xidx]).any()
                            or np.isnan(vertex_maps[next_map_idx][below_yidx, target_next_xidx]).any()
                            or not virtual_vertices[
                                tuple(vertex_maps[prev_map_idx][above_yidx, prev_xidx])
                            ].map_bones.get(prev_map_idx, None)
                            or not virtual_vertices[tuple(vertex_map[above_yidx, next_xidx])].map_bones.get(
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
                                np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, (v_xidx - 1)])
                                / np.sum(
                                    all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)]
                                )
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        next_above_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, (v_xidx - 1)])
                                / np.sum(
                                    all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)]
                                )
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        prev_below_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, (v_xidx - 1)])
                                / np.sum(
                                    all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)]
                                )
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        next_below_weight = np.nan_to_num(
                            (
                                np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, (v_xidx - 1)])
                                / np.sum(
                                    all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)]
                                )
                            )
                            * (
                                np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx])
                                / np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx])
                            )
                        )

                        if below_yidx == max_v_yidx:
                            prev_above_weight += prev_below_weight
                            next_above_weight += next_below_weight

                            # ほぼ0のものは0に置換（円周用）
                            total_weights = np.array([prev_above_weight, next_above_weight])
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

                                vv.deform = Bdef2(
                                    weight_bone_idx_0,
                                    weight_bone_idx_1,
                                    deform_weights[0],
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

                                # 登録対象の場合、残対象から削除
                                if vkey in remaining_vertices:
                                    del remaining_vertices[vkey]
                        else:
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

                                # 登録対象の場合、残対象から削除
                                if vkey in remaining_vertices:
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
    ):
        # 略称
        abb_name = param_option["abb_name"]
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"
        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]

        # 中心ボーン
        root_bone = Bone(
            f"{abb_name}中心",
            f"{abb_name}Root",
            parent_bone.position,
            parent_bone.index,
            0,
            0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010,
        )
        if root_bone.name in model.bones:
            logger.warning("同じボーン名が既に登録されているため、末尾に乱数を追加します。 既存ボーン名: %s", root_bone.name)
            root_bone.name += randomname(3)

        root_bone.index = len(model.bones)
        model.bones[root_bone.name] = root_bone
        model.bone_indexes[root_bone.index] = root_bone.name

        # 表示枠
        model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)
        model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

        return display_name, root_bone

    def create_bone(
        self,
        model: PmxModel,
        param_option: dict,
        material_name: str,
        virtual_vertices: dict,
        vertex_maps: dict,
        vertex_map_orders: dict,
    ):
        logger.info("【%s】ボーン生成", material_name, decoration=MLogger.DECORATION_LINE)

        # 中心ボーン生成

        # 略称
        abb_name = param_option["abb_name"]
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"
        # 親ボーン
        parent_bone = model.bones[param_option["parent_bone_name"]]

        # 中心ボーン
        display_name, root_bone = self.create_root_bone(model, param_option, material_name)

        logger.info("【%s】頂点距離の算出", material_name)

        all_bone_horizonal_distances = {}
        all_bone_vertical_distances = {}
        all_bone_connected = {}

        for base_map_idx, vertex_map in enumerate(vertex_maps):
            logger.info("--【No.%s】頂点距離算出", base_map_idx + 1)

            prev_vertex_cnt = 0
            vertex_cnt = 0

            bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))
            bone_vertical_distances = np.zeros((vertex_map.shape[0] - 1, vertex_map.shape[1] - 1))
            bone_connected = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

            # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
            for v_yidx in range(vertex_map.shape[0]):
                v_xidx = -1
                for v_xidx in range(0, vertex_map.shape[1] - 1):
                    if (
                        not np.isnan(vertex_map[v_yidx, v_xidx]).any()
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

                v_xidx += 1
                if not np.isnan(vertex_map[v_yidx, v_xidx]).any() and not np.isnan(vertex_map[v_yidx, 0]).any():
                    # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
                    if (
                        tuple(vertex_map[v_yidx, 0])
                        in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].connected_vvs
                    ):
                        # 横の仮想頂点と繋がっている場合、Trueで有効な距離を入れておく
                        bone_connected[v_yidx, v_xidx] = True

                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, 0])].position()
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)
                    else:
                        # とりあえずINT最大値を入れておく
                        bone_horizonal_distances[v_yidx, v_xidx] = np.iinfo(np.int).max

            all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
            all_bone_vertical_distances[base_map_idx] = bone_vertical_distances
            all_bone_connected[base_map_idx] = bone_connected

        if len(vertex_maps) > 1:
            for base_map_idx, (vertex_map, next_vertex_map) in enumerate(
                zip(vertex_maps, vertex_maps[1:] + [vertex_maps[0]])
            ):
                # 複数マップある場合、繋ぎ目をチェックする
                for v_yidx in range(vertex_map.shape[0]):
                    if (
                        not np.isnan(vertex_map[v_yidx, -1]).any()
                        and not np.isnan(next_vertex_map[v_yidx, 0]).any()
                        and (
                            tuple(next_vertex_map[v_yidx, 0])
                            in virtual_vertices[tuple(vertex_map[v_yidx, -1])].connected_vvs
                            or tuple(vertex_map[v_yidx, -1]) == tuple(next_vertex_map[v_yidx, 0])
                        )
                    ):
                        # 同じ仮想頂点もしくは繋がれた仮想頂点の場合、繋がれているとみなす
                        all_bone_connected[base_map_idx][v_yidx, -1] = True

        # 全体通してのX番号
        prev_xs = []
        all_regist_bones = {}
        tmp_all_bones = {}
        for base_map_idx, vertex_map in enumerate(vertex_maps):

            prev_bone_cnt = 0
            bone_cnt = 0

            # ボーン登録有無
            regist_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)
            all_regist_bones[base_map_idx] = regist_bones

            if base_map_idx not in vertex_map_orders:
                # ボーン生成対象外の場合、とりあえず枠だけ作ってスルー
                continue

            logger.info("--【No.%s】ボーン生成", base_map_idx + 1)

            if param_option["density_type"] == logger.transtext("距離"):
                median_vertical_distance = np.median(
                    all_bone_vertical_distances[base_map_idx][:, int(vertex_map.shape[1] / 2)]
                )
                median_horizonal_distance = np.median(
                    all_bone_horizonal_distances[base_map_idx][int(vertex_map.shape[0] / 2), :]
                )

                logger.debug(
                    f"median_horizonal_distance: {round(median_horizonal_distance, 4)}, median_vertical_distance: {round(median_vertical_distance, 4)}"
                )

                # 間隔が距離タイプの場合、均等になるように間を空ける
                y_regists = np.zeros(vertex_map.shape[0], dtype=np.int)
                prev_y_regist = 0
                for v_yidx in range(vertex_map.shape[0]):
                    if (
                        np.sum(
                            all_bone_vertical_distances[base_map_idx][
                                (prev_y_regist + 1) : (v_yidx + 1), int(vertex_map.shape[1] / 2)
                            ]
                        )
                        > median_vertical_distance * param_option["vertical_bone_density"]
                    ):
                        # 前の登録ボーンから一定距離離れたら登録対象
                        y_regists[v_yidx] = True
                        prev_y_regist = v_yidx
                # 最初と最後は必ず登録する
                y_regists[0] = y_regists[-1] = True

                x_regists = np.zeros(vertex_map.shape[1], dtype=np.int)
                prev_x_regist = 0
                for v_xidx in range(vertex_map.shape[1]):
                    if (
                        np.sum(
                            all_bone_horizonal_distances[base_map_idx][
                                int(vertex_map.shape[0] / 2), (prev_x_regist + 1) : (v_xidx + 1)
                            ]
                        )
                        > median_horizonal_distance * param_option["horizonal_bone_density"]
                    ):
                        # 前の登録ボーンから一定距離離れたら登録対象
                        x_regists[v_xidx] = True
                        prev_x_regist = v_xidx
                # 最初と最後は必ず登録する
                x_regists[0] = x_regists[-1] = True

                for v_yidx, y_regist in enumerate(y_regists):
                    for v_xidx, x_regist in enumerate(x_regists):
                        regist_bones[v_yidx, v_xidx] = y_regist and x_regist

            else:
                # 間隔が頂点タイプの場合、規則的に間を空ける
                for v_yidx in list(range(0, vertex_map.shape[0], param_option["vertical_bone_density"])) + [
                    vertex_map.shape[0] - 1
                ]:
                    for v_xidx in range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]):
                        if (
                            not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                            or not all_bone_connected[base_map_idx][v_yidx, v_xidx]
                        ):
                            regist_bones[v_yidx, v_xidx] = True
                    if not all_bone_connected[base_map_idx][v_yidx, vertex_map.shape[1] - 1]:
                        # 繋がってない場合、最後に追加する
                        if (
                            not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                            or not all_bone_connected[base_map_idx][v_yidx, v_xidx]
                        ):
                            regist_bones[v_yidx, vertex_map.shape[1] - 1] = True

            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    target_v_yidx = v_yidx
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        if not regist_bones[v_yidx, v_xidx]:
                            # 登録対象ではない場合、スルー
                            continue

                        # 自分より下のYインデックス
                        below_max_v_yidx = (
                            np.max(np.nonzero(~np.isnan(vertex_map[v_yidx:, v_xidx]))[0]) + v_yidx
                            if np.nonzero(~np.isnan(vertex_map[v_yidx:, v_xidx]))[0].any()
                            else -1
                        )
                        # 下のYインデックスがある場合、そこに登録する
                        if below_max_v_yidx > 0:
                            # 登録対象を入れ替える
                            regist_bones[v_yidx, v_xidx] = False
                            regist_bones[below_max_v_yidx, v_xidx] = True
                            target_v_yidx = below_max_v_yidx
                        else:
                            # 下のYインデックスがない場合、それより上のを登録する
                            max_v_yidx = (
                                np.max(np.nonzero(~np.isnan(vertex_map[:, v_xidx]))[0])
                                if np.nonzero(~np.isnan(vertex_map[:, v_xidx]))[0].any()
                                else vertex_map.shape[0] - 1
                            )
                            regist_bones[v_yidx, v_xidx] = False
                            regist_bones[max_v_yidx, v_xidx] = True
                            target_v_yidx = max_v_yidx

                    v_yno = target_v_yidx + 1
                    v_xno = v_xidx + len(prev_xs) + 1

                    vkey = tuple(vertex_map[target_v_yidx, v_xidx])
                    vv = virtual_vertices[vkey]

                    # 親は既にモデルに登録済みのものを選ぶ
                    parent_bone = None
                    for parent_v_yidx in range(target_v_yidx - 1, -1, -1):
                        parent_bone = virtual_vertices[tuple(vertex_map[parent_v_yidx, v_xidx])].map_bones.get(
                            base_map_idx, None
                        )
                        if parent_bone and (parent_bone.name in model.bones or parent_bone.name in tmp_all_bones):
                            # 登録されていたら終了
                            break
                    if not parent_bone:
                        # 最後まで登録されている親ボーンが見つからなければ、ルート
                        parent_bone = root_bone

                    if not vv.map_bones.get(base_map_idx, None):
                        # ボーン仮登録
                        is_add_random = False
                        bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                        if bone_name in model.bones or bone_name in tmp_all_bones:
                            # 仮登録の時点で乱数は後ろに付けておくが、メッセージは必要なのだけ出す
                            is_add_random = True
                            bone_name += randomname(3)

                        bone = Bone(bone_name, bone_name, vv.position().copy(), parent_bone.index, 0, 0x0000 | 0x0002)
                        bone.local_z_vector = vv.normal().copy()

                        bone.parent_index = parent_bone.index
                        bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                        bone.local_z_vector *= MVector3D(-1, 1, -1)
                        bone.flag |= 0x0800

                        if regist_bones[target_v_yidx, v_xidx]:
                            vv.map_bones[base_map_idx] = bone
                            bone.index = len(tmp_all_bones) + len(model.bones)

                            if is_add_random:
                                logger.warning("同じボーン名が既に登録されているため、末尾に乱数を追加します。 既存ボーン名: %s", bone.name)

                            # 登録対象である場合
                            tmp_all_bones[bone.name] = bone

                        logger.debug(f"tmp_all_bones: {bone.name}, pos: {bone.position.to_log()}")

                        bone_cnt += 1
                        if bone_cnt > 0 and bone_cnt // 1000 > prev_bone_cnt:
                            logger.info("-- --【No.%s】ボーン生成: %s個目:終了", base_map_idx + 1, bone_cnt)
                            prev_bone_cnt = bone_cnt // 1000

            prev_xs.extend(list(range(vertex_map.shape[1])))

        # まずはソートしたボーン名で登録
        registed_bone_names = {}
        for bone_name in sorted(tmp_all_bones.keys()):
            bone = tmp_all_bones[bone_name]

            is_regist = True
            for rbone_name in registed_bone_names.values():
                # 登録済みのボーンリストと比較
                rbone = model.bones[rbone_name]
                if (
                    rbone.position == bone.position
                    and rbone.parent_index in model.bone_indexes
                    and bone.parent_index in model.bone_indexes
                    and model.bones[model.bone_indexes[rbone.parent_index]].position
                    == model.bones[model.bone_indexes[bone.parent_index]].position
                ):
                    # ボーン構成がまったく同じ場合、このボーンそのものは登録対象外
                    is_regist = False
                    break

            if is_regist:
                registed_bone_names[bone.index] = bone_name
                bone.index = len(model.bones)
                model.bones[bone_name] = bone
                model.bone_indexes[bone.index] = bone_name

                model.display_slots[display_name].references.append((0, bone.index))
            else:
                # 登録対象外だった場合、前に登録されているボーン名を参照する
                registed_bone_names[bone.index] = rbone_name
                # 仮想頂点に紐付くボーンも統一する
                vv = virtual_vertices[bone.position.to_key(param_option["threshold"])]
                for midx in vv.map_bones.keys():
                    vv.map_bones[midx] = rbone

        # その後で親ボーン・表示先ボーンのINDEXを切り替える
        for bone_name in sorted(list(set(list(registed_bone_names.values())))):
            bone = model.bones[bone_name]
            parent_bone = root_bone
            if bone.parent_index in registed_bone_names:
                # 登録前のINDEXから名称を引っ張ってきて、それを新しいINDEXで置き換える
                parent_bone = model.bones[registed_bone_names[bone.parent_index]]
                bone.parent_index = parent_bone.index

            # 親ボーンの表示先も同時設定
            if parent_bone != root_bone:
                parent_bone.tail_index = bone.index
                parent_bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                # 親ボーンを表示対象にする
                parent_bone.flag |= 0x0001 | 0x0008 | 0x0010

        return (
            root_bone,
            virtual_vertices,
            all_regist_bones,
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
        vertex_maps: list,
    ):
        logger.info("【%s】胸物理生成", material_name, decoration=MLogger.DECORATION_LINE)

        for base_map_idx, vertex_map in enumerate(vertex_maps):
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
            param_rigidbody.param.mass if is_joint else 0.1,
            param_rigidbody.param.linear_damping,
            param_rigidbody.param.angular_damping,
            param_rigidbody.param.restitution,
            param_rigidbody.param.friction,
            1,
        )
        rigidbody.shape_qq = shape_qq

        return rigidbody

    def create_bone_map(
        self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list, is_root_bone=True
    ):
        bone_grid = param_option["bone_grid"]
        bone_grid_cols = param_option["bone_grid_cols"]
        bone_grid_rows = param_option["bone_grid_rows"]
        # 閾値
        threshold = param_option["threshold"]

        logger.info("【%s】ボーンマップ生成", material_name, decoration=MLogger.DECORATION_LINE)

        logger.info("%s: ウェイトボーンの確認", material_name)

        virtual_vertices = {}

        # ウェイトボーンリスト取得
        weighted_bone_pairs = []
        for n, v_idx in enumerate(model.material_vertices[material_name]):
            if v_idx not in target_vertices:
                continue

            v = model.vertex_dict[v_idx]
            v_key = v.position.to_key(threshold)
            if v_key not in virtual_vertices:
                virtual_vertices[v_key] = VirtualVertex(v_key)
            virtual_vertices[v_key].append([v], [], [])

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

                for bi0, bi1 in list(itertools.combinations(weighted_bone_indexes, 2)):
                    # ボーン2つずつのペアでウェイト繋がり具合を保持する
                    key = (min(bi0, bi1), max(bi0, bi1))
                    if key not in weighted_bone_pairs:
                        weighted_bone_pairs.append(key)

            if n > 0 and n % 1000 == 0:
                logger.info("-- ウェイトボーン確認: %s個目:終了", n)

        logger.info("%s: 仮想ボーンリストの生成", material_name)
        all_bone_connected = {}
        all_regist_bones = {}
        vertex_maps = []

        if param_option["physics_type"] in [logger.transtext("布")]:
            vertex_map = np.full((bone_grid_rows, bone_grid_cols, 3), (np.nan, np.nan, np.nan))
            bone_connected = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)
            regist_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

            for grid_col in range(bone_grid_cols):
                for grid_row in range(bone_grid_rows):
                    bone_name = bone_grid[grid_row][grid_col]
                    bone = model.bones.get(bone_name, None)
                    if not bone_name or not bone:
                        continue

                    # ボーンの位置で登録する
                    nearest_v_key = bone.position.to_key(threshold)
                    if nearest_v_key not in virtual_vertices:
                        virtual_vertices[nearest_v_key] = VirtualVertex(nearest_v_key)

                    # 最もボーンに近い頂点に紐付くボーンとして登録
                    virtual_vertices[nearest_v_key].positions.append(bone.position.data())
                    virtual_vertices[nearest_v_key].map_bones[0] = bone
                    vertex_map[grid_row, grid_col] = nearest_v_key
                    regist_bones[grid_row, grid_col] = True

                    logger.info("-- 仮想ボーン: %s: 終了", bone.name)

            # 布は横の繋がりをチェックする
            for v_yidx in range(vertex_map.shape[0]):
                v_xidx = -1
                for v_xidx in range(0, vertex_map.shape[1] - 1):
                    if (
                        not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                        and not np.isnan(vertex_map[v_yidx, v_xidx + 1]).any()
                    ):
                        vv1 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])]
                        vv2 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])]

                        vv1_bone_index = vv1.map_bones[0].index
                        vv2_bone_index = vv2.map_bones[0].index

                        key = (min(vv1_bone_index, vv2_bone_index), max(vv1_bone_index, vv2_bone_index))
                        if key in weighted_bone_pairs:
                            # ウェイトを共有するボーンの組み合わせであった場合、接続TRUE
                            bone_connected[v_yidx, v_xidx] = True

                v_xidx += 1
                if not np.isnan(vertex_map[v_yidx, v_xidx]).any() and not np.isnan(vertex_map[v_yidx, 0]).any():
                    # 輪を描いたのも繋がっているかチェック
                    vv1 = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])]
                    vv2 = virtual_vertices[tuple(vertex_map[v_yidx, 0])]

                    vv1_bone_index = vv1.map_bones[0].index
                    vv2_bone_index = vv2.map_bones[0].index

                    key = (min(vv1_bone_index, vv2_bone_index), max(vv1_bone_index, vv2_bone_index))
                    if key in weighted_bone_pairs:
                        # ウェイトを共有するボーンの組み合わせであった場合、接続TRUE
                        bone_connected[v_yidx, v_xidx] = True

            all_bone_connected[0] = bone_connected
            all_regist_bones[0] = regist_bones
            vertex_maps.append(vertex_map)
        else:
            # 布以外は一列ものとして別登録
            for grid_col in range(bone_grid_cols):
                valid_bone_grid_rows = [n for n in range(bone_grid_rows) if bone_grid[n][grid_col]]
                vertex_map = np.full((len(valid_bone_grid_rows), 1, 3), (np.nan, np.nan, np.nan))
                # 横との接続は一切なし
                bone_connected = np.zeros((len(valid_bone_grid_rows), 1), dtype=np.int)
                regist_bones = np.full((len(valid_bone_grid_rows), 1), 1, dtype=np.int)

                for grid_row in valid_bone_grid_rows:
                    bone_name = bone_grid[grid_row][grid_col]
                    bone = model.bones.get(bone_name, None)
                    if not bone_name or not bone:
                        continue

                    # ボーンの位置で登録する
                    nearest_v_key = bone.position.to_key(threshold)
                    if nearest_v_key not in virtual_vertices:
                        virtual_vertices[nearest_v_key] = VirtualVertex(nearest_v_key)

                    # 最もボーンに近い頂点に紐付くボーンとして登録
                    virtual_vertices[nearest_v_key].map_bones[grid_col] = bone
                    vertex_map[grid_row, 0] = nearest_v_key
                    regist_bones[grid_row, 0] = True

                    logger.info("-- 仮想ボーン: %s: 終了", bone.name)

                all_bone_connected[grid_col] = bone_connected
                all_regist_bones[grid_col] = regist_bones
                vertex_maps.append(vertex_map)

        # 中心ボーン
        root_bone = None
        if is_root_bone:
            _, root_bone = self.create_root_bone(model, param_option, material_name)

        return virtual_vertices, vertex_maps, all_regist_bones, all_bone_connected, root_bone

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
        logger.info("【%s】頂点マップ生成", material_name, decoration=MLogger.DECORATION_LINE)

        # 閾値
        threshold = param_option["threshold"]

        # 裏面頂点リスト
        back_vertices = []
        # 裏面リスト
        back_indexes = []
        # 残頂点リスト
        remaining_vertices = {}

        parent_bone = model.bones[param_option["parent_bone_name"]]

        # 一旦全体の位置を把握
        vertex_positions = {}
        for vertex_idx in model.material_vertices[material_name]:
            if vertex_idx not in target_vertices:
                continue
            vertex_positions[vertex_idx] = model.vertex_dict[vertex_idx].position.data()

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
        logger.info("%s: 材質頂点の傾き算出: %s", material_name, material_direction)

        logger.info("%s: 仮想頂点リストの生成", material_name)

        virtual_vertices = {}

        edge_pair_lkeys = {}
        for n, index_idx in enumerate(model.material_indices[material_name]):
            # 頂点の組み合わせから面INDEXを引く
            if (
                model.indices[index_idx][0] not in target_vertices
                or model.indices[index_idx][1] not in target_vertices
                or model.indices[index_idx][2] not in target_vertices
            ):
                # 3つ揃ってない場合、スルー
                continue

            for v0_idx, v1_idx, v2_idx in zip(
                model.indices[index_idx],
                model.indices[index_idx][1:] + [model.indices[index_idx][0]],
                [model.indices[index_idx][2]] + model.indices[index_idx][:-1],
            ):
                v0 = model.vertex_dict[v0_idx]
                v1 = model.vertex_dict[v1_idx]
                v2 = model.vertex_dict[v2_idx]

                v0_key = v0.position.to_key(threshold)
                v1_key = v1.position.to_key(threshold)
                v2_key = v2.position.to_key(threshold)

                # 一旦ルートボーンにウェイトを一括置換
                v0.deform = Bdef1(parent_bone.index)

                # 面垂線
                vv1 = v1.position - v0.position
                vv2 = v2.position - v1.position
                surface_normal = MVector3D.crossProduct(vv1, vv2).normalized()

                # 親ボーンに対する向き
                parent_direction = (
                    ((v0.position + v1.position + v2.position) / 3) - parent_bone.position
                ).normalized()

                # 親ボーンの向きとの内積
                normal_dot = MVector3D.dotProduct(
                    surface_normal * base_reverse_axis, parent_direction * base_reverse_axis
                )

                if np.isclose(material_direction, 0):
                    # 水平の場合、軸の向きだけ考える
                    normal_dot = surface_normal.data()[target_idx]

                logger.debug(
                    f"index[{index_idx}], v0[{v0.index}:{v0_key}], sn[{surface_normal.to_log()}], pd[{parent_direction.to_log()}], dot[{round(normal_dot, 5)}]"
                )

                # 面法線と同じ向き場合、辺キー生成（表面のみを対象とする）
                if np.sign(normal_dot) > 0:
                    lkey = (min(v0_key, v1_key), max(v0_key, v1_key))
                    if lkey not in edge_pair_lkeys:
                        edge_pair_lkeys[lkey] = []
                    if index_idx not in edge_pair_lkeys[lkey]:
                        edge_pair_lkeys[lkey].append(index_idx)

                    if v0_key not in virtual_vertices:
                        virtual_vertices[v0_key] = VirtualVertex(v0_key)

                    # 仮想頂点登録（該当頂点対象）
                    virtual_vertices[v0_key].append([v0], [v1_key, v2_key], [index_idx])

                    if v1_key not in virtual_vertices:
                        virtual_vertices[v1_key] = VirtualVertex(v1_key)

                    virtual_vertices[v1_key].append([v1], [v2_key, v0_key], [index_idx])

                    # 残頂点リストにまずは登録
                    if v0_key not in remaining_vertices:
                        remaining_vertices[v0_key] = virtual_vertices[v0_key]
                else:
                    # 裏面に登録
                    if index_idx not in back_indexes:
                        back_indexes.append(index_idx)

            if n > 0 and n % 500 == 0:
                logger.info("-- メッシュ確認: %s個目:終了", n)

        if param_option["special_shape"] == logger.transtext("プリーツ"):
            # プリーツは折りたたみ面を裏面として扱ってるため、裏面の頂点を仮想頂点の連結先として登録する
            for n, index_idx in enumerate(back_indexes):
                (v0_idx, v1_idx, v2_idx) = model.indices[index_idx]

                for v_idx, ov1_idx, ov2_idx in [
                    (v0_idx, v1_idx, v2_idx),
                    (v1_idx, v2_idx, v0_idx),
                    (v2_idx, v0_idx, v1_idx),
                ]:
                    v = model.vertex_dict[v_idx]
                    ov1 = model.vertex_dict[ov1_idx]
                    ov2 = model.vertex_dict[ov2_idx]

                    v_key = v.position.to_key(threshold)
                    ov1_key = ov1.position.to_key(threshold)
                    ov2_key = ov2.position.to_key(threshold)

                    if v_idx not in back_vertices and (
                        v_key not in virtual_vertices
                        or (v_key in virtual_vertices and v_idx not in virtual_vertices[v_key].real_vertices)
                    ):
                        back_vertices.append(v_idx)

                    if v_key in virtual_vertices:
                        if ov1_key not in virtual_vertices[v_key].connected_vvs:
                            virtual_vertices[v_key].connected_vvs.append(ov1_key)
                        if index_idx not in virtual_vertices[v_key].indexes:
                            virtual_vertices[v_key].indexes.append(index_idx)
                        if ov2_key not in virtual_vertices[v_key].connected_vvs:
                            virtual_vertices[v_key].connected_vvs.append(ov2_key)
                        if index_idx not in virtual_vertices[v_key].indexes:
                            virtual_vertices[v_key].indexes.append(index_idx)
                        if ov1_key in virtual_vertices and ov2_key not in virtual_vertices:
                            # v と ov1 がいて、ov2がいない場合、辺を共有する
                            lkey = (min(v_key, ov1_key), max(v_key, ov1_key))
                            if lkey not in edge_pair_lkeys:
                                edge_pair_lkeys[lkey] = []
                            if index_idx not in edge_pair_lkeys[lkey]:
                                edge_pair_lkeys[lkey].append(index_idx)
                        if ov2_key in virtual_vertices and ov1_key not in virtual_vertices:
                            lkey = (min(v_key, ov2_key), max(v_key, ov2_key))
                            if lkey not in edge_pair_lkeys:
                                edge_pair_lkeys[lkey] = []
                            if index_idx not in edge_pair_lkeys[lkey]:
                                edge_pair_lkeys[lkey].append(index_idx)

                    if v_key in virtual_vertices and ov1_key in virtual_vertices and ov2_key in virtual_vertices:
                        lkey = (min(v_key, ov1_key), max(v_key, ov1_key))
                        if lkey not in edge_pair_lkeys:
                            edge_pair_lkeys[lkey] = []
                        if index_idx not in edge_pair_lkeys[lkey]:
                            edge_pair_lkeys[lkey].append(index_idx)
                        lkey = (min(ov1_key, ov2_key), max(ov1_key, ov2_key))
                        if lkey not in edge_pair_lkeys:
                            edge_pair_lkeys[lkey] = []
                        if index_idx not in edge_pair_lkeys[lkey]:
                            edge_pair_lkeys[lkey].append(index_idx)
                        lkey = (min(v_key, ov2_key), max(v_key, ov2_key))
                        if lkey not in edge_pair_lkeys:
                            edge_pair_lkeys[lkey] = []
                        if index_idx not in edge_pair_lkeys[lkey]:
                            edge_pair_lkeys[lkey].append(index_idx)

                if n > 0 and n % 500 == 0:
                    logger.info("-- 裏メッシュ確認: %s個目:終了", n)
        else:
            # デフォルトは面の頂点だけ保持し直す
            for n, index_idx in enumerate(back_indexes):
                for v_idx in model.indices[index_idx]:
                    v = model.vertex_dict[v_idx]
                    v_key = v.position.to_key(threshold)

                    if v_idx not in back_vertices and (
                        v_key not in virtual_vertices
                        or (v_key in virtual_vertices and v_idx not in virtual_vertices[v_key].real_vertices)
                    ):
                        back_vertices.append(v_idx)

                if n > 0 and n % 1000 == 0:
                    logger.info("-- 裏メッシュ確認: %s個目:終了", n)

        if not virtual_vertices:
            logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None, None, None, None

        if not edge_pair_lkeys:
            logger.warning("対象範囲にエッジが見つけられなかった為、処理を終了します。\n面が表裏反転してないかご確認ください。", decoration=MLogger.DECORATION_BOX)
            return None, None, None, None

        edge_line_pairs = {}
        for n, ((min_vkey, max_vkey), line_iidxs) in enumerate(edge_pair_lkeys.items()):
            is_pair = False
            if len(line_iidxs) == 1:
                is_pair = True
            elif param_option["special_shape"] != logger.transtext("プリーツ"):
                surface_dots = []
                for index_idx in line_iidxs:
                    (v0_idx, v1_idx, v2_idx) = model.indices[index_idx]

                    v0 = model.vertex_dict[v0_idx]
                    v1 = model.vertex_dict[v1_idx]
                    v2 = model.vertex_dict[v2_idx]

                    # 面垂線
                    vv1 = v1.position - v0.position
                    vv2 = v2.position - v1.position
                    surface_normal = MVector3D.crossProduct(vv1, vv2).normalized()
                    surface_dots.append(MVector3D.dotProduct(surface_normal, MVector3D(0, 1, 0)))

                logger.debug(f"line_iidxs: [{line_iidxs}], surface_dots: [{surface_dots}]")

                minus_surface = np.where(np.array(surface_dots) < -0.1)[0]
                plus_surface = np.where(np.array(surface_dots) >= -0.1)[0]

                if minus_surface.shape[0] == 1 or plus_surface.shape[0] == 1:
                    # 面法線が1つしかないやつがあった場合、厚みのある材質で折り返しと共有しているとみなして、ペアリストに追加
                    is_pair = True

            if is_pair:
                if min_vkey not in virtual_vertices or max_vkey not in virtual_vertices:
                    continue

                if min_vkey not in edge_line_pairs:
                    edge_line_pairs[min_vkey] = []
                if max_vkey not in edge_line_pairs:
                    edge_line_pairs[max_vkey] = []

                edge_line_pairs[min_vkey].append(max_vkey)
                edge_line_pairs[max_vkey].append(min_vkey)

            if n > 0 and n % 500 == 0:
                logger.info("-- 辺確認: %s個目:終了", n)

        if logger.is_debug_level():
            logger.debug("--------------------------")
            logger.debug("仮想頂点リスト")
            for key, virtual_vertex in virtual_vertices.items():
                logger.debug(f"[{key}] {virtual_vertex}")

            logger.debug("--------------------------")
            logger.debug("エッジリスト")
            for (min_key, max_key), indexes in edge_pair_lkeys.items():
                logger.debug(
                    f"min[{min_key}:{virtual_vertices[min_key].vidxs()}], max[{max_key}:{virtual_vertices[max_key].vidxs()}] {indexes}"
                )

            logger.debug("--------------------------")
            logger.debug("エッジペアリスト")
            for v_key, pair_vkeys in edge_line_pairs.items():
                logger.debug(
                    f"key[{v_key}:{virtual_vertices[v_key].vidxs()}], pair[{pair_vkeys}:{[virtual_vertices[pair_vkey].vidxs() for pair_vkey in pair_vkeys]}]"
                )

        logger.info("%s: エッジの抽出準備", material_name)

        # エッジを繋いでいく
        all_edge_lines = []
        edge_vkeys = []
        while len(edge_vkeys) < len(edge_line_pairs.keys()):
            _, all_edge_lines, edge_vkeys = self.get_edge_lines(edge_line_pairs, None, all_edge_lines, edge_vkeys)

        all_edge_lines = [els for els in all_edge_lines if len(els) > 2]

        for n, edge_lines in enumerate(all_edge_lines):
            logger.info(
                "-- %s: 検出エッジ: %s", material_name, [f"{ekey}:{virtual_vertices[ekey].vidxs()}" for ekey in edge_lines]
            )

        logger.info("%s: エッジの抽出", material_name)

        horizonal_edge_lines = []
        vertical_edge_lines = []
        for n, edge_lines in enumerate(all_edge_lines):
            if 1 < len(all_edge_lines):
                horizonal_edge_lines.append([])
                vertical_edge_lines.append([])

            target_idx_poses = []

            for prev_edge_key, now_edge_key, next_edge_key in zip(
                list(edge_lines[-1:]) + list(edge_lines[:-1]), edge_lines, list(edge_lines[1:]) + list(edge_lines[:1])
            ):
                prev_edge_pos = virtual_vertices[prev_edge_key].position()
                now_edge_pos = virtual_vertices[now_edge_key].position()

                if material_direction == 0:
                    # 水平の場合
                    # キーの主に動く方向の位置。Z固定
                    target_idx_poses.append(now_edge_key[2])
                else:
                    # キーの主に動く方向の位置。スリットとかは位置が変動する
                    target_idx_poses.append(now_edge_key[target_idx])

            # https://teratail.com/questions/162391
            # 差分近似
            target_idx_pose_f_prime = np.gradient(target_idx_poses)
            # 変曲点を求める
            target_idx_pose_f_prime_sign = np.sign(target_idx_pose_f_prime)
            target_idx_pose_f_prime_diff = np.where(np.diff(target_idx_pose_f_prime_sign))[0]

            if (
                len(target_idx_pose_f_prime_diff) > 2
                and param_option["special_shape"] != logger.transtext("エッジ不定形")
                and param_option["special_shape"] != logger.transtext("プリーツ")
            ):
                target_idx_pose_indices = []
                for d in target_idx_pose_f_prime_diff:
                    if target_idx_pose_f_prime_sign[d] != 0 and target_idx_pose_f_prime_sign[d + 1] == 0:
                        target_idx_pose_indices.append(d)
                    else:
                        target_idx_pose_indices.append(d + 1)
                target_idx_pose_indices.append(target_idx_pose_f_prime.shape[0] - 1)
                # 最後と最初を繋いでみる（角があれば垂直で落とされるはず）
                target_idx_pose_indices.append(target_idx_pose_indices[0])
                # 角度の変曲点が3つ以上ある場合、エッジが分断されてるとみなす
                for ssi, esi in zip(target_idx_pose_indices, target_idx_pose_indices[1:]):
                    target_edge_lines = (
                        edge_lines[ssi : (esi + 1)] if 0 <= ssi < esi else edge_lines[ssi:] + edge_lines[: (esi + 1)]
                    )
                    slice_target_idx_poses = (
                        target_idx_poses[(ssi + 1) : (esi + 1)]
                        if 0 <= ssi < esi
                        else target_idx_poses[(ssi + 1) :] + target_idx_poses[: (esi + 1)]
                    )

                    logger.debug(
                        f"ssi[{ssi}], esi[{esi}], edge[{[(ed, virtual_vertices[ed].vidxs()) for ed in target_edge_lines]}]"
                    )

                    if -threshold <= np.mean(np.abs(np.diff(slice_target_idx_poses))) <= threshold:
                        # 同一方向の傾きに変化がなければ、水平方向
                        if 1 == len(all_edge_lines):
                            horizonal_edge_lines.append([])
                        horizonal_edge_lines[-1].append(target_edge_lines)
                        logger.info(
                            "-- %s: 水平エッジ %s",
                            material_name,
                            [(ed, virtual_vertices[ed].vidxs()) for ed in target_edge_lines],
                        )
                    else:
                        # 同一方向の傾きに変化があれば、垂直方向
                        if 1 == len(all_edge_lines):
                            vertical_edge_lines.append([])
                        vertical_edge_lines[-1].append(target_edge_lines)
                        logger.info(
                            "-- %s: 垂直エッジ %s",
                            material_name,
                            [(ed, virtual_vertices[ed].vidxs()) for ed in target_edge_lines],
                        )
            else:
                # 変曲点がほぼない場合、エッジが均一に水平に繋がってるとみなす(一枚物は有り得ない)
                if len(horizonal_edge_lines) == 0:
                    horizonal_edge_lines.append([])
                horizonal_edge_lines[-1].append(edge_lines)

        logger.debug(f"horizonal[{horizonal_edge_lines}]")
        logger.debug(f"vertical[{vertical_edge_lines}]")

        logger.info("%s: 水平エッジの上下判定", material_name)

        # 親ボーンとの距離
        horizonal_distances = []
        for edge_lines in horizonal_edge_lines:
            line_horizonal_distances = []
            for edge_line in edge_lines:
                horizonal_poses = []
                for edge_key in edge_line:
                    horizonal_poses.append(virtual_vertices[edge_key].position().data())
                line_horizonal_distances.append(
                    np.mean(
                        np.linalg.norm(np.array(horizonal_poses) - parent_bone.position.data(), ord=2, axis=1), axis=0
                    )
                )
            horizonal_distances.append(np.mean(line_horizonal_distances))

        # 水平方向を上下に分ける
        horizonal_total_mean_distance = np.mean(horizonal_distances)
        logger.debug(f"distance[{horizonal_total_mean_distance}], [{horizonal_distances}]")

        bottom_horizonal_edge_lines = []
        top_horizonal_edge_lines = []
        for n, (hd, hel) in enumerate(zip(horizonal_distances, horizonal_edge_lines)):
            if hd > horizonal_total_mean_distance:
                # 遠い方が下(BOTTOM)
                bottom_horizonal_edge_lines.append(hel)
                logger.debug(f"[{n:02d}-horizonal-bottom] {hel}")
            else:
                # 近い方が上(TOP)
                top_horizonal_edge_lines.append(hel)
                logger.debug(f"[{n:02d}-horizonal-top] {hel}")

        if not top_horizonal_edge_lines:
            logger.warning(
                "物理方向に対して水平な上部エッジが見つけられなかった為、処理を終了します。\n"
                + "「エッジの抽出準備」で検出されたエッジが2つある場合、「特殊形状」オプションで「エッジ不定形」を選んでいただくと、水平エッジとして処理されます。\n"
                + "VRoid製スカートの場合、上部のベルト部分が含まれていないかご確認ください。",
                decoration=MLogger.DECORATION_BOX,
            )
            return None, None, None, None

        top_keys = []
        top_degrees = {}
        top_edge_poses = []
        for ti, thel in enumerate(top_horizonal_edge_lines):
            for hi, the in enumerate(thel):
                for ei, thkey in enumerate(the):
                    top_edge_poses.append(virtual_vertices[thkey].position().data())

        if not bottom_horizonal_edge_lines:
            logger.warning(
                "物理方向に対して水平な下部エッジが見つけられなかった為、処理を終了します。\n"
                + "「エッジの抽出準備」で検出されたエッジが2つある場合、「特殊形状」オプションで「エッジ不定形」を選んでいただくと、水平エッジとして処理されます。\n"
                + "VRoid製スカートの場合、上部のベルト部分が含まれていないかご確認ください。",
                decoration=MLogger.DECORATION_BOX,
            )
            return None, None, None, None

        logger.info("--------------")
        bottom_keys = []
        bottom_degrees = {}
        bottom_edge_poses = []
        for bi, bhel in enumerate(bottom_horizonal_edge_lines):
            for hi, bhe in enumerate(bhel):
                for ei, bhkey in enumerate(bhe):
                    pos = virtual_vertices[bhkey].position().data()
                    bottom_edge_poses.append(pos)

        bottom_edge_mean_pos = MVector3D(np.mean(bottom_edge_poses, axis=0))
        # 真後ろに最も近い位置
        bottom_edge_start_pos = MVector3D(list(sorted(bottom_edge_poses, key=lambda x: (-x[2], abs(x[0]), -x[1])))[0])
        if np.isclose(bottom_edge_start_pos.z(), bottom_edge_mean_pos.z(), rtol=0.01):
            # 奥行きがほぼ同じ場合、ずらす
            bottom_edge_mean_pos.setZ(bottom_edge_mean_pos.z() + (np.sign(bottom_edge_mean_pos.z() * 100)))

        top_edge_mean_pos = MVector3D(np.mean(top_edge_poses, axis=0))
        # 真後ろに最も近い位置
        top_edge_start_pos = MVector3D(list(sorted(top_edge_poses, key=lambda x: (-x[2], abs(x[0]), -x[1])))[0])
        if np.isclose(top_edge_start_pos.z(), top_edge_mean_pos.z(), rtol=0.01):
            # 奥行きがほぼ同じ場合、ずらす
            top_edge_mean_pos.setZ(bottom_edge_mean_pos.z())
        logger.info(
            "%s: 水平エッジ上部 開始位置: [%s], 中央位置[%s]", material_name, top_edge_start_pos.to_log(), top_edge_mean_pos.to_log()
        )

        for ti, thel in enumerate(top_horizonal_edge_lines):
            for hi, the in enumerate(thel):
                top_keys.extend(the)
                for ei, thkey in enumerate(the):
                    top_degree0, top_degree1 = self.calc_arc_degree(
                        top_edge_start_pos, top_edge_mean_pos, virtual_vertices[thkey].position(), base_vertical_axis
                    )
                    top_degrees[(thkey, 0)] = top_degree0
                    top_degrees[(thkey, 1)] = top_degree1
                    logger.info(
                        "%s: 水平エッジ上部(%s-%s-%s): %s -> [%s, %s]",
                        material_name,
                        f"{(ti + 1):03d}",
                        f"{(hi + 1):03d}",
                        f"{(ei + 1):03d}",
                        virtual_vertices[thkey].vidxs(),
                        round(top_degree0, 3),
                        round(top_degree1, 3),
                    )

        logger.info("--------------")

        logger.info(
            "%s: 水平エッジ下部 開始位置: [%s], 中央位置[%s]",
            material_name,
            bottom_edge_start_pos.to_log(),
            bottom_edge_mean_pos.to_log(),
        )

        for bi, bhel in enumerate(bottom_horizonal_edge_lines):
            for hi, bhe in enumerate(bhel):
                bottom_keys.extend(bhe)
                for ei, bhkey in enumerate(bhe):
                    bottom_degree0, bottom_degree1 = self.calc_arc_degree(
                        bottom_edge_start_pos,
                        bottom_edge_mean_pos,
                        virtual_vertices[bhkey].position(),
                        base_vertical_axis,
                    )
                    bottom_degrees[(bhkey, 0)] = bottom_degree0
                    bottom_degrees[(bhkey, 1)] = bottom_degree1
                    logger.info(
                        "%s: 水平エッジ下部(%s-%s-%s): %s -> [%s, %s]",
                        material_name,
                        f"{(bi + 1):03d}",
                        f"{(hi + 1):03d}",
                        f"{(ei + 1):03d}",
                        virtual_vertices[bhkey].vidxs(),
                        round(bottom_degree0, 3),
                        round(bottom_degree1, 3),
                    )

        logger.info("--------------------------")
        all_vkeys_list = []
        all_scores = []
        for bi, bhel in enumerate(bottom_horizonal_edge_lines):
            all_vkeys_list.append([])
            all_scores.append([])
            for hi, bhe in enumerate(bhel):
                if hi > 0:
                    all_vkeys_list.append([])
                    all_scores.append([])
                for ki, bottom_edge_key in enumerate(bhe):
                    bottom_degree0 = bottom_degrees[(bottom_edge_key, 0)]
                    bottom_degree1 = bottom_degrees[(bottom_edge_key, 1)]
                    # 近いdegreeのものを選ぶ(大体近いでOK)
                    top_degree0_diffs = np.abs(np.array(list(top_degrees.values())) - bottom_degree0)
                    top_degree1_diffs = np.abs(np.array(list(top_degrees.values())) - bottom_degree1)
                    top_sub_idx = np.argmin([np.min(top_degree0_diffs), np.min(top_degree1_diffs)])
                    top_idx = np.argmin([top_degree0_diffs, top_degree1_diffs][top_sub_idx])
                    top_edge_key, _ = list(top_degrees.keys())[top_idx]

                    logger.debug(
                        f"** start: ({bi:02d}-{hi:02d}), top[{top_edge_key}({virtual_vertices[top_edge_key].vidxs()})][{round(top_degrees[(top_edge_key, top_sub_idx)], 3)}], bottom[{bottom_edge_key}({virtual_vertices[bottom_edge_key].vidxs()})][{round(bottom_degrees[(bottom_edge_key, 0)], 3)}]"
                    )

                    vkeys, vscores = self.create_vertex_line_map(
                        top_edge_key,
                        bottom_edge_key,
                        bottom_edge_key,
                        virtual_vertices,
                        top_keys,
                        bottom_keys,
                        base_vertical_axis,
                        [bottom_edge_key],
                        [1],
                        param_option,
                    )
                    logger.info(
                        "頂点ルート走査[%s-%s-%s]: 始端: %s -> 終端: %s, スコア: %s",
                        f"{(bi + 1):03d}",
                        f"{(hi + 1):03d}",
                        f"{(ki + 1):03d}",
                        virtual_vertices[vkeys[0]].vidxs() if vkeys else "NG",
                        virtual_vertices[vkeys[-1]].vidxs(),
                        round(
                            np.average(
                                vscores, weights=list(reversed((np.arange(1, len(vscores) + 1) ** 2).tolist()))
                            ),
                            4,
                        )
                        if vscores and param_option["special_shape"] == logger.transtext("プリーツ")
                        else round(np.mean(vscores), 4)
                        if vscores
                        else "-",
                    )
                    if len(vkeys) > 1:
                        all_vkeys_list[-1].append(vkeys)
                        all_scores[-1].append(vscores)

        logger.info("%s: 絶対頂点マップの生成", material_name)
        vertex_maps = []

        for midx, (vkeys_list, scores) in enumerate(zip(all_vkeys_list, all_scores)):
            logger.info("-- 絶対頂点マップ: %s個目: ---------", midx + 1)

            logger.info("-- 絶対頂点マップ[%s]: 頂点ルート決定", midx + 1)

            logger.debug("------------------")
            top_key_cnts = dict(Counter([vkeys[0] for vkeys in vkeys_list]))
            target_regists = [False for _ in range(len(vkeys_list))]
            if np.max(list(top_key_cnts.values())) > 1:
                # 同じ始端から2つ以上の末端に繋がっている場合
                for top_key, cnt in top_key_cnts.items():
                    total_scores = {}
                    for x, (vkeys, ss) in enumerate(zip(vkeys_list, scores)):
                        if vkeys[0] == top_key:
                            if cnt > 1:
                                if param_option["special_shape"] == logger.transtext("プリーツ"):
                                    # プリーツで2個以上同じ始端から出ている場合はスコアの重み付平均を取る
                                    total_scores[x] = np.average(
                                        ss, weights=list(reversed((np.arange(1, len(ss) + 1) ** 2).tolist()))
                                    )
                                else:
                                    # 2個以上同じ始端から出ている場合はスコア平均を取る
                                    total_scores[x] = np.mean(ss)
                                logger.debug(
                                    f"target top: [{virtual_vertices[vkeys[0]].vidxs()}], bottom: [{virtual_vertices[vkeys[-1]].vidxs()}], total: {round(total_scores[x], 3)}"
                                )
                            else:
                                # 後はそのまま登録
                                total_scores[x] = cnt
                    # 最もスコアが大きい列を登録対象とする
                    target_regists[list(total_scores.keys())[np.argmax(list(total_scores.values()))]] = True
            else:
                # 全部1個ずつ繋がっている場合はそのまま登録
                target_regists = [True for _ in range(len(vkeys_list))]

            logger.debug(f"target_regists: {target_regists}")

            logger.info("-- 絶対頂点マップ[%s]: マップ生成", midx + 1)

            # XYの最大と最小の抽出
            xu = np.unique([i for i, vks in enumerate(vkeys_list) if target_regists[i]])
            # Yは全マップの最大値分確保する
            yu = np.unique([i for vks_list in all_vkeys_list for vks in vks_list for i, vk in enumerate(vks)])

            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((len(yu), len(xu), 3), (np.nan, np.nan, np.nan))
            vertex_display_map = np.full((len(yu), len(xu)), "None  ")
            registed_vertices = []

            prev_xx = 0
            xx = 0
            for x, vkeys in enumerate(vkeys_list):
                if not target_regists[x]:
                    # 登録対象外の場合、接続仮想頂点リストにだけは追加する
                    for y, vkey in enumerate(vkeys):
                        if np.isnan(vertex_map[y, prev_xx]).any():
                            continue
                        prev_vv = virtual_vertices[tuple(vertex_map[y, prev_xx])]
                        vv = virtual_vertices[vkey]
                        prev_vv.connected_vvs.extend(vv.connected_vvs)
                    continue

                for y, vkey in enumerate(vkeys):
                    vv = virtual_vertices[vkey]
                    vv.map_bones[midx] = None
                    if not vv.vidxs():
                        prev_xx = xx
                        continue

                    logger.debug(f"x: {x}, y: {y}, vv: {vkey}, vidxs: {vv.vidxs()}")

                    vertex_map[y, xx] = vkey
                    vertex_display_map[y, xx] = ":".join([str(v) for v in vv.vidxs()])
                    registed_vertices.append(vkey)

                    if xx > 0 and not target_regists[x - 1]:
                        # 前のキーが対象外だった場合、前のキーに今のキーの接続情報を追加しておく
                        if not np.isnan(vertex_map[y, xx - 1]).any():
                            prev_vv = virtual_vertices[tuple(vertex_map[y, xx - 1])]
                            prev_vv.connected_vvs.append(vkey)
                            vv.connected_vvs.append(prev_vv.key)

                    prev_xx = xx

                xx += 1
                logger.debug("-------")

            if not target_regists[0]:
                # 最初のキーが対象外だった場合、最後のキーの接続情報を最初に追加しておく
                for y, vkey in enumerate(vkeys_list[np.max(np.where(target_regists))]):
                    if not np.isnan(vertex_map[y, 0]).any():
                        prev_vv = virtual_vertices[tuple(vertex_map[y, 0])]
                        prev_vv.connected_vvs.append(vkey)
                        vv = virtual_vertices[vkey]
                        vv.connected_vvs.append(prev_vv.key)

            remove_yidxs = []
            for v_yidx in range(vertex_map.shape[0]):
                if np.isnan(vertex_map[v_yidx, :]).all():
                    # 全部nanの場合、削除対象
                    remove_yidxs.append(v_yidx)

            vertex_map = np.delete(vertex_map, remove_yidxs, axis=0)
            vertex_display_map = np.delete(vertex_display_map, remove_yidxs, axis=0)

            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                        # ない場合、仮想頂点を設定する
                        nearest_v_yidx = (
                            np.where(~np.isnan(vertex_map[:, v_xidx, 0]))[0][
                                np.argmin(np.abs(np.where(~np.isnan(vertex_map[:, v_xidx, 0]))[0] - v_yidx))
                            ]
                            if np.where(~np.isnan(vertex_map[:, v_xidx, 0]))[0].any()
                            else 0
                        )
                        nearest_v_xidx = (
                            np.where(~np.isnan(vertex_map[nearest_v_yidx, :, 0]))[0][
                                np.argmin(np.abs(np.where(~np.isnan(vertex_map[nearest_v_yidx, :, 0]))[0] - v_xidx))
                            ]
                            if np.where(~np.isnan(vertex_map[nearest_v_yidx, :, 0]))[0].any()
                            else 0
                        )
                        nearest_above_v_yidx = (
                            np.where(~np.isnan(vertex_map[:nearest_v_yidx, nearest_v_xidx, 0]))[0][
                                np.argmin(
                                    np.abs(
                                        np.where(~np.isnan(vertex_map[:nearest_v_yidx, nearest_v_xidx, 0]))[0]
                                        - nearest_v_yidx
                                    )
                                )
                            ]
                            if np.where(~np.isnan(vertex_map[:nearest_v_yidx, nearest_v_xidx, 0]))[0].any()
                            else 0
                        )
                        above_yidx = (
                            np.where(~np.isnan(vertex_map[:v_yidx, v_xidx, 0]))[0][
                                np.argmin(np.abs(np.where(~np.isnan(vertex_map[:v_yidx, v_xidx, 0]))[0] - v_yidx))
                            ]
                            if np.where(~np.isnan(vertex_map[:v_yidx, v_xidx, 0]))[0].any()
                            else 0
                        )
                        above_above_yidx = (
                            np.where(~np.isnan(vertex_map[:above_yidx, v_xidx, 0]))[0][
                                np.argmin(
                                    np.abs(np.where(~np.isnan(vertex_map[:above_yidx, v_xidx, 0]))[0] - above_yidx)
                                )
                            ]
                            if np.where(~np.isnan(vertex_map[:above_yidx, v_xidx, 0]))[0].any()
                            else 0
                        )

                        nearest_vv = virtual_vertices[tuple(vertex_map[nearest_v_yidx, nearest_v_xidx])]
                        nearest_above_vv = virtual_vertices[tuple(vertex_map[nearest_above_v_yidx, nearest_v_xidx])]
                        above_vv = virtual_vertices[tuple(vertex_map[above_yidx, v_xidx])]
                        above_above_vv = virtual_vertices[tuple(vertex_map[above_above_yidx, v_xidx])]

                        # 直近頂点の上下距離
                        nearest_distance = nearest_vv.position().distanceToPoint(nearest_above_vv.position())

                        # ボーン進行方向(x)
                        x_direction_pos = (above_above_vv.position() - above_vv.position()).normalized()
                        # ボーン進行方向に対しての横軸(y)
                        y_direction_pos = MVector3D(1, 0, 0)
                        # ボーン進行方向に対しての縦軸(z)
                        z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
                        above_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)

                        mat = MMatrix4x4()
                        mat.setToIdentity()
                        mat.translate(above_vv.position())
                        mat.rotate(above_qq)

                        # 仮想頂点の位置
                        target_position = mat * MVector3D(0, -nearest_distance, 0)
                        target_key = target_position.to_key(threshold)
                        if target_key not in virtual_vertices:
                            virtual_vertices[target_key] = VirtualVertex(target_key)
                        virtual_vertices[target_key].positions.append(target_position.data())
                        vertex_map[v_yidx, v_xidx] = target_key

            vertex_maps.append(vertex_map)

            logger.info(
                "\n".join([", ".join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]),
                translate=False,
            )
            logger.info("-- 絶対頂点マップ: %s個目:終了 ---------", midx + 1)

            logger.debug("-----------------------")

        return vertex_maps, virtual_vertices, remaining_vertices, back_vertices

    def calc_arc_degree(
        self, start_pos: MVector3D, mean_pos: MVector3D, target_pos: MVector3D, base_vertical_axis: MVector3D
    ):
        start_normal_pos = (start_pos - mean_pos).normalized()
        target_normal_pos = (target_pos - mean_pos).normalized()
        qq = MQuaternion.rotationTo(start_normal_pos, target_normal_pos)
        degree = qq.toDegreeSign(base_vertical_axis)
        if np.isclose(MVector3D.dotProduct(start_normal_pos, target_normal_pos), -1):
            # ほぼ真後ろを向いてる場合、固定で180度を入れておく
            degree = 180
        # if degree < 0:
        #     # マイナスになった場合、360を足しておく
        #     degree += 360

        return (degree, degree + 360)

    def create_vertex_line_map(
        self,
        top_edge_key: tuple,
        bottom_edge_key: tuple,
        from_key: tuple,
        virtual_vertices: dict,
        top_keys: list,
        bottom_keys: list,
        base_vertical_axis: MVector3D,
        vkeys: list,
        vscores: list,
        param_option: dict,
        loop=0,
    ):

        if loop > 500:
            return None, None

        from_vv = virtual_vertices[from_key]
        from_pos = from_vv.position()

        top_vv = virtual_vertices[top_edge_key]
        top_pos = top_vv.position()

        bottom_vv = virtual_vertices[bottom_edge_key]
        bottom_pos = bottom_vv.position()

        local_next_base_pos = MVector3D(1, 0, 0)

        # ボーン進行方向(x)
        top_x_pos = (top_pos - bottom_pos).normalized()
        # ボーン進行方向に対しての縦軸(y)
        top_y_pos = top_vv.normal().normalized()
        # ボーン進行方向に対しての横軸(z)
        top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
        top_qq = MQuaternion.fromDirection(top_z_pos, top_y_pos)
        logger.debug(
            f" - top({top_vv.vidxs()}): x[{top_x_pos.to_log()}], y[{top_y_pos.to_log()}], z[{top_z_pos.to_log()}]"
        )

        scores = []
        for n, to_key in enumerate(from_vv.connected_vvs):
            if to_key not in virtual_vertices:
                scores.append(0)
                logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_key}], 対象外")
                continue

            to_vv = virtual_vertices[to_key]
            to_pos = to_vv.position()

            direction_dot = MVector3D.dotProduct(
                (from_pos - bottom_pos).normalized(), (to_pos - from_pos).normalized()
            )
            if to_key in vkeys or (from_key not in bottom_keys and direction_dot <= 0):
                # 到達済み、反対方向のベクトルには行かせない
                scores.append(0)
                logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], 対象外")
                continue

            mat = MMatrix4x4()
            mat.setToIdentity()
            mat.translate(from_pos)
            mat.rotate(top_qq)

            local_next_vpos = (mat.inverted() * to_pos).normalized()

            vec_yaw1 = (local_next_base_pos * MVector3D(1, 0, 1)).normalized()
            vec_yaw2 = (local_next_vpos * MVector3D(1, 0, 1)).normalized()
            yaw_score = calc_ratio(MVector3D.dotProduct(vec_yaw1, vec_yaw2), -1, 1, 0, 1)

            vec_pitch1 = (local_next_base_pos * MVector3D(0, 1, 1)).normalized()
            vec_pitch2 = (local_next_vpos * MVector3D(0, 1, 1)).normalized()
            pitch_score = calc_ratio(MVector3D.dotProduct(vec_pitch1, vec_pitch2), -1, 1, 0, 1)

            vec_roll1 = (local_next_base_pos * MVector3D(1, 1, 0)).normalized()
            vec_roll2 = (local_next_vpos * MVector3D(1, 1, 0)).normalized()
            roll_score = calc_ratio(MVector3D.dotProduct(vec_roll1, vec_roll2), -1, 1, 0, 1)

            score = (yaw_score * 20) + pitch_score + (roll_score * 2)

            scores.append(score * (2 if to_key in top_keys else 1))

            logger.debug(
                f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], local_next_vpos[{local_next_vpos.to_log()}], score: [{score}], yaw_score: {round(yaw_score, 5)}, pitch_score: {round(pitch_score, 5)}, roll_score: {round(roll_score, 5)}"
            )

        if np.count_nonzero(scores) == 0:
            # スコアが付けられなくなったら終了
            return vkeys, vscores

        # 最もスコアの高いINDEXを採用
        nearest_idx = np.argmax(scores)
        vertical_key = from_vv.connected_vvs[nearest_idx]

        # 前の辺との内積差を考慮する（プリーツライン選択用・デフォルトはTOPキーの場合に加算してるので、その分割り引く）
        prev_diff_dot = (
            MVector3D.dotProduct(
                (virtual_vertices[vkeys[0]].position() - virtual_vertices[vkeys[1]].position()).normalized(),
                (virtual_vertices[vertical_key].position() - virtual_vertices[vkeys[0]].position()).normalized(),
            )
            if len(vkeys) > 1 and param_option["special_shape"] == logger.transtext("プリーツ")
            else 0.5
            if vertical_key in top_keys
            else 1
        )

        logger.debug(
            f"direction: from: [{virtual_vertices[from_key].vidxs()}], to: [{virtual_vertices[vertical_key].vidxs()}], prev_diff_dot[{round(prev_diff_dot, 4)}]"
        )

        vscores.insert(0, np.max(scores) * prev_diff_dot)
        vkeys.insert(0, vertical_key)

        if vertical_key in top_keys:
            # 上端に辿り着いたら終了
            return vkeys, vscores

        return self.create_vertex_line_map(
            top_edge_key,
            bottom_edge_key,
            vertical_key,
            virtual_vertices,
            top_keys,
            bottom_keys,
            base_vertical_axis,
            vkeys,
            vscores,
            param_option,
            loop + 1,
        )

    def get_edge_lines(self, edge_line_pairs: dict, start_vkey: tuple, edge_lines: list, edge_vkeys: list, loop=0):
        if len(edge_vkeys) >= len(edge_line_pairs.keys()) or loop > 500:
            return start_vkey, edge_lines, edge_vkeys

        if not start_vkey:
            # X(中央揃え) - Z(降順) - Y(降順)
            sorted_edge_line_pairs = sorted(
                list(set(edge_line_pairs.keys()) - set(edge_vkeys)), key=lambda x: (abs(x[0]), -x[2], -x[1])
            )
            start_vkey = sorted_edge_line_pairs[0]
            edge_lines.append([start_vkey])
            edge_vkeys.append(start_vkey)

        for next_vkey in sorted(edge_line_pairs[start_vkey], key=lambda x: (x[0], x[2], -x[1])):
            if next_vkey not in edge_vkeys:
                edge_lines[-1].append(next_vkey)
                edge_vkeys.append(next_vkey)
                start_vkey, edge_lines, edge_vkeys = self.get_edge_lines(
                    edge_line_pairs, next_vkey, edge_lines, edge_vkeys, loop + 1
                )

        return None, edge_lines, edge_vkeys

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
        all_regist_bones: dict,
        all_bone_connected: dict,
        base_map_idx: int,
    ):
        regist_bones = all_regist_bones[base_map_idx]

        target_v_yidx = (
            np.max(np.where(regist_bones[:v_yidx, v_xidx]))
            if regist_bones[:v_yidx, v_xidx].any()
            else np.max(np.where(regist_bones[:v_yidx, np.max(np.where(regist_bones[:v_yidx, :v_xidx])[1])]))
            if np.where(regist_bones[:v_yidx, :v_xidx])[1].any()
            else regist_bones.shape[0] - 1
        )
        target_v_xidx = (
            np.max(np.where(regist_bones[target_v_yidx, : (v_xidx + 1)])[0])
            if np.where(regist_bones[target_v_yidx, : (v_xidx + 1)])[0].any()
            else regist_bones.shape[1] - 1
        )

        max_v_xidx = (
            np.max(np.where(regist_bones[target_v_yidx, :]))
            if regist_bones[target_v_yidx, :].any() and np.max(np.where(regist_bones[target_v_yidx, :])) >= v_xidx
            else np.max(np.where(regist_bones[: (target_v_yidx + 1), :])[1])
            if np.where(regist_bones[: (target_v_yidx + 1), :])[1].any()
            and np.max(np.where(regist_bones[: (target_v_yidx + 1), :])[1]) >= v_xidx
            else regist_bones.shape[1] - 1
        )

        max_v_yidx = (
            np.max(np.where(regist_bones[:, target_v_xidx])[0])
            if np.where(regist_bones[:, target_v_xidx])[0].any()
            else np.max(np.where(regist_bones[:, : (target_v_xidx + 1)])[0])
            if np.where(regist_bones[:, : (target_v_xidx + 1)])[0].any()
            else regist_bones.shape[0] - 1
        )

        prev_xidx = 0
        prev_map_idx = base_map_idx
        prev_connected = False
        if v_xidx == 0:
            if len(all_bone_connected) == 1 and all_bone_connected[base_map_idx][target_v_yidx, max_v_xidx:].any():
                # 最後が先頭と繋がっている場合(最後の有効ボーンから最初までがどこか繋がっている場合）、最後と繋ぐ
                prev_xidx = max_v_xidx
                prev_map_idx = base_map_idx
                prev_connected = True
            elif (
                base_map_idx > 0
                and all_bone_connected[list(all_bone_connected.keys())[base_map_idx - 1]].shape[0] > target_v_yidx
                and all_bone_connected[list(all_bone_connected.keys())[base_map_idx - 1]][target_v_yidx, -1].any()
            ):
                prev_map_idx = list(all_bone_connected.keys())[base_map_idx - 1]
                prev_connected = True
                # 最後が先頭と繋がっている場合(最後の有効ボーンから最初までがどこか繋がっている場合）、最後と繋ぐ
                if (
                    tuple(vertex_maps[prev_map_idx][target_v_yidx, -1])
                    == tuple(vertex_maps[base_map_idx][target_v_yidx, v_xidx])
                    and all_bone_connected[prev_map_idx][target_v_yidx, -2].any()
                ):
                    # 前のボーンが同じ仮想頂点であり、かつそのもうひとつ前と繋がっている場合
                    prev_xidx = (
                        np.max(np.where(all_regist_bones[prev_map_idx][target_v_yidx, :-1]))
                        if all_regist_bones[prev_map_idx][target_v_yidx, :-1].any()
                        else np.max(np.where(all_regist_bones[prev_map_idx][: (target_v_yidx + 1), :-1])[1])
                        if np.where(all_regist_bones[prev_map_idx][: (target_v_yidx + 1), :-1])[1].any()
                        else 0
                    )
                else:
                    # 前のボーンの仮想頂点が自分と違う場合、そのまま前のを採用
                    prev_xidx = (
                        np.max(np.where(all_regist_bones[prev_map_idx][target_v_yidx, :]))
                        if all_regist_bones[prev_map_idx][target_v_yidx, :].any()
                        else np.max(np.where(all_regist_bones[prev_map_idx][: (target_v_yidx + 1), :])[1])
                        if np.where(all_regist_bones[prev_map_idx][: (target_v_yidx + 1), :])[1].any()
                        else 0
                    )
        else:
            # 1番目以降は、自分より前で、ボーンが登録されている最も近いの
            prev_xidx = (
                np.max(np.where(regist_bones[v_yidx, :v_xidx]))
                if regist_bones[v_yidx, :v_xidx].any()
                else np.max(np.where(regist_bones[: (v_yidx + 1), :v_xidx])[1])
                if np.where(regist_bones[: (v_yidx + 1), :v_xidx])[1].any()
                else 0
            )
            prev_connected = True

        next_xidx = max_v_xidx
        next_map_idx = base_map_idx
        next_connected = False
        if v_xidx >= max_v_xidx:
            if len(all_bone_connected) == 1 and all_bone_connected[base_map_idx][v_yidx, max_v_xidx:].any():
                # 最後が先頭と繋がっている場合(最後の有効ボーンから最初までがどこか繋がっている場合）、最後と繋ぐ（マップが1つの場合）
                next_xidx = 0
                next_map_idx = base_map_idx
                next_connected = True
            elif (
                base_map_idx < len(all_bone_connected) - 1
                and all_bone_connected[base_map_idx][v_yidx, max_v_xidx:].any()
            ):
                # マップが複数、かつ最後ではない場合（次の先頭と繋ぐ）
                next_xidx = 0
                next_map_idx = base_map_idx + 1
                next_connected = True
            elif (
                base_map_idx == len(all_bone_connected) - 1
                and all_bone_connected[base_map_idx][v_yidx, max_v_xidx:].any()
            ):
                # マップが複数かつ最後である場合（最初の先頭と繋ぐ）
                next_map_idx = 0
                next_connected = True

                if (
                    tuple(vertex_maps[next_map_idx][v_yidx, 0]) == tuple(vertex_maps[base_map_idx][v_yidx, v_xidx])
                    and all_bone_connected[next_map_idx][v_yidx, 0].any()
                ):
                    # 次のボーンが同じ仮想頂点であり、かつそのもうひとつ先と繋がっている場合
                    next_xidx = 1
                else:
                    # 次のボーンの仮想頂点が自分と違う場合、そのまま前のを採用
                    next_xidx = 0
        else:
            # maxより前は、自分より前で、ボーンが登録されている最も近いの
            next_xidx = (
                np.min(np.where(regist_bones[v_yidx, (v_xidx + 1) :])) + (v_xidx + 1)
                if regist_bones[v_yidx, (v_xidx + 1) :].any()
                else np.min(np.where(regist_bones[: (v_yidx + 1), (v_xidx + 1) :])[1]) + (v_xidx + 1)
                if np.where(regist_bones[: (v_yidx + 1), (v_xidx + 1) :])[1].any()
                else max_v_xidx
            )
            next_connected = True

        above_yidx = 0
        if v_yidx > 0:
            above_yidx = (
                np.max(np.where(regist_bones[:v_yidx, v_xidx]))
                if regist_bones[:v_yidx, v_xidx].any()
                else np.max(np.where(regist_bones[:v_yidx, :v_xidx])[0])
                if np.where(regist_bones[:v_yidx, :v_xidx])[0].any()
                else 0
            )

        below_yidx = regist_bones.shape[0] - 1
        if v_yidx < regist_bones.shape[0] - 1:
            below_yidx = (
                np.min(np.where(regist_bones[(v_yidx + 1) :, v_xidx])) + (v_yidx + 1)
                if regist_bones[(v_yidx + 1) :, v_xidx].any()
                else np.min(np.where(regist_bones[(v_yidx + 1) :, :v_xidx])[0]) + (v_yidx + 1)
                if np.where(regist_bones[(v_yidx + 1) :, :v_xidx])[0].any()
                else regist_bones.shape[0] - 1
            )

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
            max_v_yidx,
            max_v_xidx,
        )


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def randomname(n) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def calc_intersect(vP0: MVector3D, vP1: MVector3D, vQ0: MVector3D, vQ1: MVector3D) -> MVector3D:
    P0 = vP0.data()
    P1 = vP1.data()
    Q0 = vQ0.data()
    Q1 = vQ1.data()

    # Direction vectors
    DP = P1 - P0
    DQ = Q1 - Q0

    # start difference vector
    PQ = Q0 - P0

    # Find values
    a = DP.dot(DP)
    b = DP.dot(DQ)
    c = DQ.dot(DQ)
    d = DP.dot(PQ)
    e = DQ.dot(PQ)

    # Find discriminant
    DD = a * c - b * b

    if np.isclose(DD, 0):
        return (vP0 + vQ0) / 2

    # Find parameters for the closest points on lines
    tt = (b * e - c * d) / DD
    uu = (a * e - b * d) / DD

    Pt = P0 + tt * DP
    Qu = Q0 + uu * DQ

    return MVector3D(Pt)


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
