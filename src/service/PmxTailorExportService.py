# -*- coding: utf-8 -*-
#
import logging
import os
import traceback
import numpy as np
import itertools
import math
import copy
import bezier

from module.MOptions import MExportOptions
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, Sdef, RigidBodyParam, IkLink, Ik, BoneMorphData # noqa
from mmd.PmxWriter import PmxWriter
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException
import utils.MBezierUtils as MBezierUtils

logger = MLogger(__name__, level=1)


class PmxTailorExportService():
    def __init__(self, options: MExportOptions):
        self.options = options

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = "PmxTailor変換処理実行\n------------------------\nexeバージョン: {version_name}\n".format(version_name=self.options.version_name) \

            service_data_txt = "{service_data_txt}　元モデル: {pmx_model}\n".format(service_data_txt=service_data_txt,
                                    pmx_model=os.path.basename(self.options.pmx_model.path)) # noqa

            for pidx, param_option in enumerate(self.options.param_options):
                service_data_txt = f"{service_data_txt}\n　【No.{pidx + 1}】 --------- "    # noqa
                service_data_txt = f"{service_data_txt}\n　　材質: {param_option['material_name']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　剛体グループ: {param_option['rigidbody'].collision_group + 1}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　検出度: {param_option['similarity']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　細かさ: {param_option['fineness']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　質量: {param_option['mass']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　柔らかさ: {param_option['air_resistance']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　張り: {param_option['shape_maintenance']}"    # noqa

            logger.info(service_data_txt, decoration=MLogger.DECORATION_BOX)

            model = self.options.pmx_model
            model.comment += "\r\n\r\n物理: PmxTailor"

            for pidx, param_option in enumerate(self.options.param_options):
                if not self.create_physics(model, param_option):
                    return False

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            PmxWriter().write(model, self.options.output_path)

            logger.info("出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title="成功")

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("PmxTailor変換処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical("PmxTailor変換処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
        finally:
            logging.shutdown()

    def create_physics(self, model: PmxModel, param_option: dict):
        model.comment += f"\r\n材質: {param_option['material_name']} --------------"    # noqa
        model.comment += f"\r\n　　剛体グループ: {param_option['rigidbody'].collision_group + 1}"    # noqa
        model.comment += f"\r\n　　細かさ: {param_option['fineness']}"    # noqa
        model.comment += f"\r\n　　質量: {param_option['mass']}"    # noqa
        model.comment += f"\r\n　　空気抵抗: {param_option['air_resistance']}"    # noqa
        model.comment += f"\r\n　　形状維持: {param_option['shape_maintenance']}"    # noqa

        if param_option['exist_physics_clear']:
            # 既存材質削除フラグONの場合
            logger.info(f"{param_option['material_name']}: 既存材質削除", decoration=MLogger.DECORATION_LINE)

            model = self.clear_exist_physics(model, param_option, param_option['material_name'])

        logger.info(f"{param_option['material_name']}: 頂点マップ生成", decoration=MLogger.DECORATION_LINE)

        vertex_maps, vertex_connecteds, duplicate_vertices, registed_iidxs, duplicate_indices, index_combs_by_vpos \
            = self.create_vertex_map(model, param_option, param_option['material_name'])
        
        # 各頂点の有効INDEX数が最も多いものをベースとする
        map_cnt = []
        for vertex_map in vertex_maps:
            map_cnt.append(np.count_nonzero(vertex_map >= 0))
        
        if len(map_cnt) == 0:
            logger.warning("有効な頂点マップが生成されななかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return False
        
        vertex_map_orders = [k for k in np.argsort(-np.array(map_cnt)) if map_cnt[k] > np.max(map_cnt) * 0.5]
        
        logger.info(f"【{param_option['material_name']}】ボーン生成", decoration=MLogger.DECORATION_LINE)

        root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances \
            = self.create_bone(model, param_option, vertex_map_orders, vertex_maps, vertex_connecteds)

        vertex_remaining_set = set(model.material_vertices[param_option['material_name']])

        for base_map_idx in vertex_map_orders:
            logger.info(f"【{param_option['material_name']}(No.{base_map_idx + 1})】 ウェイト分布", decoration=MLogger.DECORATION_LINE)

            self.create_weight(model, param_option, vertex_maps[base_map_idx], vertex_connecteds[base_map_idx], duplicate_vertices, \
                               all_registed_bone_indexs[base_map_idx], all_bone_horizonal_distances[base_map_idx], all_bone_vertical_distances[base_map_idx], \
                               vertex_remaining_set)

        if len(list(vertex_remaining_set)) > 0:
            logger.info(f"【{param_option['material_name']}】 残ウェイト分布", decoration=MLogger.DECORATION_LINE)
            
            self.create_remaining_weight(model, param_option, vertex_maps, duplicate_vertices, all_registed_bone_indexs, all_bone_horizonal_distances, \
                                         all_bone_vertical_distances, registed_iidxs, duplicate_indices, index_combs_by_vpos, vertex_remaining_set, vertex_map_orders, \
                                         sorted(list(set(list(range(len(vertex_maps)))) - set(vertex_map_orders))))
 
        if param_option['back_material_name']:
            logger.info(f"【{param_option['material_name']}】 裏面ウェイト分布", decoration=MLogger.DECORATION_LINE)

            self.create_back_weight(model, param_option)
 
        for base_map_idx in vertex_map_orders:
            logger.info(f"【{param_option['material_name']}(No.{base_map_idx + 1})】 剛体生成", decoration=MLogger.DECORATION_LINE)

            self.create_rigidbody(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx])

            logger.info(f"【{param_option['material_name']}(No.{base_map_idx + 1})】 ジョイント生成", decoration=MLogger.DECORATION_LINE)

            self.create_joint(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx], all_bone_horizonal_distances[base_map_idx])

        return True

    def create_joint(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict, bone_horizonal_distances: dict):
        # ジョイント生成
        created_joints = {}

        # 略称
        abb_name = param_option['abb_name']
        # 縦ジョイント情報
        param_vertical_joint = param_option['vertical_joint']
        # 横ジョイント情報
        param_horizonal_joint = param_option['horizonal_joint']
        # 斜めジョイント情報
        param_diagonal_joint = param_option['diagonal_joint']
        # 逆ジョイント情報
        param_reverse_joint = param_option['reverse_joint']

        v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
        prev_joint_cnt = 0

        max_vy = max(v_yidxs)
        middle_vy = (max(v_yidxs)) * 0.3
        min_vy = 0
        xs = np.arange(min_vy, max_vy, step=1)
    
        if param_vertical_joint:
            coefficient = param_option['vertical_joint_coefficient']

            vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
            vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
            vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

            vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
            vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
            vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

            vertical_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x()]])), xs)             # noqa
            vertical_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y()]])), xs)             # noqa
            vertical_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z()]])), xs)             # noqa

            vertical_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x()]])), xs)             # noqa
            vertical_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y()]])), xs)             # noqa
            vertical_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z()]])), xs)             # noqa

            vertical_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x()]])), xs)             # noqa
            vertical_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y()]])), xs)             # noqa
            vertical_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z()]])), xs)             # noqa

            vertical_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x()]])), xs)             # noqa
            vertical_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y()]])), xs)             # noqa
            vertical_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z()]])), xs)             # noqa

        if param_horizonal_joint:
            coefficient = param_option['horizonal_joint_coefficient']

            if param_option['bone_thinning_out']:
                horizonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x()]])), xs)             # noqa
                horizonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y()]])), xs)             # noqa
                horizonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z()]])), xs)             # noqa

                horizonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x()]])), xs)             # noqa
                horizonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y()]])), xs)             # noqa
                horizonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z()]])), xs)             # noqa
            else:
                max_x = 0
                for yi, v_yidx in enumerate(v_yidxs):
                    v_xidxs = list(registed_bone_indexs[v_yidx].keys())
                    max_x = len(v_xidxs) if max_x < len(v_xidxs) else max_x

                x_distances = np.zeros((len(registed_bone_indexs), max_x + 1))
                for yi, v_yidx in enumerate(v_yidxs):
                    v_xidxs = list(registed_bone_indexs[v_yidx].keys())
                    if v_yidx < len(vertex_connected) and vertex_connected[v_yidx]:
                        # 繋がってる場合、最後に最初のボーンを追加する
                        v_xidxs += [list(registed_bone_indexs[v_yidx].keys())[0]]
                    elif len(registed_bone_indexs[v_yidx]) > 2:
                        # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                        v_xidxs += [list(registed_bone_indexs[v_yidx].keys())[-2]]

                    for xi, (prev_v_xidx, next_v_xidx) in enumerate(zip(v_xidxs[:-1], v_xidxs[1:])):
                        prev_v_xidx_diff = np.array(list(registed_bone_indexs[v_yidx].values())) - registed_bone_indexs[v_yidx][prev_v_xidx]
                        prev_v_xidx = list(registed_bone_indexs[v_yidx].values())[(0 if prev_v_xidx == 0 else np.argmax(prev_v_xidx_diff))]
                        prev_bone_name = self.get_bone_name(abb_name, v_yidx + 1, prev_v_xidx + 1)

                        next_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidx].values())) - registed_bone_indexs[v_yidx][next_v_xidx])
                        next_v_xidx = list(registed_bone_indexs[v_yidx].values())[(0 if next_v_xidx == 0 else np.argmin(next_v_xidx_diff))]
                        next_bone_name = self.get_bone_name(abb_name, v_yidx + 1, next_v_xidx + 1)
                        
                        x_distances[yi, xi] = tmp_all_bones[prev_bone_name]["bone"].position.distanceToPoint(tmp_all_bones[next_bone_name]["bone"].position)
                x_ratio_distances = np.array(x_distances) / (np.min(x_distances, axis=0) * 2)

                horizonal_limit_min_mov_xs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.x())
                horizonal_limit_min_mov_ys = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.y())
                horizonal_limit_min_mov_zs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.z())

                horizonal_limit_max_mov_xs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.x())
                horizonal_limit_max_mov_ys = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.y())
                horizonal_limit_max_mov_zs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.z())

            horizonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x()]])), xs)             # noqa
            horizonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y()]])), xs)             # noqa
            horizonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z()]])), xs)             # noqa

            horizonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x()]])), xs)             # noqa
            horizonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y()]])), xs)             # noqa
            horizonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z()]])), xs)             # noqa

            horizonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x()]])), xs)             # noqa
            horizonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y()]])), xs)             # noqa
            horizonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z()]])), xs)             # noqa

            horizonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
            horizonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
            horizonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

        if param_diagonal_joint:
            coefficient = param_option['diagonal_joint_coefficient']

            diagonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x()]])), xs)             # noqa
            diagonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y()]])), xs)             # noqa
            diagonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z()]])), xs)             # noqa

            diagonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x()]])), xs)             # noqa
            diagonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y()]])), xs)             # noqa
            diagonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z()]])), xs)             # noqa

            diagonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x()]])), xs)             # noqa
            diagonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y()]])), xs)             # noqa
            diagonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z()]])), xs)             # noqa

            diagonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x()]])), xs)             # noqa
            diagonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y()]])), xs)             # noqa
            diagonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z()]])), xs)             # noqa

            diagonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x()]])), xs)             # noqa
            diagonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y()]])), xs)             # noqa
            diagonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z()]])), xs)             # noqa

            diagonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
            diagonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
            diagonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

        if param_reverse_joint:
            coefficient = param_option['reverse_joint_coefficient']

            reverse_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x()]])), xs)             # noqa
            reverse_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y()]])), xs)             # noqa
            reverse_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z()]])), xs)             # noqa

            reverse_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x()]])), xs)             # noqa
            reverse_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y()]])), xs)             # noqa
            reverse_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z()]])), xs)             # noqa

            reverse_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x()]])), xs)             # noqa
            reverse_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y()]])), xs)             # noqa
            reverse_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z()]])), xs)             # noqa

            reverse_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x()]])), xs)             # noqa
            reverse_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y()]])), xs)             # noqa
            reverse_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z()]])), xs)             # noqa

            reverse_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x()]])), xs)             # noqa
            reverse_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y()]])), xs)             # noqa
            reverse_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z()]])), xs)             # noqa

            reverse_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x()]])), xs)             # noqa
            reverse_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y()]])), xs)             # noqa
            reverse_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
                [param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z()]])), xs)             # noqa

        for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
            below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
            logger.debug(f"yi: {yi}, below_v_xidxs: {below_v_xidxs}")

            if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
                # 繋がってる場合、最後に最初のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
            elif len(registed_bone_indexs[below_v_yidx]) > 2:
                # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]
            logger.debug(f"yi: {yi}, below_v_xidxs: {below_v_xidxs}")

            for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
                prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
                next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
                below_v_yno = below_v_yidx + 1

                prev_below_bone_name = self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)
                prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
                next_below_bone_name = self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)
                next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

                prev_above_bone_name = tmp_all_bones[prev_below_bone_name]["parent"]
                prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
                prev_above_v_yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
                
                next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - registed_bone_indexs[below_v_yidx][next_below_v_xidx])
                next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
                next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
                next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

                if param_vertical_joint and prev_above_bone_name != prev_below_bone_name:
                    # 縦ジョイント
                    joint_name = f'↓|{prev_above_bone_name}|{prev_below_bone_name}'

                    if not (joint_name in created_joints or prev_above_bone_name not in model.rigidbodies or prev_below_bone_name not in model.rigidbodies):
                        # 未登録のみ追加
                        
                        # 縦ジョイント
                        joint_vec = prev_below_bone_position

                        # 回転量
                        joint_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_above_bone_name)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[prev_below_bone_name].index,
                                      joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yidx], vertical_limit_min_mov_ys[yidx], vertical_limit_min_mov_zs[yidx]), \
                                      MVector3D(vertical_limit_max_mov_xs[yidx], vertical_limit_max_mov_ys[yidx], vertical_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(vertical_limit_min_rot_xs[yidx]), math.radians(vertical_limit_min_rot_ys[yidx]), math.radians(vertical_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(vertical_limit_max_rot_xs[yidx]), math.radians(vertical_limit_max_rot_ys[yidx]), math.radians(vertical_limit_max_rot_zs[yidx])),
                                      MVector3D(vertical_spring_constant_mov_xs[yidx], vertical_spring_constant_mov_ys[yidx], vertical_spring_constant_mov_zs[yidx]), \
                                      MVector3D(vertical_spring_constant_rot_xs[yidx], vertical_spring_constant_rot_ys[yidx], vertical_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint.name] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                            prev_joint_cnt = len(created_joints) // 200
                        
                        if param_reverse_joint:
                            # 逆ジョイント
                            joint_name = f'↑|{prev_below_bone_name}|{prev_above_bone_name}'

                            if not (joint_name in created_joints or prev_below_bone_name not in model.rigidbodies or prev_above_bone_name not in model.rigidbodies):
                                # 未登録のみ追加
                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, model.rigidbodies[prev_above_bone_name].index,
                                              joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
                                              MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
                                              MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
                                              MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
                                              MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
                                              MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))  # noqa
                                created_joints[joint.name] = joint

                                if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                    logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                                    prev_joint_cnt = len(created_joints) // 200
                                
                if param_horizonal_joint:
                    # 横ジョイント
                    if xi < len(below_v_xidxs) - 1 and prev_above_bone_name != next_above_bone_name:
                        joint_name = f'→|{prev_above_bone_name}|{next_above_bone_name}'

                        if not (joint_name in created_joints or prev_above_bone_name not in model.rigidbodies or next_above_bone_name not in model.rigidbodies):
                            # 未登録のみ追加
                            
                            joint_vec = np.mean([prev_above_bone_position, prev_below_bone_position, \
                                                 next_above_bone_position, next_below_bone_position])

                            # 回転量
                            joint_axis = (next_above_bone_position - prev_above_bone_position).normalized()
                            joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                            joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                            joint_euler = joint_rotation_qq.toEulerAngles()
                            joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                            yidx, xidx = self.disassemble_bone_name(prev_above_bone_name)

                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[next_above_bone_name].index,
                                          joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi, xi], horizonal_limit_min_mov_ys[yi, xi], horizonal_limit_min_mov_zs[yi, xi]), \
                                          MVector3D(horizonal_limit_max_mov_xs[yi, xi], horizonal_limit_max_mov_ys[yi, xi], horizonal_limit_max_mov_zs[yi, xi]),
                                          MVector3D(math.radians(horizonal_limit_min_rot_xs[yidx]), math.radians(horizonal_limit_min_rot_ys[yidx]), math.radians(horizonal_limit_min_rot_zs[yidx])),
                                          MVector3D(math.radians(horizonal_limit_max_rot_xs[yidx]), math.radians(horizonal_limit_max_rot_ys[yidx]), math.radians(horizonal_limit_max_rot_zs[yidx])),
                                          MVector3D(horizonal_spring_constant_mov_xs[yidx], horizonal_spring_constant_mov_ys[yidx], horizonal_spring_constant_mov_zs[yidx]), \
                                          MVector3D(horizonal_spring_constant_rot_xs[yidx], horizonal_spring_constant_rot_ys[yidx], horizonal_spring_constant_rot_zs[yidx]))    # noqa
                            created_joints[joint.name] = joint

                            if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                                prev_joint_cnt = len(created_joints) // 200
                            
                        if param_reverse_joint:
                            # 横逆ジョイント
                            joint_name = f'←|{next_above_bone_name}|{prev_above_bone_name}'

                            if not (joint_name in created_joints or prev_above_bone_name not in model.rigidbodies or next_above_bone_name not in model.rigidbodies):
                                # 未登録のみ追加
                                
                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[next_above_bone_name].index, model.rigidbodies[prev_above_bone_name].index,
                                              joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
                                              MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
                                              MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
                                              MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
                                              MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
                                              MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))      # noqa
                                created_joints[joint.name] = joint

                                if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                    logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                                    prev_joint_cnt = len(created_joints) // 200
                                
                if param_diagonal_joint:
                    # ＼ジョイント
                    joint_name = f'＼|{prev_above_bone_name}|{next_below_bone_name}'

                    if not (joint_name in created_joints or prev_above_bone_name not in model.rigidbodies or next_below_bone_name not in model.rigidbodies):
                        # 未登録のみ追加
                        
                        # ＼ジョイント
                        joint_vec = np.mean([prev_below_bone_position, next_below_bone_position])

                        # 回転量
                        joint_axis = (next_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis_up = (prev_below_bone_position - next_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_above_bone_name)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[next_below_bone_name].index,
                                      joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
                                      MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
                                      MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
                                      MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint.name] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                            prev_joint_cnt = len(created_joints) // 200
                        
                    # ／ジョイント ---------------
                    joint_name = f'／|{prev_below_bone_name}|{next_above_bone_name}'

                    if not (joint_name in created_joints or prev_below_bone_name not in model.rigidbodies or next_below_bone_name not in model.rigidbodies):
                        # 未登録のみ追加
                    
                        # ／ジョイント

                        # 回転量
                        joint_axis = (prev_below_bone_position - next_above_bone_position).normalized()
                        joint_axis_up = (next_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_below_bone_name)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, model.rigidbodies[next_above_bone_name].index,
                                      joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
                                      MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
                                      MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
                                      MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint.name] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info(f"-- ジョイント: {len(created_joints)}個目:終了")
                            prev_joint_cnt = len(created_joints) // 200
                        
        for joint_name in sorted(created_joints.keys()):
            # ジョイントを登録
            joint = created_joints[joint_name]
            joint.index = len(model.joints)
            model.joints[joint.name] = joint
    
    def create_rigidbody(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict):
        # 剛体生成
        created_rigidbodies = {}
        # 剛体の質量
        created_rigidbody_masses = {}
        created_rigidbody_linear_dampinges = {}
        created_rigidbody_angular_dampinges = {}
        prev_rigidbody_cnt = 0

        # 略称
        abb_name = param_option['abb_name']
        # 剛体情報
        param_rigidbody = param_option['rigidbody']
        # 剛体係数
        coefficient = param_option['rigidbody_coefficient']

        v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
        rigidbody_limit_thicks = np.linspace(0.3, 0.1, len(v_yidxs))

        for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
            below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
            logger.debug(f"yi: {yi}, below_v_xidxs: {below_v_xidxs}")

            if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
                # 繋がってる場合、最後に最初のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
            elif len(registed_bone_indexs[below_v_yidx]) > 2:
                # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]
            logger.debug(f"yi: {yi}, below_v_xidxs: {below_v_xidxs}")

            for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
                prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
                next_below_v_xidx = registed_bone_indexs[below_v_yidx][next_below_v_xidx]
                next_below_v_xno = next_below_v_xidx + 1
                below_v_yno = below_v_yidx + 1

                prev_below_bone_name = self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)
                prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
                next_below_bone_name = self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)
                next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

                prev_above_bone_name = tmp_all_bones[prev_below_bone_name]["parent"]
                prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
                prev_above_v_yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
    
                next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - next_below_v_xidx)
                next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
                next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
                next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

                if prev_above_bone_name in created_rigidbodies:
                    continue

                prev_above_bone_index = -1
                if prev_above_bone_name in model.bones:
                    prev_above_bone_index = model.bones[prev_above_bone_name].index

                # 剛体の傾き
                shape_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
                shape_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
                shape_axis_up.setY(0)
                shape_axis_cross = MVector3D.crossProduct(shape_axis, shape_axis_up).normalized()

                shape_rotation_qq = MQuaternion.fromDirection(shape_axis, shape_axis_cross)
                if round(prev_below_bone_position.y(), 2) != round(prev_above_bone_position.y(), 2):
                    shape_rotation_qq *= MQuaternion.fromEulerAngles(0, 0, -90)
                    shape_rotation_qq *= MQuaternion.fromEulerAngles(-90, 0, 0)
                    shape_rotation_qq *= MQuaternion.fromEulerAngles(0, -90, 0)

                shape_rotation_euler = shape_rotation_qq.toEulerAngles()

                if round(prev_below_bone_position.y(), 2) == round(prev_above_bone_position.y(), 2):
                    shape_rotation_euler.setX(90)
                    
                shape_rotation_radians = MVector3D(math.radians(shape_rotation_euler.x()), math.radians(shape_rotation_euler.y()), math.radians(shape_rotation_euler.z()))

                # 剛体の大きさ
                x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
                y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                shape_size = MVector3D(max(0.25, x_size * 0.5), max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])

                # 剛体の位置
                rigidbody_vertical_vec = ((prev_below_bone_position - prev_above_bone_position) / 2)
                if round(prev_below_bone_position.y(), 3) != round(prev_above_bone_position.y(), 3):
                    mat = MMatrix4x4()
                    mat.setToIdentity()
                    mat.translate(prev_above_bone_position)
                    mat.rotate(shape_rotation_qq)
                    # ローカルY軸方向にボーンの長さの半分を上げる
                    mat.translate(MVector3D(0, -prev_below_bone_position.distanceToPoint(prev_above_bone_position) / 2, 0))
                    shape_position = mat * MVector3D()
                else:
                    shape_position = prev_above_bone_position + rigidbody_vertical_vec + MVector3D(0, rigidbody_limit_thicks[yi] / 2, 0)

                # 根元はボーン追従剛体、それ以降は物理剛体
                mode = 0 if yi == len(v_yidxs) - 2 else 1
                shape_type = param_rigidbody.shape_type
                if prev_above_bone_name not in model.bones:
                    # 登録ボーンの対象外である場合、余っているので球にしておく
                    ball_size = np.max([0.25, x_size * 0.5, y_size * 0.5])
                    shape_size = MVector3D(ball_size, ball_size, ball_size)
                    shape_type = 0
                mass = param_rigidbody.param.mass * shape_size.x() * shape_size.y() * shape_size.z()
                linear_damping = param_rigidbody.param.linear_damping * shape_size.x() * shape_size.y() * shape_size.z()
                angular_damping = param_rigidbody.param.angular_damping * shape_size.x() * shape_size.y() * shape_size.z()
                rigidbody = RigidBody(prev_above_bone_name, prev_above_bone_name, prev_above_bone_index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
                                      shape_type, shape_size, shape_position, shape_rotation_radians, \
                                      mass, linear_damping, angular_damping, param_rigidbody.param.restitution, param_rigidbody.param.friction, mode)
                # 別途保持しておく
                created_rigidbodies[rigidbody.name] = rigidbody
                created_rigidbody_masses[rigidbody.name] = mass
                created_rigidbody_linear_dampinges[rigidbody.name] = linear_damping
                created_rigidbody_angular_dampinges[rigidbody.name] = angular_damping

                if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 200 > prev_rigidbody_cnt:
                    logger.info(f"-- 剛体: {len(created_rigidbodies)}個目:終了")
                    prev_rigidbody_cnt = len(created_rigidbodies) // 200
                
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
            rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
            rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
                min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
            rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
                min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa
            
            model.rigidbodies[rigidbody.name] = rigidbody

    def create_weight(self, model: PmxModel, param_option: dict, vertex_map: np.ndarray, vertex_connected: dict, duplicate_vertices: dict, \
                      registed_bone_indexs: dict, bone_horizonal_distances: dict, bone_vertical_distances: dict, vertex_remaining_set: set):
        # ウェイト分布
        prev_weight_cnt = 0
        weight_cnt = 0

        # 略称
        abb_name = param_option['abb_name']

        v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
        for above_v_yidx, below_v_yidx in zip(v_yidxs[1:], v_yidxs[:-1]):
            above_v_xidxs = list(registed_bone_indexs[above_v_yidx].keys())
            below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
            # 繋がってる場合、最後に最初のボーンを追加する
            if above_v_yidx < len(vertex_connected) and vertex_connected[above_v_yidx]:
                above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[0]]
            if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]

            for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
                prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
                next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
                below_v_yno = below_v_yidx + 1

                prev_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)]
                next_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)]
                prev_above_bone = model.bones[model.bone_indexes[prev_below_bone.parent_index]]
                next_above_bone = model.bones[model.bone_indexes[next_below_bone.parent_index]]

                _, prev_above_v_xidx = self.disassemble_bone_name(prev_above_bone.name, registed_bone_indexs[above_v_yidx])
                _, next_above_v_xidx = self.disassemble_bone_name(next_above_bone.name, registed_bone_indexs[above_v_yidx])

                if xi > 0 and (next_below_v_xidx == registed_bone_indexs[below_v_yidx][list(registed_bone_indexs[below_v_yidx].keys())[0]] \
                               or next_above_v_xidx == registed_bone_indexs[above_v_yidx][list(registed_bone_indexs[above_v_yidx].keys())[0]]):
                    # nextが最初のボーンである場合、最後まで
                    v_map = vertex_map[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                    b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                    b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                else:
                    v_map = vertex_map[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                    b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                    b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                
                for vi, v_vertices in enumerate(v_map):
                    for vhi, vertex_idx in enumerate(v_vertices):
                        if vertex_idx < 0:
                            continue

                        horizonal_distance = np.sum(b_h_distances[vi, 1:])
                        v_horizonal_distance = np.sum(b_h_distances[vi, 1:(vhi + 1)])
                        vertical_distance = np.sum(b_v_distances[1:, vhi])
                        v_vertical_distance = np.sum(b_v_distances[1:(vi + 1), vhi])

                        prev_above_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
                        prev_below_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))
                        next_above_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
                        next_below_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))

                        if below_v_yidx == v_yidxs[0]:
                            # 最下段は末端ボーンにウェイトを振らない
                            # 処理対象全ボーン名
                            weight_names = np.array([prev_above_bone.name, next_above_bone.name])
                            # ウェイト
                            total_weights = np.array([prev_above_weight + prev_below_weight, next_above_weight + next_below_weight])
                        else:
                            # 処理対象全ボーン名
                            weight_names = np.array([prev_above_bone.name, next_above_bone.name, prev_below_bone.name, next_below_bone.name])
                            # ウェイト
                            total_weights = np.array([prev_above_weight, next_above_weight, prev_below_weight, next_below_weight])

                        if len(np.nonzero(total_weights)[0]) > 0:
                            weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                            weight_idxs = np.argsort(weights)
                            v = model.vertex_dict[vertex_idx]
                            vertex_remaining_set -= set(duplicate_vertices[v.position.to_log()])

                            for vvidx in duplicate_vertices[v.position.to_log()]:
                                vv = model.vertex_dict[vvidx]
                                if vv.deform.index0 == model.bones[param_option['parent_bone_name']].index:
                                    # 重複頂点にも同じウェイトを割り当てる
                                    if np.count_nonzero(weights) == 1:
                                        vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                                    elif np.count_nonzero(weights) == 2:
                                        vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
                                    else:
                                        vv.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
                                                          model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
                                                          weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])

                                weight_cnt += 1
                                if weight_cnt > 0 and weight_cnt // 1000 > prev_weight_cnt:
                                    logger.info(f"-- 頂点ウェイト: {weight_cnt}個目:終了")
                                    prev_weight_cnt = weight_cnt // 1000
        
        return vertex_remaining_set

    def create_remaining_weight(self, model: PmxModel, param_option: dict, vertex_maps: dict, duplicate_vertices: dict, all_registed_bone_indexs: dict, \
                                all_bone_horizonal_distances: dict, all_bone_vertical_distances: dict, registed_iidxs: list, duplicate_indices: dict, \
                                index_combs_by_vpos: dict, vertex_remaining_set: set, boned_base_map_idxs: list, base_map_idxs: list):
        # ウェイト分布
        prev_weight_cnt = 0
        weight_cnt = 0

        # 基準頂点マップ以外の頂点が残っていたら、それも割り当てる
        for vertex_idx in list(vertex_remaining_set):
            v = model.vertex_dict[vertex_idx]
            if vertex_idx < 0:
                continue
            
            vertex_distances = {}
            for boned_map_idx in boned_base_map_idxs:
                # 登録済み頂点との距離を測る（一番近いのと似たウェイト構成になるはず）
                boned_vertex_map = vertex_maps[boned_map_idx]

                for yi in range(boned_vertex_map.shape[0] - 1):
                    for xi in range(boned_vertex_map.shape[1] - 1):
                        if boned_vertex_map[yi, xi] >= 0:
                            vi = boned_vertex_map[yi, xi]
                            vertex_distances[vi] = v.position.distanceToPoint(model.vertex_dict[vi].position)

            nearest_vi = list(vertex_distances.keys())[np.argmin(list(vertex_distances.values()))]
            nearest_v = model.vertex_dict[nearest_vi]
            nearest_deform = nearest_v.deform

            if type(nearest_deform) is Bdef1:
                v.deform = Bdef1(nearest_deform.index0)
            elif type(nearest_deform) is Bdef2:
                weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
                weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]

                bone1_distance = v.position.distanceToPoint(weight_bone1.position)
                bone2_distance = v.position.distanceToPoint(weight_bone2.position) if nearest_deform.weight0 < 1 else 0

                weight_names = np.array([weight_bone1.name, weight_bone2.name])
                total_weights = np.array([bone1_distance / (bone1_distance + bone2_distance), bone2_distance / (bone1_distance + bone2_distance)])
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                for vvidx in duplicate_vertices[v.position.to_log()]:
                    vv = model.vertex_dict[vvidx]
                    vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])

                    if vv.deform.index0 == model.bones[param_option['parent_bone_name']].index:
                        # 重複頂点にも同じウェイトを割り当てる
                        if np.count_nonzero(weights) == 1:
                            vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                        elif np.count_nonzero(weights) == 2:
                            vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
                        else:
                            vv.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
                                              model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
                                              weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])
            elif type(nearest_deform) is Bdef4:
                weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
                weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]
                weight_bone3 = model.bones[model.bone_indexes[nearest_deform.index2]]
                weight_bone4 = model.bones[model.bone_indexes[nearest_deform.index3]]

                bone1_distance = v.position.distanceToPoint(weight_bone1.position) if nearest_deform.weight0 > 0 else 0
                bone2_distance = v.position.distanceToPoint(weight_bone2.position) if nearest_deform.weight1 > 0 else 0
                bone3_distance = v.position.distanceToPoint(weight_bone3.position) if nearest_deform.weight2 > 0 else 0
                bone4_distance = v.position.distanceToPoint(weight_bone4.position) if nearest_deform.weight3 > 0 else 0
                all_distance = bone1_distance + bone2_distance + bone3_distance + bone4_distance

                weight_names = np.array([weight_bone1.name, weight_bone2.name, weight_bone3.name, weight_bone4.name])
                total_weights = np.array([bone1_distance / all_distance, bone2_distance / all_distance, bone3_distance / all_distance, bone4_distance / all_distance])
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                for vvidx in duplicate_vertices[v.position.to_log()]:
                    vv = model.vertex_dict[vvidx]
                    vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])

                    if vv.deform.index0 == model.bones[param_option['parent_bone_name']].index:
                        # 重複頂点にも同じウェイトを割り当てる
                        if np.count_nonzero(weights) == 1:
                            vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                        elif np.count_nonzero(weights) == 2:
                            vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
                        else:
                            vv.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
                                              model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
                                              weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 100 > prev_weight_cnt:
                logger.info(f"-- 残頂点ウェイト: {weight_cnt}個目:終了")
                prev_weight_cnt = weight_cnt // 100

    def create_back_weight(self, model: PmxModel, param_option: dict):
        # ウェイト分布
        prev_weight_cnt = 0
        weight_cnt = 0

        for vertex_idx in list(model.material_vertices[param_option['back_material_name']]):
            bv = model.vertex_dict[vertex_idx]

            front_vertex_distances = {}
            for front_vertex_idx in list(model.material_vertices[param_option['material_name']]):
                front_vertex_distances[front_vertex_idx] = bv.position.distanceToPoint(model.vertex_dict[front_vertex_idx].position)

            # 直近頂点INDEXのウェイトを転写
            copy_front_vertex_idx = list(front_vertex_distances.keys())[np.argmin(list(front_vertex_distances.values()))]
            bv.deform = copy.deepcopy(model.vertex_dict[copy_front_vertex_idx].deform)

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 10 > prev_weight_cnt:
                logger.info(f"-- 裏頂点ウェイト: {weight_cnt}個目:終了")
                prev_weight_cnt = weight_cnt // 10

    def create_bone(self, model: PmxModel, param_option: dict, vertex_map_orders: list, vertex_maps: dict, vertex_connecteds: dict):
        # 中心ボーン生成

        # 略称
        abb_name = param_option['abb_name']
        # 材質名
        material_name = param_option['material_name']

        # 表示枠定義
        model.display_slots[material_name] = DisplaySlot(material_name, material_name, 0, 0)

        root_bone = Bone(f'{abb_name}中心', f'{abb_name}中心', model.bones[param_option['parent_bone_name']].position, \
                         model.bones[param_option['parent_bone_name']].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
        root_bone.index = len(list(model.bones.keys()))

        # ボーン
        model.bones[root_bone.name] = root_bone
        model.bone_indexes[root_bone.index] = root_bone.name
        # 表示枠
        model.display_slots[material_name].references.append((0, model.bones[root_bone.name].index))

        tmp_all_bones = {}
        all_bone_indexes = {}
        all_registed_bone_indexs = {}

        all_bone_horizonal_distances = {}
        all_bone_vertical_distances = {}

        for base_map_idx, vertex_map in enumerate(vertex_maps):
            bone_horizonal_distances = np.zeros(vertex_map.shape)
            bone_vertical_distances = np.zeros(vertex_map.shape)

            # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx, v_xidx - 1] >= 0:
                        now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
                        prev_v_vec = now_v_vec if v_xidx == 0 else model.vertex_dict[vertex_map[v_yidx, v_xidx - 1]].position
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)
                    if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx - 1, v_xidx] >= 0:
                        now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
                        prev_v_vec = now_v_vec if v_yidx == 0 else model.vertex_dict[vertex_map[v_yidx - 1, v_xidx]].position
                        bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)

            all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
            all_bone_vertical_distances[base_map_idx] = bone_vertical_distances

        for base_map_idx in vertex_map_orders:
            vertex_map = vertex_maps[base_map_idx]
            vertex_connected = vertex_connecteds[base_map_idx]

            # Map用INDEXリスト(根元はボーン追従にするため、常に一段だけ)
            v_yidxs = [0] + list(range(1, vertex_map.shape[0] - param_option["vertical_bone_density"] + 1, param_option["vertical_bone_density"]))
            if v_yidxs[-1] < vertex_map.shape[0] - 1:
                # 最下段は必ず登録
                v_yidxs = v_yidxs + [vertex_map.shape[0] - 1]
            
            all_bone_indexes[base_map_idx] = {}
            for yi in range(vertex_map.shape[0]):
                all_bone_indexes[base_map_idx][yi] = {}
                v_xidxs = list(range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]))
                if not vertex_connected[yi] and v_xidxs[-1] < vertex_map.shape[1] - 1:
                    # 繋がってなくて、かつ端っこが登録されていない場合、登録
                    v_xidxs = v_xidxs + [vertex_map.shape[1] - 1]
                max_xi = 0
                for midx, myidxs in all_bone_indexes.items():
                    if midx != base_map_idx and yi in all_bone_indexes[midx]:
                        max_xi = max(list(all_bone_indexes[midx][yi].keys())) + 1
                for xi in v_xidxs:
                    all_bone_indexes[base_map_idx][yi][xi] = xi + max_xi
            
        for base_map_idx in vertex_map_orders:
            vertex_map = vertex_maps[base_map_idx]
            vertex_connected = vertex_connecteds[base_map_idx]
            registed_bone_indexs = {}

            for yi, v_yidx in enumerate(v_yidxs):
                for v_xidx, total_v_xidx in all_bone_indexes[base_map_idx][yi].items():
                    if v_yidx >= vertex_map.shape[0] or v_xidx >= vertex_map.shape[1] or vertex_map[v_yidx, v_xidx] < 0:
                        # 存在しない頂点はスルー
                        continue
                    
                    v = model.vertex_dict[vertex_map[v_yidx, v_xidx]]
                    v_xno = total_v_xidx + 1
                    v_yno = v_yidx + 1

                    # ボーン仮登録
                    bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                    bone = Bone(bone_name, bone_name, v.position, root_bone.index, 0, 0x0000 | 0x0002)
                    bone.local_z_vector = v.normal.copy()
                    tmp_all_bones[bone.name] = {"bone": bone, "parent": root_bone.name, "regist": False}
                    logger.debug(f"tmp_all_bones: {bone.name}, pos: {bone.position.to_log()}")

            # 最下段の横幅最小値(段数単位)
            edge_size = np.min(all_bone_horizonal_distances[base_map_idx][-1, 1:]) * param_option["horizonal_bone_density"]

            for yi, v_yidx in enumerate(v_yidxs):
                prev_xidx = 0
                if v_yidx not in registed_bone_indexs:
                    registed_bone_indexs[v_yidx] = {}

                for v_xidx, total_v_xidx in all_bone_indexes[base_map_idx][yi].items():
                    if v_xidx == 0 or (not vertex_connected[yi] and v_xidx == list(all_bone_indexes[base_map_idx][yi].keys())[-1]) or \
                        not param_option['bone_thinning_out'] or (param_option['bone_thinning_out'] and \
                        np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, (prev_xidx + 1):(v_xidx + 1)]) >= edge_size * 0.9):  # noqa
                        # 前ボーンとの間隔が最下段の横幅平均値より開いている場合、登録対象
                        v_xno = total_v_xidx + 1
                        v_yno = v_yidx + 1

                        # ボーン名
                        bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                        if bone_name not in tmp_all_bones:
                            continue

                        # ボーン本登録
                        bone = tmp_all_bones[bone_name]["bone"]
                        bone.index = len(list(model.bones.keys()))

                        if yi > 0:
                            parent_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi - 1]].values())) - total_v_xidx)

                            # 2段目以降は最も近い親段でv_xidxを探す
                            parent_v_xidx = list(registed_bone_indexs[v_yidxs[yi - 1]].values())[(0 if vertex_connected[yi] and (v_xidxs[-1] + 1) - v_xidx < np.min(parent_v_xidx_diff) else np.argmin(parent_v_xidx_diff))]   # noqa

                            parent_bone = model.bones[self.get_bone_name(abb_name, v_yidxs[yi - 1] + 1, parent_v_xidx + 1)]
                            bone.parent_index = parent_bone.index
                            bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                            bone.local_z_vector *= MVector3D(-1, 1, -1)
                            bone.flag |= 0x0800

                            tmp_all_bones[bone.name]["parent"] = parent_bone.name

                            # 親ボーンの表示先も同時設定
                            parent_bone.tail_index = bone.index
                            parent_bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                            parent_bone.flag |= 0x0001

                            # 表示枠
                            parent_bone.flag |= 0x0008 | 0x0010
                            model.display_slots[material_name].references.append((0, parent_bone.index))

                        model.bones[bone.name] = bone
                        model.bone_indexes[bone.index] = bone.name
                        tmp_all_bones[bone.name]["regist"] = True

                        registed_bone_indexs[v_yidx][v_xidx] = total_v_xidx

                        # 前ボーンとして設定
                        prev_xidx = v_xidx
            
            logger.debug(f"registed_bone_indexs: {registed_bone_indexs}")

            all_registed_bone_indexs[base_map_idx] = registed_bone_indexs

        return root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances

    def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
        return f'{abb_name}-{(v_yno):03d}-{(v_xno):03d}'

    def disassemble_bone_name(self, bone_name: str, v_xidxs=None):
        total_v_xidx = int(bone_name[-3:]) - 1
        now_vidxs = [k for k, v in v_xidxs.items() if v == total_v_xidx] if v_xidxs else [total_v_xidx]
        v_xidx = now_vidxs[0] if now_vidxs else total_v_xidx
        v_yidx = int(bone_name[-7:-4]) - 1

        return v_yidx, v_xidx

    def clear_exist_physics(self, model: PmxModel, param_option: dict, material_name: str):
        logger.info(f"{material_name}: 削除対象抽出（頂点 - ボーン）")

        weighted_bone_indexes = {}
        for vertex_idx in model.material_vertices[material_name]:
            vertex = model.vertex_dict[vertex_idx]
            if type(vertex.deform) is Bdef1:
                if vertex.deform.index0 not in list(weighted_bone_indexes.values()):
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
            elif type(vertex.deform) is Bdef2:
                if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
            elif type(vertex.deform) is Bdef4:
                if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight1 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
                if vertex.deform.index2 not in list(weighted_bone_indexes.values()) and vertex.deform.weight2 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index2]] = vertex.deform.index2
                if vertex.deform.index3 not in list(weighted_bone_indexes.values()) and vertex.deform.weight3 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index3]] = vertex.deform.index3
            elif type(vertex.deform) is Sdef:
                if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
                    weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
        
        not_delete_bone_names = []
        # 他の材質で該当ボーンにウェイト割り当てられている場合、ボーンの削除だけは避ける
        for bone_idx, vertices in model.vertices.items():
            is_not_delete = False
            if bone_idx in list(weighted_bone_indexes.values()) and len(vertices) > 0:
                is_not_delete = False
                for vertex in vertices:
                    if vertex.index not in model.material_vertices[material_name]:
                        is_not_delete = is_not_delete or True
            if is_not_delete:
                not_delete_bone_names.append(model.bone_indexes[bone_idx])
        
        logger.debug('weighted_bone_indexes: %s', ", ".join(list(weighted_bone_indexes.keys())))
        logger.debug('not_delete_bone_names: %s', ", ".join(not_delete_bone_names))
        
        logger.info(f"{material_name}: 削除対象抽出（ボーン - 剛体）")

        weighted_rigidbody_indexes = {}
        for rigidbody in model.rigidbodies.values():
            if rigidbody.index not in list(weighted_rigidbody_indexes.values()) and rigidbody.bone_index in list(weighted_bone_indexes.values()):
                weighted_rigidbody_indexes[rigidbody.name] = rigidbody.index

        logger.debug('weighted_rigidbody_indexes: %s', ", ".join(list(weighted_rigidbody_indexes.keys())))

        logger.info(f"{material_name}: 削除対象抽出（剛体 - ジョイント）")

        weighted_joint_indexes = {}
        for joint in model.joints.values():
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_a in list(weighted_rigidbody_indexes.values()):
                weighted_joint_indexes[joint.name] = joint.name
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_b in list(weighted_rigidbody_indexes.values()):
                weighted_joint_indexes[joint.name] = joint.name

        logger.debug('weighted_joint_indexes: %s', ", ".join((weighted_joint_indexes.keys())))

        logger.info(f"{material_name}: 削除実行")

        # 削除
        for joint_name in weighted_joint_indexes.keys():
            del model.joints[joint_name]

        for rigidbody_name in weighted_rigidbody_indexes.keys():
            del model.rigidbodies[rigidbody_name]

        for bone_name in weighted_bone_indexes.keys():
            if bone_name not in not_delete_bone_names:
                del model.bones[bone_name]

        logger.info(f"{material_name}: INDEX振り直し")

        reset_rigidbodies = {}
        for ridx, (rigidbody_name, rigidbody) in enumerate(model.rigidbodies.items()):
            reset_rigidbodies[rigidbody.index] = {'name': rigidbody_name, 'index': ridx}
            model.rigidbodies[rigidbody_name].index = ridx

        reset_bones = {}
        for bidx, (bone_name, bone) in enumerate(model.bones.items()):
            reset_bones[bone.index] = {'name': bone_name, 'index': bidx}
            model.bones[bone_name].index = bidx
            model.bone_indexes[bidx] = bone_name

        logger.info(f"{material_name}: INDEX再割り当て（ジョイント - 剛体）")

        for jidx, (joint_name, joint) in enumerate(model.joints.items()):
            if joint.rigidbody_index_a in reset_rigidbodies:
                joint.rigidbody_index_a = reset_rigidbodies[joint.rigidbody_index_a]['index']
            if joint.rigidbody_index_b in reset_rigidbodies:
                joint.rigidbody_index_b = reset_rigidbodies[joint.rigidbody_index_b]['index']

        logger.info(f"{material_name}: INDEX再割り当て（剛体 - ボーン）")

        for rigidbody in model.rigidbodies.values():
            if rigidbody.bone_index in reset_bones:
                rigidbody.bone_index = reset_bones[rigidbody.bone_index]['index']

        logger.info(f"{material_name}: INDEX再割り当て（表示枠 - ボーン）")

        for display_slot in model.display_slots.values():
            new_references = []
            for display_type, bone_idx in display_slot.references:
                if display_type == 0:
                    if bone_idx in reset_bones:
                        new_references.append((display_type, reset_bones[bone_idx]['index']))
                else:
                    new_references.append((display_type, bone_idx))
            display_slot.references = new_references

        logger.info(f"{material_name}: INDEX再割り当て（モーフ - ボーン）")

        for morph in model.morphs.values():
            if morph.morph_type == 2:
                new_offsets = []
                for offset in morph.offsets:
                    if type(offset) is BoneMorphData:
                        if offset.bone_index in reset_bones:
                            offset.bone_index = reset_bones[offset.bone_index]['index']
                            new_offsets.append(offset)
                    else:
                        new_offsets.append(offset)
                morph.offsets = new_offsets

        logger.info(f"{material_name}: INDEX再割り当て（参照ボーン - ボーン）")

        for bidx, bone in enumerate(model.bones.values()):
            if bone.parent_index in reset_bones:
                bone.parent_index = reset_bones[bone.parent_index]['index']

            if bone.getConnectionFlag() and bone.tail_index in reset_bones:
                bone.tail_index = reset_bones[bone.tail_index]['index']

            if bone.getExternalRotationFlag() or bone.getExternalTranslationFlag() and bone.effect_index in reset_bones:
                bone.effect_index = reset_bones[bone.effect_index]['index']

            if bone.getIkFlag() and bone.ik.target_index in reset_bones:
                bone.ik.target_index = reset_bones[bone.ik.target_index]['index']
                for link in bone.ik.link:
                    link.bone_index = reset_bones[link.bone_index]['index']

        logger.info(f"{material_name}: INDEX再割り当て（頂点 - ボーン）")

        for vidx, vertex in enumerate(model.vertex_dict.values()):
            if type(vertex.deform) is Bdef1:
                if vertex.deform.index0 in reset_bones:
                    vertex.deform.index0 = reset_bones[vertex.deform.index0]['index']
            elif type(vertex.deform) is Bdef2:
                if vertex.deform.index0 in reset_bones:
                    vertex.deform.index0 = reset_bones[vertex.deform.index0]['index']
                if vertex.deform.index1 in reset_bones:
                    vertex.deform.index1 = reset_bones[vertex.deform.index1]['index']
            elif type(vertex.deform) is Bdef4:
                if vertex.deform.index0 in reset_bones:
                    vertex.deform.index0 = reset_bones[vertex.deform.index0]['index']
                if vertex.deform.index1 in reset_bones:
                    vertex.deform.index1 = reset_bones[vertex.deform.index1]['index']
                if vertex.deform.index2 in reset_bones:
                    vertex.deform.index2 = reset_bones[vertex.deform.index2]['index']
                if vertex.deform.index3 in reset_bones:
                    vertex.deform.index3 = reset_bones[vertex.deform.index3]['index']
            elif type(vertex.deform) is Sdef:
                if vertex.deform.index0 in reset_bones:
                    vertex.deform.index0 = reset_bones[vertex.deform.index0]['index']
                if vertex.deform.index1 in reset_bones:
                    vertex.deform.index1 = reset_bones[vertex.deform.index1]['index']

        return model

    # 頂点を展開した図を作成
    def create_vertex_map(self, model: PmxModel, param_option: dict, material_name: str):
        logger.info(f"{material_name}: 面の抽出")

        logger.info(f"{material_name}: 面の抽出準備①")

        # 位置ベースで重複頂点の抽出
        duplicate_vertices = {}
        for vertex_idx in model.material_vertices[material_name]:
            # 重複頂点の抽出
            vertex = model.vertex_dict[vertex_idx]
            key = vertex.position.to_log()
            if key not in duplicate_vertices:
                duplicate_vertices[key] = []
            if vertex.index not in duplicate_vertices[key]:
                duplicate_vertices[key].append(vertex.index)
            # 一旦ルートボーンにウェイトを一括置換
            vertex.deform = Bdef1(model.bones[param_option['parent_bone_name']].index)

        logger.info(f"{material_name}: 面の抽出準備②")

        # 面組み合わせの生成
        indices_by_vidx = {}
        indices_by_vpos = {}
        index_combs_by_vpos = {}
        duplicate_indices = {}
        below_iidx = None
        max_below_x = -9999
        for index_idx in model.material_indices[material_name]:
            # 頂点の組み合わせから面INDEXを引く
            indices_by_vidx[tuple(sorted(model.indices[index_idx]))] = index_idx
            v0 = model.vertex_dict[model.indices[index_idx][0]]
            v1 = model.vertex_dict[model.indices[index_idx][1]]
            v2 = model.vertex_dict[model.indices[index_idx][2]]
            below_x = MVector3D.dotProduct((v1.position - v0.position).normalized(), MVector3D(1, 0, 0)) * \
                v0.position.distanceToPoint(v1.position) + v1.position.distanceToPoint(v2.position) + v2.position.distanceToPoint(v0.position)
            # below_x = abs(MVector3D.crossProduct((v1.position - v0.position).normalized(), (v2.position - v0.position).normalized()).x())
            if below_x > max_below_x:
                below_iidx = index_idx
                max_below_x = below_x
            # 重複辺（2点）の組み合わせ
            index_combs = list(itertools.combinations(model.indices[index_idx], 2))
            for (iv1, iv2) in index_combs:
                for ivv1, ivv2 in list(itertools.product(duplicate_vertices[model.vertex_dict[iv1].position.to_log()], duplicate_vertices[model.vertex_dict[iv2].position.to_log()])):
                    # 小さいINDEX・大きい頂点INDEXのセットでキー生成
                    key = (min(ivv1, ivv2), max(ivv1, ivv2))
                    if key not in duplicate_indices:
                        duplicate_indices[key] = []
                    if index_idx not in duplicate_indices[key]:
                        duplicate_indices[key].append(index_idx)
            # 頂点別に組み合わせも登録
            for iv in model.indices[index_idx]:
                vpkey = model.vertex_dict[iv].position.to_log()
                if vpkey in duplicate_vertices and vpkey not in index_combs_by_vpos:
                    index_combs_by_vpos[vpkey] = []
                # 同一頂点位置を持つ面のリスト
                if vpkey in duplicate_vertices and vpkey not in indices_by_vpos:
                    indices_by_vpos[vpkey] = []
                if index_idx not in indices_by_vpos[vpkey]:
                    indices_by_vpos[vpkey].append(index_idx)
            for (iv1, iv2) in index_combs:
                # 小さいINDEX・大きい頂点INDEXのセットでキー生成
                key = (min(iv1, iv2), max(iv1, iv2))
                if key not in index_combs_by_vpos[vpkey]:
                    index_combs_by_vpos[vpkey].append(key)

        logger.info(f"{material_name}: 相対頂点マップの生成")

        # 頂点マップ生成(最初の頂点が(0, 0))
        vertex_axis_maps = []
        vertex_coordinate_maps = []
        registed_iidxs = []
        vertical_iidxs = []
        prev_index_cnt = 0

        while len(registed_iidxs) < len(model.material_indices[material_name]):
            if not vertical_iidxs:
                # 切替時はとりあえず一面取り出して判定(二次元配列になる)
                if registed_iidxs:
                    # 出来るだけ真っ直ぐの辺がある面とする
                    below_iidx = None
                    max_below_x = -9999
                    remaining_iidxs = list(set(model.material_indices[material_name]) - set(registed_iidxs))
                    for index_idx in remaining_iidxs:
                        v0 = model.vertex_dict[model.indices[index_idx][0]]
                        v1 = model.vertex_dict[model.indices[index_idx][1]]
                        v2 = model.vertex_dict[model.indices[index_idx][2]]
                        below_x = MVector3D.dotProduct((v1.position - v0.position).normalized(), MVector3D(1, 0, 0)) * \
                            v0.position.distanceToPoint(v1.position) + v1.position.distanceToPoint(v2.position) + v2.position.distanceToPoint(v0.position)
                        if below_x > max_below_x:
                            below_iidx = index_idx
                            max_below_x = below_x
                logger.debug(f'below_iidx: {below_iidx}')
                first_vertex_axis_map, first_vertex_coordinate_map = \
                    self.create_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, {}, {}, below_iidx)
                vertex_axis_maps.append(first_vertex_axis_map)
                vertex_coordinate_maps.append(first_vertex_coordinate_map)
                registed_iidxs.append(below_iidx)
                vertical_iidxs.append(below_iidx)
                
                # 斜めが埋まってる場合、残りの一点を埋める
                vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, diagonal_now_iidxs = \
                    self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
                                                           vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, vertical_iidxs)

                # これで四角形が求められた
                registed_iidxs = list(set(registed_iidxs) | set(diagonal_now_iidxs))
                vertical_iidxs = list(set(vertical_iidxs) | set(diagonal_now_iidxs))

            total_vertical_iidxs = []

            if vertical_iidxs:
                now_vertical_iidxs = vertical_iidxs
                total_vertical_iidxs.extend(now_vertical_iidxs)

                logger.debug(f'縦初回: {total_vertical_iidxs}')

                # 縦辺がいる場合（まずは下方向）
                n = 0
                while n < 200:
                    n += 1
                    vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, now_vertical_iidxs \
                        = self.fill_vertical_indices(model, param_option, duplicate_indices, duplicate_vertices, \
                                                     vertex_axis_maps[-1], vertex_coordinate_maps[-1], indices_by_vpos, indices_by_vidx, \
                                                     registed_iidxs, now_vertical_iidxs, 1)
                    total_vertical_iidxs.extend(now_vertical_iidxs)
                    logger.debug(f'縦下: {total_vertical_iidxs}')

                    if not now_vertical_iidxs:
                        break
                
                if not now_vertical_iidxs:
                    now_vertical_iidxs = vertical_iidxs

                    # 下方向が終わったら上方向
                    n = 0
                    while n < 200:
                        n += 1
                        vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, now_vertical_iidxs \
                            = self.fill_vertical_indices(model, param_option, duplicate_indices, duplicate_vertices, \
                                                         vertex_axis_maps[-1], vertex_coordinate_maps[-1], indices_by_vpos, indices_by_vidx, \
                                                         registed_iidxs, now_vertical_iidxs, -1)
                        total_vertical_iidxs.extend(now_vertical_iidxs)
                        logger.debug(f'縦上: {total_vertical_iidxs}')

                        if not now_vertical_iidxs:
                            break

                logger.debug(f'縦一列: {total_vertical_iidxs} --------------')
                
                # 縦が終わった場合、横に移動する
                min_x, min_y, max_x, max_y = self.get_axis_range(model, vertex_coordinate_maps[-1], registed_iidxs)
                logger.debug(f'axis_range: min_x[{min_x}], min_y[{min_y}], max_x[{max_x}], max_y[{max_y}]')
                
                if not now_vertical_iidxs:
                    # 左方向
                    registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
                                                                                      duplicate_vertices, registed_iidxs, min_x, min_y, max_y, -1)

                    logger.debug(f'横左: {now_vertical_iidxs}')
                    
                if not now_vertical_iidxs:
                    # 右方向
                    registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
                                                                                      duplicate_vertices, registed_iidxs, max_x, min_y, max_y, 1)
                    logger.debug(f'横右: {now_vertical_iidxs}')
                
                vertical_iidxs = now_vertical_iidxs
                
            if not vertical_iidxs:
                remaining_iidxs = list(set(model.material_indices[material_name]) - set(registed_iidxs))
                # 全頂点登録済みの面を潰していく
                for index_idx in remaining_iidxs:
                    iv0, iv1, iv2 = model.indices[index_idx]
                    if iv0 in vertex_axis_maps[-1] and iv1 in vertex_axis_maps[-1] and iv2 in vertex_axis_maps[-1]:
                        registed_iidxs.append(index_idx)
                        logger.debug(f'頂点潰し: {index_idx}')

            if len(registed_iidxs) > 0 and len(registed_iidxs) // 200 > prev_index_cnt:
                logger.info(f"-- 面: {len(registed_iidxs)}個目:終了")
                prev_index_cnt = len(registed_iidxs) // 200
            
        logger.info(f"-- 面: {len(registed_iidxs)}個目:終了")

        logger.info(f"{material_name}: 絶対頂点マップの生成")
        vertex_maps = []
        vertex_connecteds = []

        for midx, (vertex_axis_map, vertex_coordinate_map) in enumerate(zip(vertex_axis_maps, vertex_coordinate_maps)):
            logger.info(f"-- 絶対頂点マップ: {midx + 1}個目: ---------")

            # XYの最大と最小の抽出
            xs = [k[0] for k in vertex_coordinate_map.keys()]
            ys = [k[1] for k in vertex_coordinate_map.keys()]

            # それぞれの出現回数から大体全部埋まってるのを抽出。その中の最大と最小を選ぶ
            xu, x_counts = np.unique(xs, return_counts=True)
            full_xs = [i for i, x in zip(xu, x_counts) if x >= max(x_counts) * 0.6]

            min_x = min(full_xs)
            max_x = max(full_xs)

            yu, y_counts = np.unique(ys, return_counts=True)
            full_ys = [i for i, y in zip(yu, y_counts) if y >= max(y_counts) * 0.6]

            min_y = min(full_ys)
            max_y = max(full_ys)

            logger.debug(f'絶対axis_range: min_x[{min_x}], min_y[{min_y}], max_x[{max_x}], max_y[{max_y}]')
            
            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((max_y - min_y + 1, max_x - min_x + 1), -1)
            vertex_display_map = np.full((max_y - min_y + 1, max_x - min_x + 1), 'None')
            vertex_connected = []
            logger.debug(f'vertex_map.shape: {vertex_map.shape}')

            for vmap in vertex_axis_map.values():
                if vertex_map.shape[0] > vmap['y'] - min_y and vertex_map.shape[1] > vmap['x'] - min_x:
                    logger.debug(f"vertex_map: y[{vmap['y'] - min_y}], x[{vmap['x'] - min_x}]: vidx[{vmap['vidx']}] orgx[{vmap['x']}] orgy[{vmap['y']}] pos[{vmap['position'].to_log()}]")

                    try:
                        vertex_map[vmap['y'] - min_y, vmap['x'] - min_x] = vertex_coordinate_map[(vmap['x'], vmap['y'])][0]
                        vertex_display_map[vmap['y'] - min_y, vmap['x'] - min_x] = ':'.join([str(v) for v in vertex_coordinate_map[(vmap['x'], vmap['y'])]])
                    except Exception as e:
                        logger.debug("vertex_map失敗: %s", e)

            # 左端と右端で面が連続しているかチェック
            for yi in range(vertex_map.shape[0]):
                is_connect = False
                if vertex_map[yi, 0] in model.vertex_dict and vertex_map[yi, -1] in model.vertex_dict:
                    for (iv1, iv2) in list(itertools.product(duplicate_vertices[model.vertex_dict[vertex_map[yi, 0]].position.to_log()], \
                                                             duplicate_vertices[model.vertex_dict[vertex_map[yi, -1]].position.to_log()])):
                        if (min(iv1, iv2), max(iv1, iv2)) in duplicate_indices:
                            is_connect = True
                            break
                vertex_connected.append(is_connect)

            logger.debug(f'vertex_connected: {vertex_connected}')

            vertex_maps.append(vertex_map)
            vertex_connecteds.append(vertex_connected)

            logger.info('\n'.join([', '.join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]))
            logger.info(f"-- 絶対頂点マップ: {midx + 1}個目:終了 ---------")

        return vertex_maps, vertex_connecteds, duplicate_vertices, registed_iidxs, duplicate_indices, index_combs_by_vpos
    
    def get_axis_range(self, model: PmxModel, vertex_coordinate_map: dict, registed_iidxs: list):
        xs = [k[0] for k in vertex_coordinate_map.keys()]
        ys = [k[1] for k in vertex_coordinate_map.keys()]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)
        
        return min_x, min_y, max_x, max_y
    
    def fill_horizonal_now_idxs(self, model: PmxModel, param_option: dict, vertex_axis_map: dict, vertex_coordinate_map: dict, duplicate_indices: dict, \
                                duplicate_vertices: dict, registed_iidxs: list, first_x: int, min_y: int, max_y: int, offset: int):
        now_iidxs = []
        first_vidxs = None
        second_vidxs = None
        for first_y in range(min_y + int((max_y - min_y) / 2), max_y + 1):
            if (first_x, first_y) in vertex_coordinate_map:
                first_vidxs = vertex_coordinate_map[(first_x, first_y)]
                break

        if first_vidxs:
            for second_y in range(first_y + 1, max_y + 1):
                if (first_x, second_y) in vertex_coordinate_map:
                    second_vidxs = vertex_coordinate_map[(first_x, second_y)]
                    break

        if first_vidxs and second_vidxs:
            # 小さいINDEX・大きい頂点INDEXのセットでキー生成
            for (iv1, iv2) in list(itertools.product(first_vidxs, second_vidxs)):
                key = (min(iv1, iv2), max(iv1, iv2))
                if key in duplicate_indices:
                    for index_idx in duplicate_indices[key]:
                        if index_idx in registed_iidxs + now_iidxs:
                            continue
                        
                        # 登録されてない残りの頂点INDEX
                        remaining_vidx = tuple(set(model.indices[index_idx]) - set(duplicate_vertices[model.vertex_dict[iv1].position.to_log()]) \
                            - set(duplicate_vertices[model.vertex_dict[iv2].position.to_log()]))[0]     # noqa
                        remaining_vidxs = duplicate_vertices[model.vertex_dict[remaining_vidx].position.to_log()]
                        if abs(model.vertex_dict[iv1].position.y() - model.vertex_dict[remaining_vidx].position.y()) == \
                            abs(model.vertex_dict[iv2].position.y() - model.vertex_dict[remaining_vidx].position.y()):  # noqa
                            ivy = vertex_axis_map[iv1]['y'] if model.vertex_dict[iv1].position.distanceToPoint(model.vertex_dict[remaining_vidx].position) < \
                                model.vertex_dict[iv2].position.distanceToPoint(model.vertex_dict[remaining_vidx].position) else vertex_axis_map[iv2]['y']
                        else:
                            ivy = vertex_axis_map[iv1]['y'] if abs(model.vertex_dict[iv1].position.y() - model.vertex_dict[remaining_vidx].position.y()) < \
                                abs(model.vertex_dict[iv2].position.y() - model.vertex_dict[remaining_vidx].position.y()) else vertex_axis_map[iv2]['y']
                        
                        iv1_map = (vertex_axis_map[iv1]['x'] + offset, ivy)
                        if iv1_map not in vertex_coordinate_map:
                            is_regist = False
                            for vidx in remaining_vidxs:
                                if vidx not in vertex_axis_map:
                                    is_regist = True
                                    vertex_axis_map[vidx] = {'vidx': vidx, 'x': iv1_map[0], 'y': iv1_map[1], 'position': model.vertex_dict[vidx].position}
                            if is_regist:
                                vertex_coordinate_map[iv1_map] = remaining_vidxs
                                logger.debug(f"fill_horizonal_now_idxs: key[{iv1_map}], v[{remaining_vidxs}]")
                            now_iidxs.append(index_idx)
                        
                        if len(now_iidxs) > 0:
                            break
                if len(now_iidxs) > 0:
                    break
        
        registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))
        
        for index_idx in now_iidxs:
            # 斜めが埋まってる場合、残りの一点を埋める
            vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
                self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
                                                       vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs)

        return registed_iidxs, now_iidxs
    
    def fill_diagonal_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
                                          vertex_axis_map: dict, vertex_coordinate_map: dict, registed_iidxs: list, now_iidxs: list):

        # 斜めが埋まっている場合、残りの一点を求める（四角形を求められる）
        for index_idx in now_iidxs:
            # 面の辺を抽出
            _, _, diagonal_vs = self.judge_index_edge(model, vertex_axis_map, index_idx)

            if diagonal_vs and diagonal_vs in duplicate_indices:
                for iidx in duplicate_indices[diagonal_vs]:
                    edge_size = len(set(model.indices[iidx]) & set(vertex_axis_map.keys()))
                    if edge_size >= 2:
                        if edge_size == 2:
                            # 重複頂点(2つの頂点)を持つ面(=連続面)
                            vertex_axis_map, vertex_coordinate_map = \
                                self.create_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
                                                                vertex_axis_map, vertex_coordinate_map, iidx)
                        
                        # 登録済みでなければ保持
                        if iidx not in now_iidxs:
                            now_iidxs.append(iidx)

        registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))

        return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs
    
    def fill_vertical_indices(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
                              vertex_axis_map: dict, vertex_coordinate_map: dict, indices_by_vpos: dict, indices_by_vidx: dict, \
                              registed_iidxs: list, vertical_iidxs: list, offset: int):
        vertical_vs_list = []

        for index_idx in vertical_iidxs:
            # 面の辺を抽出
            vertical_vs, _, _ = self.judge_index_edge(model, vertex_axis_map, index_idx)
            if not vertical_vs:
                continue

            if vertical_vs not in vertical_vs_list:
                vertical_vs_list.append(vertical_vs)

        now_iidxs = []

        if vertical_vs_list:
            # 縦が埋まっている場合、重複頂点から縦方向のベクトルが近いものを抽出する
            vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
                self.fill_vertical_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
                                                       vertex_axis_map, vertex_coordinate_map, indices_by_vpos, \
                                                       indices_by_vidx, vertical_vs_list, registed_iidxs, vertical_iidxs, offset)

        return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs

    def fill_vertical_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
                                          vertex_axis_map: dict, vertex_coordinate_map: dict, indices_by_vpos: dict, indices_by_vidx: dict, \
                                          vertical_vs_list: list, registed_iidxs: list, vertical_iidxs: list, offset: int):
        horizonaled_duplicate_indexs = []
        horizonaled_index_combs = []
        horizonaled_duplicate_dots = []
        horizonaled_vertical_above_v = []
        horizonaled_vertical_below_v = []
        not_horizonaled_duplicate_indexs = []
        not_horizonaled_index_combs = []
        not_horizonaled_duplicate_dots = []
        not_horizonaled_vertical_above_v = []
        not_horizonaled_vertical_below_v = []

        now_iidxs = []
        for vertical_vs in vertical_vs_list:
            # 該当縦辺の頂点(0が上(＋大きい))
            v0 = model.vertex_dict[vertical_vs[0]]
            v1 = model.vertex_dict[vertical_vs[1]]

            if offset > 0:
                # 下方向の走査
                vertical_vec = v0.position - v1.position
                vertical_above_v = v0
                vertical_below_v = v1
            else:
                # 上方向の走査
                vertical_vec = v1.position - v0.position
                vertical_above_v = v1
                vertical_below_v = v0

            if vertical_below_v.position.to_log() in indices_by_vpos:
                for duplicate_index_idx in indices_by_vpos[vertical_below_v.position.to_log()]:
                    if duplicate_index_idx in registed_iidxs + vertical_iidxs + now_iidxs:
                        # 既に登録済みの面である場合、スルー
                        continue

                    # 面の辺を抽出
                    vertical_in_vs, horizonal_in_vs, _ = self.judge_index_edge(model, vertex_axis_map, duplicate_index_idx)

                    if vertical_in_vs and horizonal_in_vs:
                        if ((offset > 0 and vertical_in_vs[0] in duplicate_vertices[vertical_below_v.position.to_log()]) \
                           or (offset < 0 and vertical_in_vs[1] in duplicate_vertices[vertical_below_v.position.to_log()])):
                            # 既に縦辺が求められていてそれに今回算出対象が含まれている場合
                            # 縦も横も求められているなら、該当面は必ず対象となる
                            horizonaled_duplicate_indexs.append(duplicate_index_idx)
                            horizonaled_vertical_below_v.append(vertical_below_v)
                            if offset > 0:
                                horizonaled_index_combs.append((vertical_in_vs[0], vertical_in_vs[1]))
                            else:
                                horizonaled_index_combs.append((vertical_in_vs[1], vertical_in_vs[0]))
                            horizonaled_duplicate_dots.append(1)
                        else:
                            # 既に縦辺が求められていてそれに今回算出対象が含まれていない場合、スルー
                            continue

                    # 重複辺（2点）の組み合わせ
                    index_combs = list(itertools.combinations(model.indices[duplicate_index_idx], 2))
                    for (iv0_comb_idx, iv1_comb_idx) in index_combs:
                        if horizonal_in_vs:
                            horizonaled_duplicate_indexs.append(duplicate_index_idx)
                            horizonaled_vertical_below_v.append(vertical_below_v)
                            horizonaled_vertical_above_v.append(vertical_above_v)

                            iv0 = None
                            iv1 = None

                            if iv0_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv1_comb_idx) not in horizonaled_index_combs:
                                iv0 = model.vertex_dict[iv0_comb_idx]
                                iv1 = model.vertex_dict[iv1_comb_idx]
                                horizonaled_index_combs.append((vertical_below_v.index, iv1_comb_idx))
                            elif iv1_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv0_comb_idx) not in horizonaled_index_combs:
                                iv0 = model.vertex_dict[iv1_comb_idx]
                                iv1 = model.vertex_dict[iv0_comb_idx]
                                horizonaled_index_combs.append((vertical_below_v.index, iv0_comb_idx))
                            else:
                                horizonaled_index_combs.append((-1, -1))

                            if iv0 and iv1:
                                if iv0.index in vertex_axis_map and (vertex_axis_map[iv0.index]['x'], vertex_axis_map[iv0.index]['y'] + offset) not in vertex_coordinate_map:
                                    # v1から繋がる辺のベクトル
                                    iv0 = model.vertex_dict[iv0.index]
                                    iv1 = model.vertex_dict[iv1.index]
                                    duplicate_vec = (iv0.position - iv1.position)
                                    horizonaled_duplicate_dots.append(MVector3D.dotProduct(vertical_vec.normalized(), duplicate_vec.normalized()))
                                else:
                                    horizonaled_duplicate_dots.append(0)
                            else:
                                horizonaled_duplicate_dots.append(0)
                        else:
                            not_horizonaled_duplicate_indexs.append(duplicate_index_idx)
                            not_horizonaled_vertical_below_v.append(vertical_below_v)
                            not_horizonaled_vertical_above_v.append(vertical_above_v)

                            iv0 = None
                            iv1 = None

                            if iv0_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv1_comb_idx) not in not_horizonaled_index_combs \
                                   and (vertical_below_v.index, iv1_comb_idx) not in horizonaled_index_combs:   # noqa
                                iv0 = model.vertex_dict[iv0_comb_idx]
                                iv1 = model.vertex_dict[iv1_comb_idx]
                                not_horizonaled_index_combs.append((vertical_below_v.index, iv1_comb_idx))
                            elif iv1_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv0_comb_idx) not in not_horizonaled_index_combs \
                                    and (vertical_below_v.index, iv0_comb_idx) not in horizonaled_index_combs:  # noqa
                                iv0 = model.vertex_dict[iv1_comb_idx]
                                iv1 = model.vertex_dict[iv0_comb_idx]
                                not_horizonaled_index_combs.append((vertical_below_v.index, iv0_comb_idx))
                            else:
                                not_horizonaled_index_combs.append((-1, -1))

                            if iv0 and iv1:
                                if iv0.index in vertex_axis_map and (vertex_axis_map[iv0.index]['x'], vertex_axis_map[iv0.index]['y'] + offset) not in vertex_coordinate_map:
                                    # v1から繋がる辺のベクトル
                                    iv0 = model.vertex_dict[iv0.index]
                                    iv1 = model.vertex_dict[iv1.index]
                                    duplicate_vec = (iv0.position - iv1.position)
                                    not_horizonaled_duplicate_dots.append(MVector3D.dotProduct(vertical_vec.normalized(), duplicate_vec.normalized()))
                                else:
                                    not_horizonaled_duplicate_dots.append(0)
                            else:
                                not_horizonaled_duplicate_dots.append(0)

        if len(horizonaled_duplicate_dots) > 0 and np.max(horizonaled_duplicate_dots) >= param_option['similarity']:
            logger.debug(f"fill_vertical: horizonaled_duplicate_dots[{horizonaled_duplicate_dots}], horizonaled_index_combs[{horizonaled_index_combs}]")

            full_d = [i for i, d in enumerate(horizonaled_duplicate_dots) if np.round(d, decimals=5) == np.max(np.round(horizonaled_duplicate_dots, decimals=5))]  # noqa
            not_full_d = [i for i, d in enumerate(not_horizonaled_duplicate_dots) if np.round(d, decimals=5) > np.max(np.round(horizonaled_duplicate_dots, decimals=5)) + 0.05]  # noqa
            if full_d:
                if not_full_d:
                    # 平行辺の内積より一定以上近い内積のINDEX組合せがあった場合、臨時採用
                    for vidx in not_full_d:
                        # 正方向に繋がる重複辺があり、かつそれが一定以上の場合、採用
                        vertical_vidxs = not_horizonaled_index_combs[vidx]
                        duplicate_index_idx = not_horizonaled_duplicate_indexs[vidx]
                        vertical_below_v = not_horizonaled_vertical_below_v[vidx]
                        vertical_above_v = not_horizonaled_vertical_above_v[vidx]

                        remaining_x = vertex_axis_map[vertical_below_v.index]['x']
                        remaining_y = vertex_axis_map[vertical_below_v.index]['y'] + offset
                        remaining_vidx = tuple(set(vertical_vidxs) - {vertical_below_v.index})[0]
                        remaining_v = model.vertex_dict[remaining_vidx]
                        # ほぼ同じベクトルを向いていたら、垂直頂点として登録
                        is_regist = False
                        for below_vidx in duplicate_vertices[remaining_v.position.to_log()]:
                            if below_vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
                                is_regist = True
                                vertex_axis_map[below_vidx] = {'vidx': below_vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[below_vidx].position}
                        if is_regist:
                            vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
                            logger.debug(f"fill_vertical1: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}]")

                        now_iidxs.append(duplicate_index_idx)
                else:
                    vidx = full_d[0]

                    # 正方向に繋がる重複辺があり、かつそれが一定以上の場合、採用
                    vertical_vidxs = horizonaled_index_combs[vidx]
                    duplicate_index_idx = horizonaled_duplicate_indexs[vidx]
                    vertical_below_v = horizonaled_vertical_below_v[vidx]

                    remaining_x = vertex_axis_map[vertical_below_v.index]['x']
                    remaining_y = vertex_axis_map[vertical_below_v.index]['y'] + offset
                    remaining_vidx = tuple(set(vertical_vidxs) - {vertical_below_v.index})[0]
                    remaining_v = model.vertex_dict[remaining_vidx]
                    # ほぼ同じベクトルを向いていたら、垂直頂点として登録
                    is_regist = False
                    for below_vidx in duplicate_vertices[remaining_v.position.to_log()]:
                        if below_vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
                            is_regist = True
                            vertex_axis_map[below_vidx] = {'vidx': below_vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[below_vidx].position}
                    if is_regist:
                        vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
                        logger.debug(f"fill_vertical1: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}]")

                    now_iidxs.append(duplicate_index_idx)

                    if vertical_vidxs[0] in vertex_axis_map and vertical_vidxs[1] in vertex_axis_map:
                        vertical_v0 = vertex_axis_map[vertical_vidxs[0]]
                        vertical_v1 = vertex_axis_map[vertical_vidxs[1]]
                        remaining_v = model.vertex_dict[tuple(set(model.indices[duplicate_index_idx]) - set(vertical_vidxs))[0]]

                        if remaining_v.index not in vertex_axis_map:
                            # 残り一点のマップ位置
                            remaining_x, remaining_y = self.get_remaining_vertex_vec(vertical_v0['vidx'], vertical_v0['x'], vertical_v0['y'], vertical_v0['position'], \
                                                                                     vertical_v1['vidx'], vertical_v1['x'], vertical_v1['y'], vertical_v1['position'], \
                                                                                     remaining_v, vertex_coordinate_map, duplicate_indices, duplicate_vertices)

                            is_regist = False
                            for vidx in duplicate_vertices[remaining_v.position.to_log()]:
                                if vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
                                    is_regist = True
                                    vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
                            if is_regist:
                                vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
                                logger.debug(f"fill_vertical2: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}]")

                        # 斜めが埋められそうなら埋める
                        vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
                            self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, vertex_axis_map, \
                                                                   vertex_coordinate_map, registed_iidxs, now_iidxs)
        
        registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))

        return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs

    def judge_index_edge(self, model: PmxModel, vertex_axis_map: dict, index_idx: int):
        # 該当面の頂点
        v0 = model.vertex_dict[model.indices[index_idx][0]]
        v1 = model.vertex_dict[model.indices[index_idx][1]]
        v2 = model.vertex_dict[model.indices[index_idx][2]]

        # 縦の辺を抽出
        vertical_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map and vertex_axis_map[v0.index]['x'] == vertex_axis_map[v1.index]['x'] \
            else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v0.index]['x'] == vertex_axis_map[v2.index]['x'] \
            else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v1.index]['x'] == vertex_axis_map[v2.index]['x'] else None
        if vertical_vs:
            vertical_vs = (vertical_vs[0], vertical_vs[1]) if vertex_axis_map[vertical_vs[0]]['y'] < vertex_axis_map[vertical_vs[1]]['y'] else (vertical_vs[1], vertical_vs[0])

        # 横の辺を抽出
        horizonal_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v1.index]['y'] \
            else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v2.index]['y'] \
            else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v1.index]['y'] == vertex_axis_map[v2.index]['y'] else None
        if horizonal_vs:
            horizonal_vs = (horizonal_vs[0], horizonal_vs[1]) if vertex_axis_map[horizonal_vs[0]]['x'] < vertex_axis_map[horizonal_vs[1]]['x'] else (horizonal_vs[1], horizonal_vs[0])

        # 斜めの辺を抽出
        diagonal_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map \
            and vertex_axis_map[v0.index]['x'] != vertex_axis_map[v1.index]['x'] and vertex_axis_map[v0.index]['y'] != vertex_axis_map[v1.index]['y'] \
            else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map \
            and vertex_axis_map[v0.index]['x'] != vertex_axis_map[v2.index]['x'] and vertex_axis_map[v0.index]['y'] != vertex_axis_map[v2.index]['y'] \
            else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map \
            and vertex_axis_map[v1.index]['x'] != vertex_axis_map[v2.index]['x'] and vertex_axis_map[v1.index]['y'] != vertex_axis_map[v2.index]['y'] else None
        if diagonal_vs:
            diagonal_vs = (min(diagonal_vs[0], diagonal_vs[1]), max(diagonal_vs[0], diagonal_vs[1]))

        return vertical_vs, horizonal_vs, diagonal_vs

    def create_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
                                   vertex_axis_map: dict, vertex_coordinate_map: dict, index_idx: int):
        # 該当面の頂点
        v0 = model.vertex_dict[model.indices[index_idx][0]]
        v1 = model.vertex_dict[model.indices[index_idx][1]]
        v2 = model.vertex_dict[model.indices[index_idx][2]]

        # 重複を含む頂点一覧
        vs_duplicated = {}
        vs_duplicated[v0.index] = duplicate_vertices[v0.position.to_log()]
        vs_duplicated[v1.index] = duplicate_vertices[v1.position.to_log()]
        vs_duplicated[v2.index] = duplicate_vertices[v2.position.to_log()]

        if not vertex_axis_map:
            # 空の場合、原点として0番目を設定する
            # 表向き=時計回りで当てはめていく
            for vidx in vs_duplicated[v0.index]:
                vertex_axis_map[vidx] = {'vidx': vidx, 'x': 0, 'y': 0, 'position': model.vertex_dict[vidx].position}
            vertex_coordinate_map[(0, 0)] = vs_duplicated[v0.index]

            for vidx in vs_duplicated[v1.index]:
                # v1 は位置関係で当てはめる
                if round(v0.position.y(), 2) == round(v1.position.y(), 2) == round(v2.position.y(), 2):
                    v1dot = MVector3D.dotProduct(MVector3D(1, 0, 0), (v1.position - v0.position).normalized())
                    v2dot = MVector3D.dotProduct(MVector3D(1, 0, 0), (v2.position - v0.position).normalized())

                    if abs(v1dot) > abs(v2dot):
                        # v1の方がv2よりX平行に近い場合、v1は横方向とみなす
                        vx = int(np.sign(MVector3D.crossProduct((v1.position - v0.position).normalized(), (v2.position - v0.position).normalized()).y()))
                        vy = 0
                    else:
                        # v2の方がv1よりX平行に近い場合、v1は縦方向とみなす
                        vx = 0
                        vy = int(np.sign(MVector3D.crossProduct((v1.position - v0.position).normalized(), (v2.position - v0.position).normalized()).y()))
                else:
                    # 方向に応じて判定値を変える
                    if param_option['direction'] == '上':
                        v0v = -v0.position.y()
                        v1v = -v1.position.y()
                        v2v = -v2.position.y()
                    elif param_option['direction'] == '右':
                        v0v = v0.position.x()
                        v1v = v1.position.x()
                        v2v = v2.position.x()
                    elif param_option['direction'] == '左':
                        v0v = -v0.position.x()
                        v1v = -v1.position.x()
                        v2v = -v2.position.x()
                    else:
                        # デフォルトは下
                        v0v = v0.position.y()
                        v1v = v1.position.y()
                        v2v = v2.position.y()

                    if abs(v1v - v0v) < abs(v2v - v0v):
                        # v1は横展開とみなす
                        if v0v < v2v:
                            # 上にあがる場合、v1は左方向
                            vx = -1
                        else:
                            # 下におりる場合、v1は右方向
                            vx = 1
                        vy = 0
                    else:
                        vy = -1 if v0v < v1v else 1
                        v1dot = MVector3D.dotProduct(MVector3D(0, -vy, 0), (v1.position - v0.position).normalized())
                        v2dot = MVector3D.dotProduct(MVector3D(0, -vy, 0), (v2.position - v0.position).normalized())
                        if v1dot > v2dot:
                            # v1の方がv0に近い場合、v1は縦展開とみなす
                            vx = 0
                        else:
                            # v2の方がv0に近い場合、v1は斜めと見なす
                            if abs(v1v - v0v) < abs(v2v - v0v):
                                # ＼の場合、v1は左方向
                                vx = -1
                            else:
                                # ／の場合、v1は右方向
                                vx = 1

                vertex_axis_map[vidx] = {'vidx': vidx, 'x': vx, 'y': vy, 'position': model.vertex_dict[vidx].position, 'duplicate': duplicate_vertices[model.vertex_dict[vidx].position.to_log()]}
            vertex_coordinate_map[(vx, vy)] = vs_duplicated[v1.index]

            # 残り一点のマップ位置
            remaining_x, remaining_y = self.get_remaining_vertex_vec(v0.index, vertex_axis_map[v0.index]['x'], vertex_axis_map[v0.index]['y'], vertex_axis_map[v0.index]['position'], \
                                                                     v1.index, vertex_axis_map[v1.index]['x'], vertex_axis_map[v1.index]['y'], vertex_axis_map[v1.index]['position'], \
                                                                     v2, vertex_coordinate_map, duplicate_indices, duplicate_vertices)

            for vidx in vs_duplicated[v2.index]:
                vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
            vertex_coordinate_map[(remaining_x, remaining_y)] = vs_duplicated[v2.index]

            logger.debug(f"初期iidx: iidx[{index_idx}], coodinate[{vertex_coordinate_map}]")
        else:
            # 残りの頂点INDEX
            remaining_v = None
            
            # 重複辺のマップ情報（時計回りで設定する）
            v_duplicated_maps = []
            if v0.index not in vertex_axis_map:
                remaining_v = v0
                v_duplicated_maps.append(vertex_axis_map[v1.index])
                v_duplicated_maps.append(vertex_axis_map[v2.index])

            if v1.index not in vertex_axis_map:
                remaining_v = v1
                v_duplicated_maps.append(vertex_axis_map[v2.index])
                v_duplicated_maps.append(vertex_axis_map[v0.index])

            if v2.index not in vertex_axis_map:
                remaining_v = v2
                v_duplicated_maps.append(vertex_axis_map[v0.index])
                v_duplicated_maps.append(vertex_axis_map[v1.index])
            
            # 残り一点のマップ位置
            remaining_x, remaining_y = self.get_remaining_vertex_vec(v_duplicated_maps[0]['vidx'], v_duplicated_maps[0]['x'], v_duplicated_maps[0]['y'], v_duplicated_maps[0]['position'], \
                                                                     v_duplicated_maps[1]['vidx'], v_duplicated_maps[1]['x'], v_duplicated_maps[1]['y'], v_duplicated_maps[1]['position'], \
                                                                     remaining_v, vertex_coordinate_map, duplicate_indices, duplicate_vertices)

            is_regist = False
            for vidx in vs_duplicated[remaining_v.index]:
                if vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
                    is_regist = True
                    vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
            if is_regist:
                vertex_coordinate_map[(remaining_x, remaining_y)] = vs_duplicated[remaining_v.index]

        return vertex_axis_map, vertex_coordinate_map

    def get_remaining_vertex_vec(self, vv0_idx: int, vv0_x: int, vv0_y: int, vv0_vec: MVector3D, \
                                 vv1_idx: int, vv1_x: int, vv1_y: int, vv1_vec: MVector3D, remaining_v: Vertex, vertex_coordinate_map: dict, \
                                 duplicate_indices: dict, duplicate_vertices: dict):
        # 時計回りと見なして位置を合わせる
        if vv0_x == vv1_x:
            # 元が縦方向に一致している場合
            if vv0_y > vv1_y:
                remaining_x = vv0_x + 1
            else:
                remaining_x = vv0_x - 1

            if (remaining_x, vv0_y) in vertex_coordinate_map:
                remaining_y = vv1_y
                logger.debug(f"get_remaining_vertex_vec(縦): {remaining_x}, {remaining_y}")
            elif (remaining_x, vv1_y) in vertex_coordinate_map:
                remaining_y = vv0_y
                logger.debug(f"get_remaining_vertex_vec(縦): {remaining_x}, {remaining_y}")
            else:
                remaining_y = vv1_y if vv1_vec.distanceToPoint(remaining_v.position) < vv0_vec.distanceToPoint(remaining_v.position) else vv0_y
                logger.debug(f"get_remaining_vertex_vec(縦計算): {remaining_x}, {remaining_y}")

        elif vv0_y == vv1_y:
            # 元が横方向に一致している場合

            if vv0_x < vv1_x:
                remaining_y = vv0_y + 1
            else:
                remaining_y = vv0_y - 1
            
            if (vv0_x, remaining_y) in vertex_coordinate_map:
                remaining_x = vv1_x
                logger.debug(f"get_remaining_vertex_vec(横): {remaining_x}, {remaining_y}")
            elif (vv1_x, remaining_y) in vertex_coordinate_map:
                remaining_x = vv0_x
                logger.debug(f"get_remaining_vertex_vec(横): {remaining_x}, {remaining_y}")
            else:
                remaining_x = vv1_x if vv1_vec.distanceToPoint(remaining_v.position) < vv0_vec.distanceToPoint(remaining_v.position) else vv0_x
                logger.debug(f"get_remaining_vertex_vec(横計算): {remaining_x}, {remaining_y}")
        else:
            # 斜めが一致している場合
            if (vv0_x > vv1_x and vv0_y < vv1_y) or (vv0_x < vv1_x and vv0_y > vv1_y):
                # ／↑の場合、↓、↓／の場合、↑、／←の場合、→
                remaining_x = vv1_x
                remaining_y = vv0_y
                logger.debug(f"get_remaining_vertex_vec(斜1): {remaining_x}, {remaining_y}")
            else:
                # ＼←の場合、→、／→の場合、←
                remaining_x = vv0_x
                remaining_y = vv1_y
                logger.debug(f"get_remaining_vertex_vec(斜2): {remaining_x}, {remaining_y}")
        
        return remaining_x, remaining_y


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
