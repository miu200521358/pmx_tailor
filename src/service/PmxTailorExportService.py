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
import csv
import random
import string
from collections import Counter

from module.MOptions import MExportOptions
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, Sdef, RigidBodyParam, IkLink, Ik, BoneMorphData # noqa
from mmd.PmxWriter import PmxWriter
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException
import utils.MBezierUtils as MBezierUtils

logger = MLogger(__name__, level=1)


class VirtualVertex:

    def __init__(self, key):
        self.key = key
        self.real_vertices = []
        self.positions = []
        self.normals = []
        self.indexs = []
        self.line_vertices = []

    def append(self, v: Vertex, line_v1: Vertex, line_v2: Vertex, index_idx: int):
        if v not in self.real_vertices:
            self.real_vertices.append(v)
            self.positions.append(v.position.data())
            self.normals.append(v.normal.data())
        
        if index_idx not in self.indexs:
            self.indexs.append(index_idx)
        
        if line_v1 not in self.line_vertices:
            self.line_vertices.append(line_v1)

        if line_v2 not in self.line_vertices:
            self.line_vertices.append(line_v2)
    
    def vidxs(self):
        return [v.index for v in self.real_vertices]
    
    def position(self):
        return MVector3D(np.mean(self.positions, axis=0))
    
    def normal(self):
        return MVector3D(np.mean(self.normals, axis=0))
    
    def lines(self):
        lines = []
        for v in self.line_vertices:
            key = v.position.to_key()
            if key not in lines:
                lines.append(key)
        return lines
        
    def __str__(self):
        return f"v[{','.join([str(v.index) for v in self.real_vertices])}] pos[{self.position().to_log()}] nor[{self.normal().to_log()}], lines[{self.lines()}]"


class PmxTailorExportService():
    def __init__(self, options: MExportOptions):
        self.options = options

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = f"{logger.transtext('PmxTailor変換処理実行')}\n------------------------\n{logger.transtext('exeバージョン')}: {self.options.version_name}\n"
            service_data_txt = f"{service_data_txt}　{logger.transtext('元モデル')}: {os.path.basename(self.options.pmx_model.path)}\n"

            for pidx, param_option in enumerate(self.options.param_options):
                service_data_txt = f"{service_data_txt}\n　【No.{pidx + 1}】 --------- "    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('材質')}: {param_option['material_name']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('検出度')}: {param_option['similarity']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('細かさ')}: {param_option['fineness']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('質量')}: {param_option['mass']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('柔らかさ')}: {param_option['air_resistance']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　{logger.transtext('張り')}: {param_option['shape_maintenance']}"    # noqa

            logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

            model = self.options.pmx_model
            model.comment += f"\r\n\r\n{logger.transtext('物理')}: PmxTailor({self.options.version_name})"

            # 保持ボーンは全設定を確認する
            saved_bone_names = self.get_saved_bone_names(model)

            for pidx, param_option in enumerate(self.options.param_options):
                if not self.create_physics(model, param_option, saved_bone_names):
                    return False

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            PmxWriter().write(model, self.options.output_path)

            logger.info("出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title=logger.transtext("成功"))

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("PmxTailor変換処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical("PmxTailor変換処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
        finally:
            logging.shutdown()

    def create_physics(self, model: PmxModel, param_option: dict, saved_bone_names: list):
        model.comment += f"\r\n{logger.transtext('材質')}: {param_option['material_name']} --------------"    # noqa
        model.comment += f"\r\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"    # noqa
        model.comment += f", {logger.transtext('細かさ')}: {param_option['fineness']}"    # noqa
        model.comment += f", {logger.transtext('質量')}: {param_option['mass']}"    # noqa
        model.comment += f", {logger.transtext('柔らかさ')}: {param_option['air_resistance']}"    # noqa
        model.comment += f", {logger.transtext('張り')}: {param_option['shape_maintenance']}"    # noqa

        # 頂点CSVが指定されている場合、対象頂点リスト生成
        if param_option['vertices_csv']:
            target_vertices = []
            try:
                with open(param_option['vertices_csv'], encoding='cp932', mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)            # ヘッダーを読み飛ばす
                    for row in reader:
                        if len(row) > 1 and int(row[1]) in model.material_vertices[param_option['material_name']]:
                            target_vertices.append(int(row[1]))
            except Exception:
                logger.warning("頂点CSVが正常に読み込めなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
                return False
        else:
            target_vertices = list(model.material_vertices[param_option['material_name']])

        if param_option['exist_physics_clear'] in [logger.transtext('上書き'), logger.transtext('再利用')]:
            # 既存材質削除フラグONの場合
            logger.info("【%s】既存材質削除", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

            model = self.clear_exist_physics(model, param_option, param_option['material_name'], target_vertices, saved_bone_names)

            if not model:
                return False

        if param_option['exist_physics_clear'] == logger.transtext('再利用'):
            if param_option['physics_type'] in [logger.transtext('髪'), logger.transtext('単一揺'), logger.transtext('胸')]:
                logger.info("【%s】ボーンマップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                logger.info("【%s】剛体生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                root_rigidbody, registed_rigidbodies = self.create_vertical_rigidbody(model, param_option)

                logger.info("【%s】ジョイント生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                self.create_vertical_joint(model, param_option, root_rigidbody, registed_rigidbodies)
            else:
                logger.info("【%s】ボーンマップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                bone_blocks = self.create_bone_blocks(model, param_option, param_option['material_name'])

                if not bone_blocks:
                    logger.warning("有効なボーンマップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
                    return False

                logger.info("【%s】剛体生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                root_rigidbody, registed_rigidbodies = self.create_rigidbody_by_bone_blocks(model, param_option, bone_blocks)

                logger.info("【%s】ジョイント生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

                self.create_joint_by_bone_blocks(model, param_option, bone_blocks, root_rigidbody, registed_rigidbodies)

        else:
            logger.info("【%s】頂点マップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

            vertex_maps, vertex_connecteds, virtual_vertices \
                = self.create_vertex_map(model, param_option, param_option['material_name'], target_vertices)
            
            if not vertex_maps:
                return False
            
            # 各頂点の有効INDEX数が最も多いものをベースとする
            map_cnt = []
            for vertex_map in vertex_maps:
                map_cnt.append(np.count_nonzero(~np.isinf(vertex_map)) / 3)
            
            if len(map_cnt) == 0:
                logger.warning("有効な頂点マップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
                return False
            
            vertex_map_orders = [k for k in np.argsort(-np.array(map_cnt)) if map_cnt[k] > np.max(map_cnt) * 0.5]
            
            logger.info("【%s】ボーン生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

            root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances \
                = self.create_bone(model, param_option, vertex_map_orders, vertex_maps, vertex_connecteds, virtual_vertices)

            # vertex_remaining_set = set(target_vertices)

            # for base_map_idx in vertex_map_orders:
            #     logger.info("【%s(No.%s)】ウェイト分布", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

            #     self.create_weight(model, param_option, vertex_maps[base_map_idx], vertex_connecteds[base_map_idx], duplicate_vertices, \
            #                        all_registed_bone_indexs[base_map_idx], all_bone_horizonal_distances[base_map_idx], all_bone_vertical_distances[base_map_idx], \
            #                        vertex_remaining_set, target_vertices)

            # if len(list(vertex_remaining_set)) > 0:
            #     logger.info("【%s】残ウェイト分布", param_option['material_name'], decoration=MLogger.DECORATION_LINE)
                
            #     self.create_remaining_weight(model, param_option, vertex_maps, vertex_remaining_set, vertex_map_orders, target_vertices)
    
            # if param_option['edge_material_name']:
            #     logger.info("【%s】裾ウェイト分布", param_option['edge_material_name'], decoration=MLogger.DECORATION_LINE)

            #     edge_vertices = set(model.material_vertices[param_option['edge_material_name']])
            #     self.create_remaining_weight(model, param_option, vertex_maps, edge_vertices, vertex_map_orders, edge_vertices)
        
            # if param_option['back_material_name']:
            #     logger.info("【%s】裏面ウェイト分布", param_option['back_material_name'], decoration=MLogger.DECORATION_LINE)

            #     self.create_back_weight(model, param_option)
    
            # for base_map_idx in vertex_map_orders:
            #     logger.info("【%s(No.%s)】剛体生成", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

            #     root_rigidbody, registed_rigidbodies = self.create_rigidbody(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx], root_bone)

            #     logger.info("【%s(No.%s)】ジョイント生成", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

            #     self.create_joint(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx], root_rigidbody, registed_rigidbodies)

        return True

    def create_bone_blocks(self, model: PmxModel, param_option: dict, material_name: str):
        bone_grid = param_option["bone_grid"]
        bone_grid_cols = param_option["bone_grid_cols"]
        bone_grid_rows = param_option["bone_grid_rows"]

        # ウェイトボーンリスト取得（ついでにウェイト正規化）
        weighted_bone_pairs = []
        for vertex_idx in model.material_vertices[material_name]:
            vertex = model.vertex_dict[vertex_idx]
            if type(vertex.deform) is Bdef2 or type(vertex.deform) is Sdef:
                if 0 < vertex.deform.weight0 < 1:
                    # 2つめのボーンも有効値を持っている場合、判定対象
                    key = (min(vertex.deform.index0, vertex.deform.index1), max(vertex.deform.index0, vertex.deform.index1))
                    if key not in weighted_bone_pairs:
                        weighted_bone_pairs.append(key)
            elif type(vertex.deform) is Bdef4:
                # ウェイト正規化
                total_weights = np.array([vertex.deform.weight0, vertex.deform.weight1, vertex.deform.weight2, vertex.deform.weight3])
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)

                vertex.deform.weight0 = weights[0]
                vertex.deform.weight1 = weights[1]
                vertex.deform.weight2 = weights[2]
                vertex.deform.weight3 = weights[3]

                weighted_bone_indexes = []
                if vertex.deform.weight0 > 0:
                    weighted_bone_indexes.append(vertex.deform.index0)
                if vertex.deform.weight1 > 0:
                    weighted_bone_indexes.append(vertex.deform.index1)
                if vertex.deform.weight2 > 0:
                    weighted_bone_indexes.append(vertex.deform.index2)
                if vertex.deform.weight3 > 0:
                    weighted_bone_indexes.append(vertex.deform.index3)

                for bi0, bi1 in list(itertools.combinations(weighted_bone_indexes, 2)):
                    # ボーン2つずつのペアでウェイト繋がり具合を保持する
                    key = (min(bi0, bi1), max(bi0, bi1))
                    if key not in weighted_bone_pairs:
                        weighted_bone_pairs.append(key)
        
        bone_blocks = {}
        for pac in range(bone_grid_cols):
            prev_above_bone_name = None
            prev_above_bone_position = None
            for par in range(bone_grid_rows):
                prev_above_bone_name = bone_grid[par][pac]
                if not prev_above_bone_name:
                    continue
                
                is_above_connected = True
                prev_above_bone_position = model.bones[prev_above_bone_name].position
                prev_above_bone_index = model.bones[prev_above_bone_name].index
                if prev_above_bone_name:
                    prev_below_bone_name = None
                    prev_below_bone_position = None
                    pbr = par + 1
                    if pbr in bone_grid and pac in bone_grid[pbr]:
                        prev_below_bone_name = bone_grid[pbr][pac]
                        if prev_below_bone_name:
                            prev_below_bone_position = model.bones[prev_below_bone_name].position
                        if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                            # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
                            prev_below_bone_name = prev_above_bone_name
                            prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
                    elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                        prev_below_bone_name = prev_above_bone_name
                        prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

                    next_above_bone_name = None
                    next_above_bone_position = None
                    nnac = [k for k, v in bone_grid[par].items() if v][-1]
                    if pac < nnac:
                        # 右周りにボーンの連携をチェック
                        nac = pac + 1
                        if par in bone_grid and nac in bone_grid[par]:
                            next_above_bone_name = bone_grid[par][nac]
                            if next_above_bone_name:
                                next_above_bone_position = model.bones[next_above_bone_name].position
                                next_above_bone_index = model.bones[next_above_bone_name].index
                            else:
                                # 隣がない場合、1つ前のボーンと結合させる
                                nac = pac - 1
                                next_above_bone_name = bone_grid[par][nac]
                                if next_above_bone_name:
                                    next_above_bone_position = model.bones[next_above_bone_name].position
                                    next_above_bone_index = model.bones[next_above_bone_name].index
                                    is_above_connected = False
                    else:
                        # 一旦円周を描いてみる
                        next_above_bone_name = bone_grid[par][0]
                        nac = 0
                        if next_above_bone_name:
                            next_above_bone_position = model.bones[next_above_bone_name].position
                            next_above_bone_index = model.bones[next_above_bone_name].index
                            key = (min(prev_above_bone_index, next_above_bone_index), max(prev_above_bone_index, next_above_bone_index))
                            if key not in weighted_bone_pairs:
                                # ウェイトが乗ってなかった場合、2つ前のボーンと結合させる
                                nac = pac - 1
                                if par in bone_grid and nac in bone_grid[par]:
                                    next_above_bone_name = bone_grid[par][nac]
                                    if next_above_bone_name:
                                        next_above_bone_position = model.bones[next_above_bone_name].position
                                        next_above_bone_index = model.bones[next_above_bone_name].index
                                        is_above_connected = False

                    next_below_bone_name = None
                    next_below_bone_position = None
                    nbr = par + 1
                    if nbr in bone_grid and nac in bone_grid[nbr]:
                        next_below_bone_name = bone_grid[nbr][nac]
                        if next_below_bone_name:
                            next_below_bone_position = model.bones[next_below_bone_name].position
                        if next_above_bone_name and not next_below_bone_name and model.bones[next_above_bone_name].tail_position != MVector3D():
                            # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
                            next_below_bone_name = next_above_bone_name
                            next_below_bone_position = next_above_bone_position + model.bones[next_above_bone_name].tail_position
                    elif next_above_bone_name and model.bones[next_above_bone_name].tail_position != MVector3D():
                        next_below_bone_name = next_above_bone_name
                        next_below_bone_position = next_above_bone_position + model.bones[next_above_bone_name].tail_position

                    prev_prev_above_bone_name = None
                    prev_prev_above_bone_position = None
                    if pac > 0:
                        # 左周りにボーンの連携をチェック
                        ppac = pac - 1
                        if par in bone_grid and ppac in bone_grid[par]:
                            prev_prev_above_bone_name = bone_grid[par][ppac]
                            if prev_prev_above_bone_name:
                                prev_prev_above_bone_position = model.bones[prev_prev_above_bone_name].position
                            else:
                                # 隣がない場合、prev_aboveと同じにする
                                prev_prev_above_bone_name = prev_above_bone_name
                                prev_prev_above_bone_position = prev_above_bone_position
                        else:
                            prev_prev_above_bone_name = prev_above_bone_name
                            prev_prev_above_bone_position = prev_above_bone_position
                    else:
                        # 一旦円周を描いてみる
                        ppac = [k for k, v in bone_grid[par].items() if v][-1]
                        prev_prev_above_bone_name = bone_grid[par][ppac]
                        if prev_prev_above_bone_name:
                            prev_prev_above_bone_position = model.bones[prev_prev_above_bone_name].position
                            prev_prev_above_bone_index = model.bones[prev_prev_above_bone_name].index
                            key = (min(prev_above_bone_index, prev_prev_above_bone_index), max(prev_above_bone_index, prev_prev_above_bone_index))
                            if key not in weighted_bone_pairs:
                                # ウェイトが乗ってなかった場合、prev_aboveと同じにする
                                prev_prev_above_bone_name = prev_above_bone_name
                                prev_prev_above_bone_position = prev_above_bone_position
                        else:
                            prev_prev_above_bone_name = prev_above_bone_name
                            prev_prev_above_bone_position = prev_above_bone_position

                    if prev_above_bone_name and prev_below_bone_name and next_above_bone_name and next_below_bone_name:
                        bone_blocks[prev_above_bone_name] = {'prev_above': prev_above_bone_name, 'prev_below': prev_below_bone_name, \
                                                             'next_above': next_above_bone_name, 'next_below': next_below_bone_name, \
                                                             'prev_above_pos': prev_above_bone_position, 'prev_below_pos': prev_below_bone_position, \
                                                             'next_above_pos': next_above_bone_position, 'next_below_pos': next_below_bone_position, \
                                                             'prev_prev_above': prev_prev_above_bone_name, 'prev_prev_above_pos': prev_prev_above_bone_position, \
                                                             'yi': par, 'xi': pac, 'is_above_connected': is_above_connected}
                        logger.debug(f'prev_above: {prev_above_bone_name}, [{prev_above_bone_position.to_log()}], ' \
                                     + f'next_above: {next_above_bone_name}, [{next_above_bone_position.to_log()}], ' \
                                     + f'prev_below: {prev_below_bone_name}, [{prev_below_bone_position.to_log()}], ' \
                                     + f'next_below: {next_below_bone_name}, [{next_below_bone_position.to_log()}], ' \
                                     + f'prev_prev_above: {prev_prev_above_bone_name}, [{prev_prev_above_bone_position.to_log()}], ' \
                                     + f'yi: {par}, xi: {pac}, is_above_connected: {is_above_connected}')

        return bone_blocks
    
    def create_vertical_joint(self, model: PmxModel, param_option: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
        bone_grid_cols = param_option["bone_grid_cols"]
        bone_grid_rows = param_option["bone_grid_rows"]
        bone_grid = param_option["bone_grid"]

        # ジョイント生成
        created_joints = {}

        # 縦ジョイント情報
        param_vertical_joint = param_option['vertical_joint']

        prev_joint_cnt = 0

        for pac in range(bone_grid_cols):
            # ジョイント生成
            created_joints = {}

            valid_rows = [par for par in range(bone_grid_rows) if par]
            if len(valid_rows) == 0:
                continue
            
            max_vy = valid_rows[-1]
            min_vy = 0
            xs = np.arange(min_vy, max_vy, step=1)
        
            if param_vertical_joint:
                coefficient = param_option['vertical_joint_coefficient']

                vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
                vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
                vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

                vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
                vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
                vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

                vertical_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x()]])), xs)             # noqa
                vertical_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y()]])), xs)             # noqa
                vertical_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z()]])), xs)             # noqa

                vertical_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x()]])), xs)             # noqa
                vertical_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y()]])), xs)             # noqa
                vertical_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z()]])), xs)             # noqa

                vertical_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x()]])), xs)             # noqa
                vertical_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y()]])), xs)             # noqa
                vertical_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z()]])), xs)             # noqa

                vertical_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x()]])), xs)             # noqa
                vertical_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y()]])), xs)             # noqa
                vertical_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z()]])), xs)             # noqa
            
            prev_above_bone_name = None
            prev_above_bone_position = None
            for par in range(bone_grid_rows):
                prev_above_bone_name = bone_grid[par][pac]
                if not prev_above_bone_name or prev_above_bone_name not in model.bones:
                    continue

                prev_above_bone_position = model.bones[prev_above_bone_name].position
                prev_below_bone_name = None
                prev_below_bone_position = None
                prev_below_below_bone_name = None
                prev_below_below_bone_position = None
                if prev_above_bone_name:
                    pbr = par + 1
                    if pbr in bone_grid and pac in bone_grid[pbr]:
                        prev_below_bone_name = bone_grid[pbr][pac]
                        if prev_below_bone_name:
                            prev_below_bone_position = model.bones[prev_below_bone_name].position

                            pbbr = pbr + 1
                            if pbbr in bone_grid and pac in bone_grid[pbbr]:
                                prev_below_below_bone_name = bone_grid[pbbr][pac]
                                if prev_below_below_bone_name:
                                    prev_below_below_bone_position = model.bones[prev_below_below_bone_name].position
                                if not prev_below_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                                    # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
                                    prev_below_below_bone_name = prev_above_bone_name
                                    prev_below_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
                            elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                                prev_below_below_bone_name = prev_above_bone_name
                                prev_below_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

                        if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                            # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
                            prev_below_bone_name = prev_above_bone_name
                            prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
                    elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                        prev_below_bone_name = prev_above_bone_name
                        prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

                if prev_above_bone_position and prev_below_bone_position:
                    if not prev_below_below_bone_position:
                        prev_below_below_bone_position = prev_below_bone_position

                    if par == 0 and prev_above_bone_name in registed_rigidbodies:
                        # ルート剛体と根元剛体を繋ぐジョイント
                        joint_name = f'↓|{root_rigidbody.name}|{registed_rigidbodies[prev_above_bone_name]}'

                        # 縦ジョイント
                        joint_vec = prev_above_bone_position

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (root_rigidbody.shape_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        joint = Joint(joint_name, joint_name, 0, root_rigidbody.index, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[par], vertical_limit_min_mov_ys[par], vertical_limit_min_mov_zs[par]), \
                                      MVector3D(vertical_limit_max_mov_xs[par], vertical_limit_max_mov_ys[par], vertical_limit_max_mov_zs[par]),
                                      MVector3D(math.radians(vertical_limit_min_rot_xs[par]), math.radians(vertical_limit_min_rot_ys[par]), math.radians(vertical_limit_min_rot_zs[par])),
                                      MVector3D(math.radians(vertical_limit_max_rot_xs[par]), math.radians(vertical_limit_max_rot_ys[par]), math.radians(vertical_limit_max_rot_zs[par])),
                                      MVector3D(vertical_spring_constant_mov_xs[par], vertical_spring_constant_mov_ys[par], vertical_spring_constant_mov_zs[par]), \
                                      MVector3D(vertical_spring_constant_rot_xs[par], vertical_spring_constant_rot_ys[par], vertical_spring_constant_rot_zs[par]))
                        created_joints[f'0:{root_rigidbody.index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'] = joint

                        # バランサー剛体が必要な場合
                        if param_option["rigidbody_balancer"]:
                            balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
                            joint_name = f'B|{prev_above_bone_name}|{balancer_prev_above_bone_name}'
                            joint_key = f'8:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}'

                            joint_vec = model.rigidbodies[prev_above_bone_name].shape_position

                            # 回転量
                            joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis = (model.rigidbodies[balancer_prev_above_bone_name].shape_position - prev_above_bone_position).normalized()
                            joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                            joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                            joint_euler = joint_rotation_qq.toEulerAngles()
                            joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[balancer_prev_above_bone_name].index,
                                          joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                          MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
                            created_joints[joint_key] = joint
            
                    if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
                        # 縦ジョイント
                        joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{prev_below_bone_name}'
                        joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'   # noqa

                        if joint_key not in created_joints:
                            # 未登録のみ追加
                            
                            # 縦ジョイント
                            joint_vec = prev_below_bone_position

                            # 回転量
                            joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis = (prev_below_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                            joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                            joint_euler = joint_rotation_qq.toEulerAngles()
                            joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                          model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
                                          joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[par], vertical_limit_min_mov_ys[par], vertical_limit_min_mov_zs[par]), \
                                          MVector3D(vertical_limit_max_mov_xs[par], vertical_limit_max_mov_ys[par], vertical_limit_max_mov_zs[par]),
                                          MVector3D(math.radians(vertical_limit_min_rot_xs[par]), math.radians(vertical_limit_min_rot_ys[par]), math.radians(vertical_limit_min_rot_zs[par])),
                                          MVector3D(math.radians(vertical_limit_max_rot_xs[par]), math.radians(vertical_limit_max_rot_ys[par]), math.radians(vertical_limit_max_rot_zs[par])),
                                          MVector3D(vertical_spring_constant_mov_xs[par], vertical_spring_constant_mov_ys[par], vertical_spring_constant_mov_zs[par]), \
                                          MVector3D(vertical_spring_constant_rot_xs[par], vertical_spring_constant_rot_ys[par], vertical_spring_constant_rot_zs[par]))
                            created_joints[joint_key] = joint

                            # バランサー剛体が必要な場合
                            if param_option["rigidbody_balancer"]:
                                balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
                                joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
                                joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

                                joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

                                # 回転量
                                joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                                joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
                                joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                                joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                                joint_euler = joint_rotation_qq.toEulerAngles()
                                joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
                                              model.rigidbodies[balancer_prev_below_bone_name].index,
                                              joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                              MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
                                created_joints[joint_key] = joint

                                # バランサー補助剛体
                                balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
                                joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
                                joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'  # noqa
                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
                                              model.rigidbodies[balancer_prev_below_bone_name].index,
                                              MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
                                              MVector3D(),MVector3D(), MVector3D())   # noqa
                                created_joints[joint_key] = joint
                                
            for joint_key in sorted(created_joints.keys()):
                # ジョイントを登録
                joint = created_joints[joint_key]
                joint.index = len(model.joints)

                if joint.name in model.joints:
                    logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
                    joint.name += randomname(3)

                model.joints[joint.name] = joint

            prev_joint_cnt += len(created_joints)

        logger.info("-- ジョイント: %s個目:終了", prev_joint_cnt)
                            
        return root_rigidbody
    
    def create_joint_by_bone_blocks(self, model: PmxModel, param_option: dict, bone_blocks: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
        bone_grid_rows = param_option["bone_grid_rows"]
        # bone_grid_cols = param_option["bone_grid_cols"]

        # ジョイント生成
        created_joints = {}

        # # 略称
        # abb_name = param_option['abb_name']
        # 縦ジョイント情報
        param_vertical_joint = param_option['vertical_joint']
        # 横ジョイント情報
        param_horizonal_joint = param_option['horizonal_joint']
        # 斜めジョイント情報
        param_diagonal_joint = param_option['diagonal_joint']
        # 逆ジョイント情報
        param_reverse_joint = param_option['reverse_joint']

        prev_joint_cnt = 0

        max_vy = bone_grid_rows
        middle_vy = (bone_grid_rows) * 0.3
        min_vy = 0
        xs = np.arange(min_vy, max_vy, step=1)
    
        if param_vertical_joint:
            coefficient = param_option['vertical_joint_coefficient']

            vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
            vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
            vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

            vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
            vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
            vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

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
            
            horizonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x()]])), xs)             # noqa
            horizonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y()]])), xs)             # noqa
            horizonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z()]])), xs)             # noqa

            horizonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x()]])), xs)             # noqa
            horizonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y()]])), xs)             # noqa
            horizonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z()]])), xs)             # noqa

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

            diagonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x()]])), xs)             # noqa
            diagonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y()]])), xs)             # noqa
            diagonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z()]])), xs)             # noqa

            diagonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x()]])), xs)             # noqa
            diagonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y()]])), xs)             # noqa
            diagonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z()]])), xs)             # noqa

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

            reverse_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x()]])), xs)             # noqa
            reverse_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y()]])), xs)             # noqa
            reverse_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z()]])), xs)             # noqa

            reverse_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x()]])), xs)             # noqa
            reverse_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y()]])), xs)             # noqa
            reverse_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z()]])), xs)             # noqa

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
        
        for bone_block in bone_blocks.values():
            prev_above_bone_name = bone_block['prev_above']
            prev_above_bone_position = bone_block['prev_above_pos']
            prev_below_bone_name = bone_block['prev_below']
            prev_below_bone_position = bone_block['prev_below_pos']
            next_above_bone_name = bone_block['next_above']
            next_above_bone_position = bone_block['next_above_pos']
            next_below_bone_name = bone_block['next_below']
            next_below_bone_position = bone_block['next_below_pos']
            yi = bone_block['yi']
            # xi = bone_block['xi']

            if yi == 0 and prev_above_bone_name in registed_rigidbodies:
                # ルート剛体と根元剛体を繋ぐジョイント
                joint_name = f'↓|{root_rigidbody.name}|{registed_rigidbodies[prev_above_bone_name]}'

                # 縦ジョイント
                joint_vec = prev_above_bone_position

                # 回転量
                joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                joint_axis = (root_rigidbody.shape_position - prev_above_bone_position).normalized()
                joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                joint_euler = joint_rotation_qq.toEulerAngles()
                joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                joint = Joint(joint_name, joint_name, 0, root_rigidbody.index, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                              joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yi], vertical_limit_min_mov_ys[yi], vertical_limit_min_mov_zs[yi]), \
                              MVector3D(vertical_limit_max_mov_xs[yi], vertical_limit_max_mov_ys[yi], vertical_limit_max_mov_zs[yi]),
                              MVector3D(math.radians(vertical_limit_min_rot_xs[yi]), math.radians(vertical_limit_min_rot_ys[yi]), math.radians(vertical_limit_min_rot_zs[yi])),
                              MVector3D(math.radians(vertical_limit_max_rot_xs[yi]), math.radians(vertical_limit_max_rot_ys[yi]), math.radians(vertical_limit_max_rot_zs[yi])),
                              MVector3D(vertical_spring_constant_mov_xs[yi], vertical_spring_constant_mov_ys[yi], vertical_spring_constant_mov_zs[yi]), \
                              MVector3D(vertical_spring_constant_rot_xs[yi], vertical_spring_constant_rot_ys[yi], vertical_spring_constant_rot_zs[yi]))   # noqa
                created_joints[f'0:{root_rigidbody.index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'] = joint

                # バランサー剛体が必要な場合
                if param_option["rigidbody_balancer"]:
                    balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
                    joint_name = f'B|{prev_above_bone_name}|{balancer_prev_above_bone_name}'
                    joint_key = f'8:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}'

                    joint_vec = model.rigidbodies[prev_above_bone_name].shape_position

                    # 回転量
                    joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                    joint_axis = (model.rigidbodies[balancer_prev_above_bone_name].shape_position - prev_above_bone_position).normalized()
                    joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                    joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                    joint_euler = joint_rotation_qq.toEulerAngles()
                    joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                    joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[balancer_prev_above_bone_name].index,
                                  joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                  MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
                    created_joints[joint_key] = joint
    
            if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
                # 縦ジョイント
                joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[prev_below_bone_name]}'
                joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

                if joint_key not in created_joints:
                    # 未登録のみ追加
                    
                    # 縦ジョイント
                    joint_vec = prev_below_bone_position

                    # 回転量
                    joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                    joint_axis = (next_above_bone_position - prev_above_bone_position).normalized()
                    joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                    joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                    joint_euler = joint_rotation_qq.toEulerAngles()
                    joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                    joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                  model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
                                  joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yi], vertical_limit_min_mov_ys[yi], vertical_limit_min_mov_zs[yi]), \
                                  MVector3D(vertical_limit_max_mov_xs[yi], vertical_limit_max_mov_ys[yi], vertical_limit_max_mov_zs[yi]),
                                  MVector3D(math.radians(vertical_limit_min_rot_xs[yi]), math.radians(vertical_limit_min_rot_ys[yi]), math.radians(vertical_limit_min_rot_zs[yi])),
                                  MVector3D(math.radians(vertical_limit_max_rot_xs[yi]), math.radians(vertical_limit_max_rot_ys[yi]), math.radians(vertical_limit_max_rot_zs[yi])),
                                  MVector3D(vertical_spring_constant_mov_xs[yi], vertical_spring_constant_mov_ys[yi], vertical_spring_constant_mov_zs[yi]), \
                                  MVector3D(vertical_spring_constant_rot_xs[yi], vertical_spring_constant_rot_ys[yi], vertical_spring_constant_rot_zs[yi]))   # noqa
                    created_joints[joint_key] = joint

                    if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                        logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                        prev_joint_cnt = len(created_joints) // 200
                    
                    if param_reverse_joint and prev_below_bone_name in registed_rigidbodies and prev_above_bone_name in registed_rigidbodies:
                        # 逆ジョイント
                        joint_name = f'↑|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
                        joint_key = f'1:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

                        if joint_key not in created_joints:
                            # 未登録のみ追加
                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
                                          model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                                          joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yi], reverse_limit_min_mov_ys[yi], reverse_limit_min_mov_zs[yi]), \
                                          MVector3D(reverse_limit_max_mov_xs[yi], reverse_limit_max_mov_ys[yi], reverse_limit_max_mov_zs[yi]),
                                          MVector3D(math.radians(reverse_limit_min_rot_xs[yi]), math.radians(reverse_limit_min_rot_ys[yi]), math.radians(reverse_limit_min_rot_zs[yi])),
                                          MVector3D(math.radians(reverse_limit_max_rot_xs[yi]), math.radians(reverse_limit_max_rot_ys[yi]), math.radians(reverse_limit_max_rot_zs[yi])),
                                          MVector3D(reverse_spring_constant_mov_xs[yi], reverse_spring_constant_mov_ys[yi], reverse_spring_constant_mov_zs[yi]), \
                                          MVector3D(reverse_spring_constant_rot_xs[yi], reverse_spring_constant_rot_ys[yi], reverse_spring_constant_rot_zs[yi]))
                            created_joints[joint_key] = joint

                            if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                                prev_joint_cnt = len(created_joints) // 200

                    # バランサー剛体が必要な場合
                    if param_option["rigidbody_balancer"]:
                        balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
                        joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
                        joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

                        joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
                                      model.rigidbodies[balancer_prev_below_bone_name].index,
                                      joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                      MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))
                        created_joints[joint_key] = joint

                        # バランサー補助剛体
                        balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
                        joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
                        joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'
                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
                                      model.rigidbodies[balancer_prev_below_bone_name].index,
                                      MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
                                      MVector3D(), MVector3D(), MVector3D())
                        created_joints[joint_key] = joint
                                                    
            if param_horizonal_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
                # 横ジョイント
                if prev_above_bone_name != next_above_bone_name:
                    joint_name = f'→|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
                    joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                        
                        joint_vec = np.mean([prev_above_bone_position, prev_below_bone_position, \
                                             next_above_bone_position, next_below_bone_position])

                        # 回転量
                        joint_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                      model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi], horizonal_limit_min_mov_ys[yi], horizonal_limit_min_mov_zs[yi]), \
                                      MVector3D(horizonal_limit_max_mov_xs[yi], horizonal_limit_max_mov_ys[yi], horizonal_limit_max_mov_zs[yi]),
                                      MVector3D(math.radians(horizonal_limit_min_rot_xs[yi]), math.radians(horizonal_limit_min_rot_ys[yi]), math.radians(horizonal_limit_min_rot_zs[yi])),
                                      MVector3D(math.radians(horizonal_limit_max_rot_xs[yi]), math.radians(horizonal_limit_max_rot_ys[yi]), math.radians(horizonal_limit_max_rot_zs[yi])),
                                      MVector3D(horizonal_spring_constant_mov_xs[yi], horizonal_spring_constant_mov_ys[yi], horizonal_spring_constant_mov_zs[yi]), \
                                      MVector3D(horizonal_spring_constant_rot_xs[yi], horizonal_spring_constant_rot_ys[yi], horizonal_spring_constant_rot_zs[yi]))    # noqa
                        created_joints[joint_key] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                            prev_joint_cnt = len(created_joints) // 200
                        
                    if param_reverse_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
                        # 横逆ジョイント
                        joint_name = f'←|{registed_rigidbodies[next_above_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
                        joint_key = f'3:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

                        if joint_key not in created_joints:
                            # 未登録のみ追加
                            
                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index, \
                                          model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                                          joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yi], reverse_limit_min_mov_ys[yi], reverse_limit_min_mov_zs[yi]), \
                                          MVector3D(reverse_limit_max_mov_xs[yi], reverse_limit_max_mov_ys[yi], reverse_limit_max_mov_zs[yi]),
                                          MVector3D(math.radians(reverse_limit_min_rot_xs[yi]), math.radians(reverse_limit_min_rot_ys[yi]), math.radians(reverse_limit_min_rot_zs[yi])),
                                          MVector3D(math.radians(reverse_limit_max_rot_xs[yi]), math.radians(reverse_limit_max_rot_ys[yi]), math.radians(reverse_limit_max_rot_zs[yi])),
                                          MVector3D(reverse_spring_constant_mov_xs[yi], reverse_spring_constant_mov_ys[yi], reverse_spring_constant_mov_zs[yi]), \
                                          MVector3D(reverse_spring_constant_rot_xs[yi], reverse_spring_constant_rot_ys[yi], reverse_spring_constant_rot_zs[yi]))      # noqa
                            created_joints[joint_key] = joint

                            if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                                prev_joint_cnt = len(created_joints) // 200
                            
            if param_diagonal_joint and prev_above_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies and \
                    prev_below_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:                                # noqa
                # ＼ジョイント
                joint_name = f'＼|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
                joint_key = f'4:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

                if joint_key not in created_joints:
                    # 未登録のみ追加
                    
                    # ＼ジョイント
                    joint_vec = np.mean([prev_below_bone_position, next_below_bone_position])

                    # 回転量
                    joint_axis_up = (next_below_bone_position - prev_above_bone_position).normalized()
                    joint_axis = (prev_below_bone_position - next_above_bone_position).normalized()
                    joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                    joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                    joint_euler = joint_rotation_qq.toEulerAngles()
                    joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                    joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                  model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
                                  joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yi], diagonal_limit_min_mov_ys[yi], diagonal_limit_min_mov_zs[yi]), \
                                  MVector3D(diagonal_limit_max_mov_xs[yi], diagonal_limit_max_mov_ys[yi], diagonal_limit_max_mov_zs[yi]),
                                  MVector3D(math.radians(diagonal_limit_min_rot_xs[yi]), math.radians(diagonal_limit_min_rot_ys[yi]), math.radians(diagonal_limit_min_rot_zs[yi])),
                                  MVector3D(math.radians(diagonal_limit_max_rot_xs[yi]), math.radians(diagonal_limit_max_rot_ys[yi]), math.radians(diagonal_limit_max_rot_zs[yi])),
                                  MVector3D(diagonal_spring_constant_mov_xs[yi], diagonal_spring_constant_mov_ys[yi], diagonal_spring_constant_mov_zs[yi]), \
                                  MVector3D(diagonal_spring_constant_rot_xs[yi], diagonal_spring_constant_rot_ys[yi], diagonal_spring_constant_rot_zs[yi]))   # noqa
                    created_joints[joint_key] = joint

                    if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                        logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                        prev_joint_cnt = len(created_joints) // 200
                    
                # ／ジョイント ---------------
                joint_name = f'／|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
                joint_key = f'5:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

                if joint_key not in created_joints:
                    # 未登録のみ追加
                
                    # ／ジョイント

                    # 回転量
                    joint_axis_up = (prev_below_bone_position - next_above_bone_position).normalized()
                    joint_axis = (next_below_bone_position - prev_above_bone_position).normalized()
                    joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                    joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                    joint_euler = joint_rotation_qq.toEulerAngles()
                    joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                    joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
                                  model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
                                  joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yi], diagonal_limit_min_mov_ys[yi], diagonal_limit_min_mov_zs[yi]), \
                                  MVector3D(diagonal_limit_max_mov_xs[yi], diagonal_limit_max_mov_ys[yi], diagonal_limit_max_mov_zs[yi]),
                                  MVector3D(math.radians(diagonal_limit_min_rot_xs[yi]), math.radians(diagonal_limit_min_rot_ys[yi]), math.radians(diagonal_limit_min_rot_zs[yi])),
                                  MVector3D(math.radians(diagonal_limit_max_rot_xs[yi]), math.radians(diagonal_limit_max_rot_ys[yi]), math.radians(diagonal_limit_max_rot_zs[yi])),
                                  MVector3D(diagonal_spring_constant_mov_xs[yi], diagonal_spring_constant_mov_ys[yi], diagonal_spring_constant_mov_zs[yi]), \
                                  MVector3D(diagonal_spring_constant_rot_xs[yi], diagonal_spring_constant_rot_ys[yi], diagonal_spring_constant_rot_zs[yi]))   # noqa
                    created_joints[joint_key] = joint

                    if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                        logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                        prev_joint_cnt = len(created_joints) // 200
                    
        logger.info("-- ジョイント: %s個目:終了", len(created_joints))

        for joint_key in sorted(created_joints.keys()):
            # ジョイントを登録
            joint = created_joints[joint_key]
            joint.index = len(model.joints)

            if joint.name in model.joints:
                logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
                joint.name += randomname(3)

            model.joints[joint.name] = joint

    def create_joint(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
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

            vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
            vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
            vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

            vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
            vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
            vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

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
                horizonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x()]])), xs)             # noqa
                horizonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y()]])), xs)             # noqa
                horizonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z()]])), xs)             # noqa

                horizonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x()]])), xs)             # noqa
                horizonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y()]])), xs)             # noqa
                horizonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                    [param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z()]])), xs)             # noqa
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

            diagonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x()]])), xs)             # noqa
            diagonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y()]])), xs)             # noqa
            diagonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z()]])), xs)             # noqa

            diagonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x()]])), xs)             # noqa
            diagonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y()]])), xs)             # noqa
            diagonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z()]])), xs)             # noqa

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

            reverse_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x()]])), xs)             # noqa
            reverse_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y()]])), xs)             # noqa
            reverse_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z()]])), xs)             # noqa

            reverse_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x()]])), xs)             # noqa
            reverse_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y()]])), xs)             # noqa
            reverse_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
                [param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z()]])), xs)             # noqa

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

        for yi, (below_below_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[-2:-1], v_yidxs[-1:])):
            # ルート剛体と先頭剛体を繋ぐジョイント
            below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())

            if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
                # 繋がってる場合、最後に最初のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
            elif len(registed_bone_indexs[below_v_yidx]) > 2:
                # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]

            for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
                prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
                next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
                below_v_yno = below_v_yidx + 1

                prev_above_bone_name = root_rigidbody.name
                prev_above_bone_position = root_rigidbody.shape_position
                prev_below_bone_name = self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)
                prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
                next_below_bone_name = self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)
                next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

                prev_below_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_below_v_yidx].values())) - registed_bone_indexs[below_v_yidx][prev_below_v_xidx])
                prev_below_below_v_xidx = list(registed_bone_indexs[below_below_v_yidx].values())[(0 if prev_below_v_xidx == 0 else np.argmin(prev_below_below_v_xidx_diff))]
                prev_below_below_bone_name = self.get_bone_name(abb_name, below_below_v_yidx + 1, prev_below_below_v_xidx + 1)
                prev_below_below_bone_position = tmp_all_bones[prev_below_below_bone_name]["bone"].position
                
                next_below_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_below_v_yidx].values())) - registed_bone_indexs[below_v_yidx][next_below_v_xidx])
                next_below_below_v_xidx = list(registed_bone_indexs[below_below_v_yidx].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_below_below_v_xidx_diff))]
                next_below_below_bone_name = self.get_bone_name(abb_name, below_below_v_yidx + 1, next_below_below_v_xidx + 1)
                next_below_below_bone_position = tmp_all_bones[next_below_below_bone_name]["bone"].position

                if prev_above_bone_name in model.rigidbodies and prev_below_bone_name in registed_rigidbodies:
                    joint_name = f'↓|{prev_above_bone_name}|{registed_rigidbodies[prev_below_bone_name]}'
                    joint_key = f'0:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                        
                        # 縦ジョイント
                        joint_vec = prev_below_bone_position

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (next_below_bone_position - prev_below_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx = 0
                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, \
                                      model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yidx], vertical_limit_min_mov_ys[yidx], vertical_limit_min_mov_zs[yidx]), \
                                      MVector3D(vertical_limit_max_mov_xs[yidx], vertical_limit_max_mov_ys[yidx], vertical_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(vertical_limit_min_rot_xs[yidx]), math.radians(vertical_limit_min_rot_ys[yidx]), math.radians(vertical_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(vertical_limit_max_rot_xs[yidx]), math.radians(vertical_limit_max_rot_ys[yidx]), math.radians(vertical_limit_max_rot_zs[yidx])),
                                      MVector3D(vertical_spring_constant_mov_xs[yidx], vertical_spring_constant_mov_ys[yidx], vertical_spring_constant_mov_zs[yidx]), \
                                      MVector3D(vertical_spring_constant_rot_xs[yidx], vertical_spring_constant_rot_ys[yidx], vertical_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint_key] = joint

                    # バランサー剛体が必要な場合
                    if param_option["rigidbody_balancer"]:
                        balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
                        joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
                        joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

                        joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
                                      model.rigidbodies[balancer_prev_below_bone_name].index,
                                      joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                      MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
                        created_joints[joint_key] = joint

                # 横ジョイント
                if prev_below_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:
                    joint_name = f'→|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
                    joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                        
                        joint_vec = np.mean([prev_below_below_bone_position, prev_below_bone_position, \
                                             next_below_below_bone_position, next_below_bone_position])

                        # 回転量
                        joint_axis_up = (next_below_bone_position - prev_below_bone_position).normalized()
                        joint_axis = (prev_below_below_bone_position - prev_below_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx = 0
                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
                                      model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi, xi], horizonal_limit_min_mov_ys[yi, xi], horizonal_limit_min_mov_zs[yi, xi]), \
                                      MVector3D(horizonal_limit_max_mov_xs[yi, xi], horizonal_limit_max_mov_ys[yi, xi], horizonal_limit_max_mov_zs[yi, xi]),
                                      MVector3D(math.radians(horizonal_limit_min_rot_xs[yidx]), math.radians(horizonal_limit_min_rot_ys[yidx]), math.radians(horizonal_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(horizonal_limit_max_rot_xs[yidx]), math.radians(horizonal_limit_max_rot_ys[yidx]), math.radians(horizonal_limit_max_rot_zs[yidx])),
                                      MVector3D(horizonal_spring_constant_mov_xs[yidx], horizonal_spring_constant_mov_ys[yidx], horizonal_spring_constant_mov_zs[yidx]), \
                                      MVector3D(horizonal_spring_constant_rot_xs[yidx], horizonal_spring_constant_rot_ys[yidx], horizonal_spring_constant_rot_zs[yidx]))    # noqa
                        created_joints[joint_key] = joint

        for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
            below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
            logger.debug(f"before yi: {yi}, below_v_xidxs: {below_v_xidxs}")

            if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
                # 繋がってる場合、最後に最初のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
            elif len(registed_bone_indexs[below_v_yidx]) > 2:
                # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]
            logger.debug(f"after yi: {yi}, below_v_xidxs: {below_v_xidxs}")

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
                
                next_above_bone_name = tmp_all_bones[next_below_bone_name]["parent"]
                next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position
                
                # next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - registed_bone_indexs[below_v_yidx][next_below_v_xidx])
                # next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
                # next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
                # next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

                if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
                    # 縦ジョイント
                    joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[prev_below_bone_name]}'
                    joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                        
                        # 縦ジョイント
                        joint_vec = prev_below_bone_position

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (next_above_bone_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
                        yidx = min(len(vertical_limit_min_mov_xs) - 1, yidx)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                      model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yidx], vertical_limit_min_mov_ys[yidx], vertical_limit_min_mov_zs[yidx]), \
                                      MVector3D(vertical_limit_max_mov_xs[yidx], vertical_limit_max_mov_ys[yidx], vertical_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(vertical_limit_min_rot_xs[yidx]), math.radians(vertical_limit_min_rot_ys[yidx]), math.radians(vertical_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(vertical_limit_max_rot_xs[yidx]), math.radians(vertical_limit_max_rot_ys[yidx]), math.radians(vertical_limit_max_rot_zs[yidx])),
                                      MVector3D(vertical_spring_constant_mov_xs[yidx], vertical_spring_constant_mov_ys[yidx], vertical_spring_constant_mov_zs[yidx]), \
                                      MVector3D(vertical_spring_constant_rot_xs[yidx], vertical_spring_constant_rot_ys[yidx], vertical_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint_key] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                            prev_joint_cnt = len(created_joints) // 200
                        
                        if param_reverse_joint:
                            # 逆ジョイント
                            joint_name = f'↑|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
                            joint_key = f'1:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

                            if not (joint_key in created_joints or prev_below_bone_name not in registed_rigidbodies or prev_above_bone_name not in registed_rigidbodies):
                                # 未登録のみ追加
                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
                                              model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                                              joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
                                              MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
                                              MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
                                              MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
                                              MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
                                              MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))  # noqa
                                created_joints[joint_key] = joint

                                if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                    logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                                    prev_joint_cnt = len(created_joints) // 200

                        # バランサー剛体が必要な場合
                        if param_option["rigidbody_balancer"]:
                            balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
                            joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
                            joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'   # noqa

                            joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

                            # 回転量
                            joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
                            joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                            joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                            joint_euler = joint_rotation_qq.toEulerAngles()
                            joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
                                          model.rigidbodies[balancer_prev_below_bone_name].index,
                                          joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
                                          MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))
                            created_joints[joint_key] = joint

                            # バランサー補助剛体
                            balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
                            joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
                            joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'
                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
                                          model.rigidbodies[balancer_prev_below_bone_name].index,
                                          MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
                                          MVector3D(), MVector3D(), MVector3D())
                            created_joints[joint_key] = joint
                                                                            
                if param_horizonal_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
                    # 横ジョイント
                    if xi < len(below_v_xidxs) - 1 and prev_above_bone_name != next_above_bone_name:
                        joint_name = f'→|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
                        joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

                        if joint_key not in created_joints:
                            # 未登録のみ追加
                            
                            joint_vec = np.mean([prev_above_bone_position, prev_below_bone_position, \
                                                 next_above_bone_position, next_below_bone_position])

                            # 回転量
                            joint_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
                            joint_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
                            joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                            joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                            joint_euler = joint_rotation_qq.toEulerAngles()
                            joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                            yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
                            yidx = min(len(horizonal_limit_min_mov_xs) - 1, yidx)

                            joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                          model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
                                          joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi, xi], horizonal_limit_min_mov_ys[yi, xi], horizonal_limit_min_mov_zs[yi, xi]), \
                                          MVector3D(horizonal_limit_max_mov_xs[yi, xi], horizonal_limit_max_mov_ys[yi, xi], horizonal_limit_max_mov_zs[yi, xi]),
                                          MVector3D(math.radians(horizonal_limit_min_rot_xs[yidx]), math.radians(horizonal_limit_min_rot_ys[yidx]), math.radians(horizonal_limit_min_rot_zs[yidx])),
                                          MVector3D(math.radians(horizonal_limit_max_rot_xs[yidx]), math.radians(horizonal_limit_max_rot_ys[yidx]), math.radians(horizonal_limit_max_rot_zs[yidx])),
                                          MVector3D(horizonal_spring_constant_mov_xs[yidx], horizonal_spring_constant_mov_ys[yidx], horizonal_spring_constant_mov_zs[yidx]), \
                                          MVector3D(horizonal_spring_constant_rot_xs[yidx], horizonal_spring_constant_rot_ys[yidx], horizonal_spring_constant_rot_zs[yidx]))
                            created_joints[joint_key] = joint

                            if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                                prev_joint_cnt = len(created_joints) // 200
                            
                        if param_reverse_joint:
                            # 横逆ジョイント
                            joint_name = f'←|{registed_rigidbodies[next_above_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
                            joint_key = f'3:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

                            if not (joint_key in created_joints or prev_above_bone_name not in registed_rigidbodies or next_above_bone_name not in registed_rigidbodies):
                                # 未登録のみ追加
                                
                                joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index, \
                                              model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
                                              joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
                                              MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
                                              MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
                                              MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
                                              MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
                                              MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))      # noqa
                                created_joints[joint_key] = joint

                                if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                                    logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                                    prev_joint_cnt = len(created_joints) // 200
                                
                if param_diagonal_joint and prev_above_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:
                    # ＼ジョイント
                    joint_name = f'＼|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
                    joint_key = f'4:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                        
                        # ＼ジョイント
                        joint_vec = np.mean([prev_below_bone_position, next_below_bone_position])

                        # 回転量
                        joint_axis_up = (next_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis = (prev_below_bone_position - next_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
                        yidx = min(len(diagonal_limit_min_mov_xs) - 1, yidx)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
                                      model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
                                      MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
                                      MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
                                      MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint_key] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                            prev_joint_cnt = len(created_joints) // 200
                        
                if param_diagonal_joint and prev_below_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:    # noqa
                    # ／ジョイント ---------------
                    joint_name = f'／|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
                    joint_key = f'5:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

                    if joint_key not in created_joints:
                        # 未登録のみ追加
                    
                        # ／ジョイント

                        # 回転量
                        joint_axis_up = (prev_below_bone_position - next_above_bone_position).normalized()
                        joint_axis = (next_below_bone_position - prev_above_bone_position).normalized()
                        joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
                        joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
                        joint_euler = joint_rotation_qq.toEulerAngles()
                        joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

                        yidx, _ = self.disassemble_bone_name(prev_below_bone_name)
                        yidx = min(len(diagonal_limit_min_mov_xs) - 1, yidx)

                        joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
                                      model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
                                      joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
                                      MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
                                      MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
                                      MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
                                      MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
                                      MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
                        created_joints[joint_key] = joint

                        if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
                            logger.info("-- ジョイント: %s個目:終了", len(created_joints))
                            prev_joint_cnt = len(created_joints) // 200

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
    
    def create_vertical_rigidbody(self, model: PmxModel, param_option: dict):
        bone_grid_cols = param_option["bone_grid_cols"]
        bone_grid_rows = param_option["bone_grid_rows"]
        bone_grid = param_option["bone_grid"]

        prev_rigidbody_cnt = 0

        registed_rigidbodies = {}

        # 剛体情報
        param_rigidbody = param_option['rigidbody']
        # 剛体係数
        coefficient = param_option['rigidbody_coefficient']

        # 親ボーンに紐付く剛体がある場合、それを利用
        parent_bone = model.bones[param_option['parent_bone_name']]
        parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
        if not parent_bone_rigidbody:
            # 親ボーンに紐付く剛体がない場合、自前で作成
            parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
                                              0, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
            parent_bone_rigidbody.index = len(model.rigidbodies)

            if parent_bone_rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
                parent_bone_rigidbody.name += randomname(3)

            # 登録したボーン名と剛体の対比表を保持
            registed_rigidbodies[model.bone_indexes[parent_bone_rigidbody.bone_index]] = parent_bone_rigidbody.name

            model.rigidbodies[parent_bone.name] = parent_bone_rigidbody

        target_rigidbodies = {}
        for pac in range(bone_grid_cols):
            target_rigidbodies[pac] = []
            # 剛体生成
            created_rigidbodies = {}
            # 剛体の質量
            created_rigidbody_masses = {}
            created_rigidbody_linear_dampinges = {}
            created_rigidbody_angular_dampinges = {}

            prev_above_bone_name = None
            prev_above_bone_position = None
            for par in range(bone_grid_rows):
                prev_above_bone_name = bone_grid[par][pac]
                if not prev_above_bone_name or prev_above_bone_name not in model.bones:
                    continue

                prev_above_bone_position = model.bones[prev_above_bone_name].position
                prev_above_bone_index = model.bones[prev_above_bone_name].index
                prev_below_bone_name = None
                prev_below_bone_position = None
                if prev_above_bone_name:
                    pbr = par + 1
                    if pbr in bone_grid and pac in bone_grid[pbr]:
                        prev_below_bone_name = bone_grid[pbr][pac]
                        if prev_below_bone_name:
                            prev_below_bone_position = model.bones[prev_below_bone_name].position
                        if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                            # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
                            prev_below_bone_name = prev_above_bone_name
                            prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
                    elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
                        prev_below_bone_name = prev_above_bone_name
                        prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

                if prev_above_bone_position and prev_below_bone_position:
                    target_rigidbodies[pac].append(prev_above_bone_name)

                    bone = model.bones[prev_above_bone_name]
                    if bone.index not in model.vertices:
                        if bone.tail_index in model.bone_indexes:
                            # ウェイトが乗っていない場合、ボーンの長さで見る
                            min_vertex = model.bones[model.bone_indexes[bone.tail_index]].position.data()
                        else:
                            min_vertex = np.array([0, 0, 0])
                        max_vertex = bone.position.data()
                        max_vertex[0] = 1
                    else:
                        # 剛体生成対象の場合のみ作成
                        vertex_list = []
                        normal_list = []
                        for vertex in model.vertices[bone.index]:
                            vertex_list.append(vertex.position.data().tolist())
                            normal_list.append(vertex.normal.data().tolist())
                        vertex_ary = np.array(vertex_list)
                        min_vertex = np.min(vertex_ary, axis=0)
                        max_vertex = np.max(vertex_ary, axis=0)

                    axis_vec = prev_below_bone_position - bone.position
                    
                    # 回転量
                    rot = MQuaternion.rotationTo(MVector3D(0, 1, 0), axis_vec.normalized())
                    shape_euler = rot.toEulerAngles()
                    shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

                    # サイズ
                    diff_size = np.abs(max_vertex - min_vertex)
                    shape_size = MVector3D(diff_size[0] * 0.3, abs(axis_vec.y() * 0.8), diff_size[2])
                    shape_position = bone.position + (prev_below_bone_position - bone.position) / 2

                    # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
                    mode = 2 if par == 0 else 1
                    shape_type = param_rigidbody.shape_type
                    mass = param_rigidbody.param.mass
                    linear_damping = param_rigidbody.param.linear_damping
                    angular_damping = param_rigidbody.param.angular_damping
                    rigidbody = RigidBody(prev_above_bone_name, prev_above_bone_name, prev_above_bone_index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
                                          shape_type, shape_size, shape_position, shape_rotation_radians, \
                                          mass, linear_damping, angular_damping, param_rigidbody.param.restitution, param_rigidbody.param.friction, mode)
                    # 別途保持しておく
                    created_rigidbodies[rigidbody.name] = rigidbody
                    created_rigidbody_masses[rigidbody.name] = mass
                    created_rigidbody_linear_dampinges[rigidbody.name] = linear_damping
                    created_rigidbody_angular_dampinges[rigidbody.name] = angular_damping
            
            if len(created_rigidbodies) == 0:
                continue

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
                    rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
                if min_linear_damping != max_linear_damping:
                    rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
                        min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
                if min_angular_damping != max_angular_damping:
                    rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
                        min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa
                
                if rigidbody.name in model.rigidbodies:
                    logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                    rigidbody.name += randomname(3)
            
                # 登録したボーン名と剛体の対比表を保持
                registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name

                model.rigidbodies[rigidbody.name] = rigidbody
            
            prev_rigidbody_cnt += len(created_rigidbodies)

        # バランサー剛体が必要な場合
        if param_option["rigidbody_balancer"]:
            # すべて非衝突対象
            balancer_no_collision_group = 0
            # 剛体生成
            created_rigidbodies = {}
            # ボーン生成
            created_bones = {}

            for rigidbody_params in target_rigidbodies.values():
                rigidbody_mass = 0
                rigidbody_volume = MVector3D()
                for org_rigidbody_name in reversed(rigidbody_params):
                    org_rigidbody = model.rigidbodies[org_rigidbody_name]
                    org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
                    org_tail_position = org_bone.tail_position
                    if org_bone.tail_index >= 0:
                        org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
                    org_axis = (org_tail_position - org_bone.position).normalized()

                    if rigidbody_mass > 0:
                        # 中間は子の1.5倍
                        org_rigidbody.param.mass = rigidbody_mass * 1.5
                    org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
                                                                   math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

                    # 名前にバランサー追加
                    rigidbody_name = f'B-{org_rigidbody_name}'
                    # 質量は子の1.5倍
                    rigidbody_mass = org_rigidbody.param.mass

                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
                        rigidbody_qq *= org_rigidbody_qq
                    else:
                        # 上を向いてたらX方向に反転
                        rigidbody_qq = org_rigidbody_qq
                        rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                    shape_euler = rigidbody_qq.toEulerAngles()
                    shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

                    # 剛体の位置は剛体の上端から反対向き
                    mat = MMatrix4x4()
                    mat.setToIdentity()
                    mat.translate(org_rigidbody.shape_position)
                    mat.rotate(org_rigidbody_qq)
                    # X方向に反転
                    mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    edge_pos = MVector3D()
                    if org_rigidbody.shape_type == 0:
                        # 球の場合、半径分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
                    elif org_rigidbody.shape_type == 1:
                        # 箱の場合、高さの半分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
                    elif org_rigidbody.shape_type == 2:
                        # カプセルの場合、高さの半分 + 半径
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

                    mat.translate(edge_pos)
                    
                    # 元剛体の先端位置
                    org_rigidbody_pos = mat * MVector3D()

                    mat2 = MMatrix4x4()
                    mat2.setToIdentity()
                    # 元剛体の先端位置
                    mat2.translate(org_rigidbody_pos)
                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
                        mat2.rotate(org_rigidbody_qq)
                    else:
                        # 上を向いてたらX方向に反転
                        mat2.rotate(org_rigidbody_qq)
                        mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    # バランサー剛体の位置
                    shape_position = mat2 * (edge_pos + rigidbody_volume * 2)

                    # バランサー剛体のサイズ
                    shape_size = org_rigidbody.shape_size + (rigidbody_volume * 4)

                    # バランサー剛体用のボーン
                    balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
                    created_bones[balancer_bone.name] = balancer_bone

                    rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
                                          2, shape_size, shape_position, shape_rotation_radians, \
                                          rigidbody_mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
                                          org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
                    created_rigidbodies[rigidbody.name] = rigidbody
                    # 子剛体のサイズを保持
                    rigidbody_volume += edge_pos

            for rigidbody_name in sorted(created_rigidbodies.keys()):
                # ボーンを登録
                bone = created_bones[rigidbody_name]
                bone.index = len(model.bones)
                model.bones[bone.name] = bone

                # 剛体を登録
                rigidbody = created_rigidbodies[rigidbody_name]
                rigidbody.bone_index = bone.index
                rigidbody.index = len(model.rigidbodies)

                if rigidbody.name in model.rigidbodies:
                    logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                    rigidbody.name += randomname(3)

                # 登録したボーン名と剛体の対比表を保持
                registed_rigidbodies[rigidbody_name] = rigidbody.name

                model.rigidbodies[rigidbody.name] = rigidbody
   
        logger.info("-- 剛体: %s個目:終了", prev_rigidbody_cnt)

        return parent_bone_rigidbody, registed_rigidbodies

    def create_rigidbody_by_bone_blocks(self, model: PmxModel, param_option: dict, bone_blocks: dict):
        # bone_grid_cols = param_option["bone_grid_cols"]
        bone_grid_rows = param_option["bone_grid_rows"]
        # bone_grid = param_option["bone_grid"]

        # 剛体生成
        registed_rigidbodies = {}
        created_rigidbodies = {}
        # 剛体の質量
        created_rigidbody_masses = {}
        created_rigidbody_linear_dampinges = {}
        created_rigidbody_angular_dampinges = {}
        prev_rigidbody_cnt = 0

        # 剛体情報
        param_rigidbody = param_option['rigidbody']
        # 剛体係数
        coefficient = param_option['rigidbody_coefficient']
        # 剛体形状
        rigidbody_shape_type = param_option["rigidbody_shape_type"]
        # 物理タイプ
        physics_type = param_option["physics_type"]

        rigidbody_limit_thicks = np.linspace(0.1, 0.3, bone_grid_rows)

        # 親ボーンに紐付く剛体がある場合、それを利用
        parent_bone = model.bones[param_option['parent_bone_name']]
        parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
        if not parent_bone_rigidbody:
            # 親ボーンに紐付く剛体がない場合、自前で作成
            parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
                                              parent_bone_rigidbody.shape_type, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
            parent_bone_rigidbody.index = len(model.rigidbodies)

            if parent_bone_rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
                parent_bone_rigidbody.name += randomname(3)

            model.rigidbodies[parent_bone.name] = parent_bone_rigidbody
        
        # 略称
        abb_name = param_option['abb_name']
        root_rigidbody_name = f'{abb_name}中心'
        
        # 中心剛体を接触なしボーン追従剛体で生成
        root_rigidbody = None if root_rigidbody_name not in model.rigidbodies else model.rigidbodies[root_rigidbody_name]
        if not root_rigidbody:
            root_rigidbody = RigidBody(root_rigidbody_name, root_rigidbody_name, parent_bone.index, param_rigidbody.collision_group, 0, \
                                       parent_bone_rigidbody.shape_type, parent_bone_rigidbody.shape_size, parent_bone_rigidbody.shape_position, \
                                       parent_bone_rigidbody.shape_rotation, 1, 0.5, 0.5, 0, 0, 0)
            root_rigidbody.index = len(model.rigidbodies)
            model.rigidbodies[root_rigidbody.name] = root_rigidbody

        # 登録したボーン名と剛体の対比表を保持
        registed_rigidbodies[model.bone_indexes[parent_bone.index]] = root_rigidbody.name

        target_rigidbodies = {}
        for bone_block in bone_blocks.values():
            prev_above_bone_name = bone_block['prev_above']
            prev_above_bone_position = bone_block['prev_above_pos']
            # prev_below_bone_name = bone_block['prev_below']
            prev_below_bone_position = bone_block['prev_below_pos']
            # next_above_bone_name = bone_block['next_above']
            next_above_bone_position = bone_block['next_above_pos']
            # next_below_bone_name = bone_block['next_below']
            next_below_bone_position = bone_block['next_below_pos']
            # prev_prev_above_bone_name = bone_block['prev_prev_above']
            prev_prev_above_bone_position = bone_block['prev_prev_above_pos']
            xi = bone_block['xi']
            yi = bone_block['yi']
            is_above_connected = bone_block['is_above_connected']

            if prev_above_bone_name in created_rigidbodies:
                continue

            prev_above_bone_index = -1
            if prev_above_bone_name in model.bones:
                prev_above_bone_index = model.bones[prev_above_bone_name].index
            else:
                continue
            
            if xi not in target_rigidbodies:
                target_rigidbodies[xi] = []

            target_rigidbodies[xi].append(prev_above_bone_name)

            # 剛体の傾き
            shape_axis = (prev_below_bone_position - prev_above_bone_position).round(5).normalized()
            shape_axis_up = (next_above_bone_position - prev_prev_above_bone_position).round(5).normalized()
            shape_axis_cross = MVector3D.crossProduct(shape_axis, shape_axis_up).round(5).normalized()

            # if shape_axis_up.round(2) == MVector3D(1, 0, 0):
            #     shape_rotation_qq = MQuaternion.fromEulerAngles(0, 180, 0)
            # else:
            #     shape_rotation_qq = MQuaternion.rotationTo(MVector3D(-1, 0, 0), shape_axis_up)

            shape_rotation_qq = MQuaternion.fromDirection(shape_axis, shape_axis_cross)
            if round(prev_below_bone_position.y(), 2) != round(prev_above_bone_position.y(), 2):
                shape_rotation_qq *= MQuaternion.fromEulerAngles(0, 0, -90)
                shape_rotation_qq *= MQuaternion.fromEulerAngles(-90, 0, 0)
                if is_above_connected:
                    shape_rotation_qq *= MQuaternion.fromEulerAngles(0, -90, 0)

            shape_rotation_euler = shape_rotation_qq.toEulerAngles()

            if round(prev_below_bone_position.y(), 2) == round(prev_above_bone_position.y(), 2):
                shape_rotation_euler.setX(90)
                
            shape_rotation_radians = MVector3D(math.radians(shape_rotation_euler.x()), math.radians(shape_rotation_euler.y()), math.radians(shape_rotation_euler.z()))

            # 剛体の大きさ
            if rigidbody_shape_type == 0:
                x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position), \
                                 prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                ball_size = max(0.25, x_size * 0.5)
                shape_size = MVector3D(ball_size, ball_size, ball_size)
            elif rigidbody_shape_type == 2:
                x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
                y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                if physics_type == logger.transtext('袖'):
                    shape_size = MVector3D(x_size * 0.4, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
                else:
                    shape_size = MVector3D(x_size * 0.5, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
            else:
                x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
                y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                shape_size = MVector3D(max(0.25, x_size * 0.55), max(0.25, y_size * 0.55), rigidbody_limit_thicks[yi])

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

            # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
            mode = 2 if yi == 0 else 1
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
                logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))
                prev_rigidbody_cnt = len(created_rigidbodies) // 200

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
                rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
            if min_linear_damping != max_linear_damping:
                rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
                    min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
            if min_angular_damping != max_angular_damping:
                rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
                    min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa

            if rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                rigidbody.name += randomname(3)

            # 登録したボーン名と剛体の対比表を保持
            registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name
            
            model.rigidbodies[rigidbody.name] = rigidbody

        logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

        # バランサー剛体が必要な場合
        if param_option["rigidbody_balancer"]:
            # すべて非衝突対象
            balancer_no_collision_group = 0
            # 剛体生成
            created_rigidbodies = {}
            # ボーン生成
            created_bones = {}

            for rigidbody_params in target_rigidbodies.values():
                rigidbody_mass = 0
                rigidbody_volume = MVector3D()
                for org_rigidbody_name in reversed(rigidbody_params):
                    org_rigidbody = model.rigidbodies[org_rigidbody_name]
                    org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
                    org_tail_position = org_bone.tail_position
                    if org_bone.tail_index >= 0:
                        org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
                    org_axis = (org_tail_position - org_bone.position).normalized()

                    if rigidbody_mass > 0:
                        # 中間は子の1.5倍
                        org_rigidbody.param.mass = rigidbody_mass * 1.5
                    org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
                                                                   math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

                    # 名前にバランサー追加
                    rigidbody_name = f'B-{org_rigidbody_name}'
                    # 質量は子の1.5倍
                    rigidbody_mass = org_rigidbody.param.mass

                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
                        rigidbody_qq *= org_rigidbody_qq
                    else:
                        # 上を向いてたらX方向に反転
                        rigidbody_qq = org_rigidbody_qq
                        rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                    shape_euler = rigidbody_qq.toEulerAngles()
                    shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

                    # 剛体の位置は剛体の上端から反対向き
                    mat = MMatrix4x4()
                    mat.setToIdentity()
                    mat.translate(org_rigidbody.shape_position)
                    mat.rotate(org_rigidbody_qq)
                    # X方向に反転
                    mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    edge_pos = MVector3D()
                    if org_rigidbody.shape_type == 0:
                        # 球の場合、半径分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
                    elif org_rigidbody.shape_type == 1:
                        # 箱の場合、高さの半分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
                    elif org_rigidbody.shape_type == 2:
                        # カプセルの場合、高さの半分 + 半径
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

                    mat.translate(-edge_pos)
                    
                    # 元剛体の先端位置
                    org_rigidbody_pos = mat * MVector3D()

                    mat2 = MMatrix4x4()
                    mat2.setToIdentity()
                    # 元剛体の先端位置
                    mat2.translate(org_rigidbody_pos)
                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
                        mat2.rotate(org_rigidbody_qq)
                    else:
                        # 上を向いてたらX方向に反転
                        mat2.rotate(org_rigidbody_qq)
                        mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    # バランサー剛体の位置
                    shape_position = mat2 * (-edge_pos - rigidbody_volume * 4)

                    # バランサー剛体のサイズ
                    shape_size = org_rigidbody.shape_size + (rigidbody_volume * 8)
                    if org_rigidbody.shape_type != 2:
                        shape_size.setX(0.3)

                    # バランサー剛体用のボーン
                    balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
                    created_bones[balancer_bone.name] = balancer_bone

                    rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
                                          2, shape_size, shape_position, shape_rotation_radians, \
                                          rigidbody_mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
                                          org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
                    created_rigidbodies[rigidbody.name] = rigidbody
                    # 子剛体のサイズを保持
                    rigidbody_volume += edge_pos

            for rigidbody_name in sorted(created_rigidbodies.keys()):
                # ボーンを登録
                bone = created_bones[rigidbody_name]
                bone.index = len(model.bones)
                model.bones[bone.name] = bone

                # 剛体を登録
                rigidbody = created_rigidbodies[rigidbody_name]
                rigidbody.bone_index = bone.index
                rigidbody.index = len(model.rigidbodies)

                if rigidbody.name in model.rigidbodies:
                    logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                    rigidbody.name += randomname(3)

                # 登録したボーン名と剛体の対比表を保持
                registed_rigidbodies[rigidbody_name] = rigidbody.name
                
                model.rigidbodies[rigidbody.name] = rigidbody

        logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

        return root_rigidbody, registed_rigidbodies

    def create_rigidbody(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict, root_bone: Bone):
        # 剛体生成
        registed_rigidbodies = {}
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
        # 剛体形状
        rigidbody_shape_type = param_option["rigidbody_shape_type"]
        # 物理タイプ
        physics_type = param_option["physics_type"]

        # 親ボーンに紐付く剛体がある場合、それを利用
        parent_bone = model.bones[param_option['parent_bone_name']]
        parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
        if not parent_bone_rigidbody:
            # 親ボーンに紐付く剛体がない場合、自前で作成
            parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
                                              0, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
            parent_bone_rigidbody.index = len(model.rigidbodies)

            if parent_bone_rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
                parent_bone_rigidbody.name += randomname(3)

            # 登録したボーン名と剛体の対比表を保持
            registed_rigidbodies[model.bone_indexes[parent_bone_rigidbody.bone_index]] = parent_bone_rigidbody.name

            model.rigidbodies[parent_bone.name] = parent_bone_rigidbody
        
        root_rigidbody = self.get_rigidbody(model, root_bone.name)
        if not root_rigidbody:
            # 中心剛体を接触なしボーン追従剛体で生成
            root_rigidbody = RigidBody(root_bone.name, root_bone.english_name, root_bone.index, param_rigidbody.collision_group, 0, \
                                       parent_bone_rigidbody.shape_type, parent_bone_rigidbody.shape_size, parent_bone_rigidbody.shape_position, \
                                       parent_bone_rigidbody.shape_rotation, 1, 0.5, 0.5, 0, 0, 0)
            root_rigidbody.index = len(model.rigidbodies)
            model.rigidbodies[root_rigidbody.name] = root_rigidbody

        # 登録したボーン名と剛体の対比表を保持
        registed_rigidbodies[model.bone_indexes[root_bone.index]] = root_rigidbody.name

        v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
        rigidbody_limit_thicks = np.linspace(0.3, 0.1, len(v_yidxs))

        target_rigidbodies = {}
        for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
            above_v_xidxs = list(registed_bone_indexs[above_v_yidx].keys())
            logger.debug(f"yi: {yi}, above_v_xidxs: {above_v_xidxs}")

            if above_v_yidx < len(vertex_connected) and vertex_connected[above_v_yidx]:
                # 繋がってる場合、最後に最初のボーンを追加する
                above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[0]]
            elif len(registed_bone_indexs[above_v_yidx]) > 2:
                # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
                above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[-2]]
            logger.debug(f"yi: {yi}, above_v_xidxs: {above_v_xidxs}")

            target_rigidbodies[yi] = []

            for xi, (prev_above_vxidx, next_above_vxidx) in enumerate(zip(above_v_xidxs[:-1], above_v_xidxs[1:])):
                prev_above_v_xidx = registed_bone_indexs[above_v_yidx][prev_above_vxidx]
                prev_above_v_xno = prev_above_v_xidx + 1
                next_above_v_xidx = registed_bone_indexs[above_v_yidx][next_above_vxidx]
                next_above_v_xno = next_above_v_xidx + 1
                above_v_yno = above_v_yidx + 1

                prev_above_bone_name = self.get_bone_name(abb_name, above_v_yno, prev_above_v_xno)
                prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
                next_above_bone_name = self.get_bone_name(abb_name, above_v_yno, next_above_v_xno)
                next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

                prev_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, prev_above_v_xidx + 1)
                if prev_below_bone_name not in tmp_all_bones:
                    prev_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_v_yidx].values())) - registed_bone_indexs[above_v_yidx][prev_above_vxidx])
                    prev_below_v_xidx = list(registed_bone_indexs[below_v_yidx].values())[(0 if prev_above_vxidx == 0 else np.argmin(prev_below_v_xidx_diff))]
                    prev_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, prev_below_v_xidx + 1)
                prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
                
                next_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, next_above_v_xidx + 1)
                if next_below_bone_name not in tmp_all_bones:
                    next_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_v_yidx].values())) - registed_bone_indexs[above_v_yidx][next_above_vxidx])
                    next_below_v_xidx = list(registed_bone_indexs[below_v_yidx].values())[(0 if next_above_vxidx == 0 else np.argmin(next_below_v_xidx_diff))]
                    next_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, next_below_v_xidx + 1)

                next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

                # prev_above_bone_name = tmp_all_bones[prev_below_bone_name]["parent"]
                # prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
                # prev_above_v_yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
    
                # next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - next_below_v_xidx)
                # next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
                # next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
                # next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

                prev_prev_above_bone_position = None
                if 0 == xi:
                    # 先頭の場合、繋がっていたら最後のを加える
                    if vertex_connected[above_v_yidx]:
                        prev_prev_above_v_xidx = list(registed_bone_indexs[above_v_yidx].keys())[-1]
                        prev_prev_above_bone_name = self.get_bone_name(abb_name, above_v_yidx + 1, prev_prev_above_v_xidx + 1)
                        if prev_prev_above_bone_name in tmp_all_bones:
                            prev_prev_above_bone_position = tmp_all_bones[prev_prev_above_bone_name]["bone"].position
                else:
                    prev_prev_above_v_xidx = registed_bone_indexs[above_v_yidx][above_v_xidxs[xi - 1]]
                    prev_prev_above_bone_name = self.get_bone_name(abb_name, above_v_yidx + 1, prev_prev_above_v_xidx + 1)
                    if prev_prev_above_bone_name in tmp_all_bones:
                        prev_prev_above_bone_position = tmp_all_bones[prev_prev_above_bone_name]["bone"].position
                
                if prev_above_bone_name in created_rigidbodies or (prev_above_bone_name in model.bones and not model.bones[prev_above_bone_name].getVisibleFlag()):
                    continue

                prev_above_bone_index = -1
                if prev_above_bone_name in model.bones:
                    prev_above_bone_index = model.bones[prev_above_bone_name].index

                target_rigidbodies[yi].append(prev_above_bone_name)

                # 剛体の傾き
                shape_axis = (prev_below_bone_position - prev_above_bone_position).round(5).normalized()
                if prev_prev_above_bone_position:
                    shape_axis_up = (next_above_bone_position - prev_prev_above_bone_position).round(5).normalized()
                else:
                    shape_axis_up = (next_above_bone_position - prev_above_bone_position).round(5).normalized()
                shape_axis_cross = MVector3D.crossProduct(shape_axis, shape_axis_up).round(5).normalized()

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
                if rigidbody_shape_type == 0:
                    x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position), \
                                     prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                    ball_size = max(0.25, x_size * 0.5)
                    shape_size = MVector3D(ball_size, ball_size, ball_size)
                elif rigidbody_shape_type == 2:
                    x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
                    y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
                    if physics_type == logger.transtext('袖'):
                        shape_size = MVector3D(x_size * 0.4, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
                    else:
                        shape_size = MVector3D(x_size * 0.5, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
                else:
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

                # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
                mode = 2 if yi == len(v_yidxs) - 2 else 1
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
                    logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))
                    prev_rigidbody_cnt = len(created_rigidbodies) // 200
        
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
                rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
            if min_linear_damping != max_linear_damping:
                rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
                    min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
            if min_angular_damping != max_angular_damping:
                rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
                    min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa

            if rigidbody.name in model.rigidbodies:
                logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                rigidbody.name += randomname(3)

            # 登録したボーン名と剛体の対比表を保持
            registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name
            
            model.rigidbodies[rigidbody.name] = rigidbody
            logger.debug(f"rigidbody: {rigidbody}")

        logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

        # バランサー剛体が必要な場合
        if param_option["rigidbody_balancer"]:
            # すべて非衝突対象
            balancer_no_collision_group = 0
            # 剛体生成
            created_rigidbodies = {}
            # ボーン生成
            created_bones = {}

            rigidbody_volume = MVector3D()
            rigidbody_mass = 0
            for yi in sorted(target_rigidbodies.keys()):
                rigidbody_params = target_rigidbodies[yi]
                for org_rigidbody_name in rigidbody_params:
                    org_rigidbody = model.rigidbodies[org_rigidbody_name]
                    org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
                    org_tail_position = org_bone.tail_position
                    if org_bone.tail_index >= 0:
                        org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
                    org_axis = (org_tail_position - org_bone.position).normalized()

                    if rigidbody_mass > 0:
                        # 中間は子の1.5倍
                        org_rigidbody.param.mass = rigidbody_mass * 1.5
                    org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
                                                                   math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

                    # 名前にバランサー追加
                    rigidbody_name = f'B-{org_rigidbody_name}'

                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
                        rigidbody_qq *= org_rigidbody_qq
                    else:
                        # 上を向いてたらX方向に反転
                        rigidbody_qq = org_rigidbody_qq
                        rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

                    shape_euler = rigidbody_qq.toEulerAngles()
                    shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

                    # 剛体の位置は剛体の上端から反対向き
                    mat = MMatrix4x4()
                    mat.setToIdentity()
                    mat.translate(org_rigidbody.shape_position)
                    mat.rotate(org_rigidbody_qq)
                    # X方向に反転
                    mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    edge_pos = MVector3D()
                    if org_rigidbody.shape_type == 0:
                        # 球の場合、半径分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
                    elif org_rigidbody.shape_type == 1:
                        # 箱の場合、高さの半分移動
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
                    elif org_rigidbody.shape_type == 2:
                        # カプセルの場合、高さの半分 + 半径
                        edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

                    mat.translate(-edge_pos)
                    
                    # 元剛体の先端位置
                    org_rigidbody_pos = mat * MVector3D()

                    mat2 = MMatrix4x4()
                    mat2.setToIdentity()
                    # 元剛体の先端位置
                    mat2.translate(org_rigidbody_pos)
                    if org_axis.y() < 0:
                        # 下を向いてたらY方向に反転
                        mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
                        mat2.rotate(org_rigidbody_qq)
                    else:
                        # 上を向いてたらX方向に反転
                        mat2.rotate(org_rigidbody_qq)
                        mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

                    # バランサー剛体の位置
                    shape_position = mat2 * (-edge_pos - rigidbody_volume * 4)

                    # バランサー剛体のサイズ
                    shape_size = org_rigidbody.shape_size + (rigidbody_volume * 8)
                    if org_rigidbody.shape_type != 2:
                        shape_size.setX(0.3)

                    # バランサー剛体用のボーン
                    balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
                    created_bones[balancer_bone.name] = balancer_bone

                    rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
                                          2, shape_size, shape_position, shape_rotation_radians, \
                                          org_rigidbody.param.mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
                                          org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
                    created_rigidbodies[rigidbody.name] = rigidbody
                # 子剛体のサイズを保持
                rigidbody_volume += edge_pos
                # 質量は子の1.5倍
                rigidbody_mass = org_rigidbody.param.mass

            for rigidbody_name in sorted(created_rigidbodies.keys()):
                # ボーンを登録
                bone = created_bones[rigidbody_name]
                bone.index = len(model.bones)
                model.bones[bone.name] = bone

                # 剛体を登録
                rigidbody = created_rigidbodies[rigidbody_name]
                rigidbody.bone_index = bone.index
                rigidbody.index = len(model.rigidbodies)

                if rigidbody.name in model.rigidbodies:
                    logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
                    rigidbody.name += randomname(3)

                # 登録したボーン名と剛体の対比表を保持
                registed_rigidbodies[rigidbody_name] = rigidbody.name
                
                model.rigidbodies[rigidbody.name] = rigidbody

        logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

        return root_rigidbody, registed_rigidbodies

    def create_weight(self, model: PmxModel, param_option: dict, vertex_map: np.ndarray, vertex_connected: dict, duplicate_vertices: dict, \
                      registed_bone_indexs: dict, bone_horizonal_distances: dict, bone_vertical_distances: dict, vertex_remaining_set: set, target_vertices: list):
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
                        if vertex_idx < 0 or vertex_idx not in target_vertices:
                            continue

                        horizonal_distance = np.sum(b_h_distances[vi, :])
                        v_horizonal_distance = np.sum(b_h_distances[vi, :(vhi + 1)]) - b_h_distances[vi, 0]
                        vertical_distance = np.sum(b_v_distances[:, vhi])
                        v_vertical_distance = np.sum(b_v_distances[:(vi + 1), vhi]) - b_v_distances[0, vhi]

                        prev_above_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
                        prev_below_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))
                        next_above_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
                        next_below_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))

                        if below_v_yidx == v_yidxs[0]:
                            # 最下段は末端ボーンにウェイトを振らない
                            # 処理対象全ボーン名
                            weight_bones = [prev_above_bone, next_above_bone]
                            # ウェイト
                            total_weights = [prev_above_weight + prev_below_weight, next_above_weight + next_below_weight]
                        else:
                            # 全処理対象ボーン名
                            weight_bones = [prev_above_bone, next_above_bone, prev_below_bone, next_below_bone]
                            # ウェイト
                            total_weights = [prev_above_weight, next_above_weight, prev_below_weight, next_below_weight]

                        bone_weights = {}
                        for b, w in zip(weight_bones, total_weights):
                            if b and b.getVisibleFlag():
                                if b not in bone_weights:
                                    bone_weights[b.name] = 0
                                bone_weights[b.name] += w
                        
                        if len(bone_weights) > 2:
                            for _ in range(len(bone_weights), 5):
                                bone_weights[param_option['parent_bone_name']] = 0

                        # 対象となるウェイト値
                        weight_names = list(bone_weights.keys())
                        total_weights = np.array(list(bone_weights.values()))

                        if len(np.nonzero(total_weights)[0]) > 0:
                            weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                            weight_idxs = np.argsort(weights)
                            v = model.vertex_dict[vertex_idx]
                            vertex_remaining_set -= set(duplicate_vertices[v.position.to_log()])

                            for vvidx in duplicate_vertices[v.position.to_log()]:
                                vv = model.vertex_dict[vvidx]

                                logger.debug(f'vertex_idx: {vvidx}, weight_names: [{weight_names}], total_weights: [{total_weights}]')

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
                                    logger.info("-- 頂点ウェイト: %s個目:終了", weight_cnt)
                                    prev_weight_cnt = weight_cnt // 1000

        logger.info("-- 頂点ウェイト: %s個目:終了", weight_cnt)
        
        return vertex_remaining_set

    def create_remaining_weight(self, model: PmxModel, param_option: dict, vertex_maps: dict, \
                                vertex_remaining_set: set, boned_base_map_idxs: list, target_vertices: list):
        # ウェイト分布
        prev_weight_cnt = 0
        weight_cnt = 0

        vertex_distances = {}
        for boned_map_idx in boned_base_map_idxs:
            # 登録済み頂点との距離を測る（一番近いのと似たウェイト構成になるはず）
            boned_vertex_map = vertex_maps[boned_map_idx]
            for yi in range(boned_vertex_map.shape[0] - 1):
                for xi in range(boned_vertex_map.shape[1] - 1):
                    if boned_vertex_map[yi, xi] >= 0:
                        vi = boned_vertex_map[yi, xi]
                        vertex_distances[vi] = model.vertex_dict[vi].position.data()

        # 基準頂点マップ以外の頂点が残っていたら、それも割り当てる
        for vertex_idx in list(vertex_remaining_set):
            v = model.vertex_dict[vertex_idx]
            if vertex_idx < 0 or vertex_idx not in target_vertices:
                continue
            
            # 各頂点の位置との差分から距離を測る
            rv_distances = np.linalg.norm((np.array(list(vertex_distances.values())) - v.position.data()), ord=2, axis=1)

            # 近い頂点のうち、親ボーンにウェイトが乗ってないのを選択
            for nearest_vi in np.argsort(rv_distances):
                nearest_vidx = list(vertex_distances.keys())[nearest_vi]
                nearest_v = model.vertex_dict[nearest_vidx]
                nearest_deform = nearest_v.deform
                if type(nearest_deform) is Bdef1 and nearest_deform.index0 == model.bones[param_option['parent_bone_name']].index:
                    # 直近が親ボーンの場合、一旦スルー
                    continue
                else:
                    break

            if type(nearest_deform) is Bdef1:
                logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{model.bone_indexes[nearest_deform.index0]}], total_weights: [1]')

                v.deform = Bdef1(nearest_deform.index0)
            elif type(nearest_deform) is Bdef2:
                weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
                weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]

                bone1_distance = v.position.distanceToPoint(weight_bone1.position)
                bone2_distance = v.position.distanceToPoint(weight_bone2.position) if nearest_deform.weight0 < 1 else 0

                weight_names = np.array([weight_bone1.name, weight_bone2.name])
                if bone1_distance + bone2_distance != 0:
                    total_weights = np.array([bone1_distance / (bone1_distance + bone2_distance), bone2_distance / (bone1_distance + bone2_distance)])
                else:
                    total_weights = np.array([1, 0])
                    logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", v.index)
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{weight_names}], total_weights: [{total_weights}]')
                
                if np.count_nonzero(weights) == 1:
                    v.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                elif np.count_nonzero(weights) == 2:
                    v.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
                else:
                    v.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
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
                if all_distance != 0:
                    total_weights = np.array([bone1_distance / all_distance, bone2_distance / all_distance, bone3_distance / all_distance, bone4_distance / all_distance])
                else:
                    total_weights = np.array([1, bone2_distance, bone3_distance, bone4_distance])
                    logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", v.index)
                weights = total_weights / total_weights.sum(axis=0, keepdims=1)
                weight_idxs = np.argsort(weights)

                logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{weight_names}], total_weights: [{total_weights}]')

                if np.count_nonzero(weights) == 1:
                    v.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
                elif np.count_nonzero(weights) == 2:
                    v.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
                else:
                    v.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
                                     model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
                                     weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])
            
            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 100 > prev_weight_cnt:
                logger.info("-- 残頂点ウェイト: %s個目:終了", weight_cnt)
                prev_weight_cnt = weight_cnt // 100

        logger.info("-- 残頂点ウェイト: %s個目:終了", weight_cnt)

    def create_back_weight(self, model: PmxModel, param_option: dict):
        # ウェイト分布
        prev_weight_cnt = 0
        weight_cnt = 0

        front_vertex_keys = []
        front_vertex_positions = []
        for front_vertex_idx in list(model.material_vertices[param_option['material_name']]):
            front_vertex_keys.append(front_vertex_idx)
            front_vertex_positions.append(model.vertex_dict[front_vertex_idx].position.data())

        for vertex_idx in list(model.material_vertices[param_option['back_material_name']]):
            bv = model.vertex_dict[vertex_idx]

            # 各頂点の位置との差分から距離を測る
            bv_distances = np.linalg.norm((np.array(front_vertex_positions) - bv.position.data()), ord=2, axis=1)

            # 直近頂点INDEXのウェイトを転写
            copy_front_vertex_idx = front_vertex_keys[np.argmin(bv_distances)]
            bv.deform = copy.deepcopy(model.vertex_dict[copy_front_vertex_idx].deform)

            weight_cnt += 1
            if weight_cnt > 0 and weight_cnt // 200 > prev_weight_cnt:
                logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)
                prev_weight_cnt = weight_cnt // 200

        logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)

    def create_root_bone(self, model: PmxModel, param_option: dict):
        # 略称
        abb_name = param_option['abb_name']

        root_bone = Bone(f'{abb_name}中心', f'{abb_name}中心', model.bones[param_option['parent_bone_name']].position, \
                         model.bones[param_option['parent_bone_name']].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
        root_bone.index = len(list(model.bones.keys()))

        # ボーン
        model.bones[root_bone.name] = root_bone
        model.bone_indexes[root_bone.index] = root_bone.name

        return root_bone

    def create_bone(self, model: PmxModel, param_option: dict, vertex_map_orders: list, vertex_maps: dict, vertex_connecteds: dict, virtual_vertices: dict):
        # 中心ボーン生成

        # 略称
        abb_name = param_option['abb_name']
        # 材質名
        material_name = param_option['material_name']
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"

        # 表示枠定義
        model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

        root_bone = self.create_root_bone(model, param_option)
        
        # 表示枠
        model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

        tmp_all_bones = {}
        all_yidxs = {}
        all_bone_indexes = {}
        all_registed_bone_indexs = {}

        all_bone_horizonal_distances = {}
        all_bone_vertical_distances = {}

        for base_map_idx, vertex_map in enumerate(vertex_maps):
            bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1] + 1))
            bone_vertical_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))

            # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx, v_xidx - 1] != np.inf).all():
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        prev_v_vec = now_v_vec if v_xidx == 0 else virtual_vertices[tuple(vertex_map[v_yidx, v_xidx - 1])].position()
                        # prev_v_vec = now_v_vec if v_xidx == 0 else model.vertex_dict[vertex_map[v_yidx, v_xidx - 1]].position
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)
                    if (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx - 1, v_xidx] != np.inf).all():
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        prev_v_vec = now_v_vec if v_yidx == 0 else virtual_vertices[tuple(vertex_map[v_yidx - 1, v_xidx])].position()
                        # now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
                        # prev_v_vec = now_v_vec if v_yidx == 0 else model.vertex_dict[vertex_map[v_yidx - 1, v_xidx]].position
                        bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)
                if (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx, 0] != np.inf).all():
                    # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
                    now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                    prev_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, 0])].position()
                    # now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
                    # prev_v_vec = model.vertex_dict[vertex_map[v_yidx, 0]].position
                    bone_horizonal_distances[v_yidx, v_xidx + 1] = now_v_vec.distanceToPoint(prev_v_vec)

            all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
            all_bone_vertical_distances[base_map_idx] = bone_vertical_distances

        for base_map_idx in vertex_map_orders:
            vertex_map = vertex_maps[base_map_idx]
            vertex_connected = vertex_connecteds[base_map_idx]

            bone_vertical_distances = all_bone_vertical_distances[base_map_idx]
            full_xs = np.arange(0, vertex_map.shape[1])[np.count_nonzero(bone_vertical_distances, axis=0) == max(np.count_nonzero(bone_vertical_distances, axis=0))]
            median_x = int(np.median(full_xs))
            median_y_distance = np.mean(bone_vertical_distances[:, median_x][np.nonzero(bone_vertical_distances[:, median_x])])

            prev_yi = 0
            v_yidxs = []
            if param_option["density_type"] == logger.transtext('距離'):
                for yi, bh in enumerate(bone_vertical_distances[1:, median_x]):
                    if yi == 0 or np.sum(bone_vertical_distances[prev_yi:(yi + 1), median_x]) >= median_y_distance * param_option["vertical_bone_density"] * 0.8:
                        v_yidxs.append(yi)
                        prev_yi = yi + 1
            else:
                v_yidxs = list(range(0, vertex_map.shape[0], param_option["horizonal_bone_density"]))
            if v_yidxs[-1] < vertex_map.shape[0] - 1:
                # 最下段は必ず登録
                v_yidxs = v_yidxs + [vertex_map.shape[0] - 1]
            all_yidxs[base_map_idx] = v_yidxs

            # 中央あたりの横幅中央値ベースで横の割りを決める
            bone_horizonal_distances = all_bone_horizonal_distances[base_map_idx]
            full_ys = [y for i, y in enumerate(v_yidxs) if np.count_nonzero(bone_horizonal_distances[i, :]) == max(np.count_nonzero(bone_horizonal_distances, axis=1))]
            if not full_ys:
                full_ys = v_yidxs
            median_y = int(np.median(full_ys))
            median_x_distance = np.median(bone_horizonal_distances[median_y, :][np.nonzero(bone_horizonal_distances[median_y, :])])

            prev_xi = 0
            base_v_xidxs = []
            if param_option["density_type"] == logger.transtext('距離'):
                # 距離ベースの場合、中間距離で割りを決める
                for xi, bh in enumerate(bone_horizonal_distances[median_y, 1:]):
                    if xi == 0 or np.sum(bone_horizonal_distances[median_y, prev_xi:(xi + 1)]) >= median_x_distance * param_option["horizonal_bone_density"] * 0.8:
                        base_v_xidxs.append(xi)
                        prev_xi = xi + 1
            else:
                base_v_xidxs = list(range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]))

            if base_v_xidxs[-1] < vertex_map.shape[1] - param_option["horizonal_bone_density"]:
                # 右端は必ず登録
                base_v_xidxs = base_v_xidxs + [vertex_map.shape[1] - param_option["horizonal_bone_density"]]

            all_bone_indexes[base_map_idx] = {}
            for yi in range(vertex_map.shape[0]):
                all_bone_indexes[base_map_idx][yi] = {}
                v_xidxs = copy.deepcopy(base_v_xidxs)
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
            v_yidxs = all_yidxs[base_map_idx]
            vertex_map = vertex_maps[base_map_idx]
            vertex_connected = vertex_connecteds[base_map_idx]
            registed_bone_indexs = {}

            for yi, v_yidx in enumerate(v_yidxs):
                for v_xidx, total_v_xidx in all_bone_indexes[base_map_idx][yi].items():
                    # if v_yidx >= vertex_map.shape[0] or v_xidx >= vertex_map.shape[1] or vertex_map[v_yidx, v_xidx] < 0:
                    if v_yidx >= vertex_map.shape[0] or v_xidx >= vertex_map.shape[1] or (vertex_map[v_yidx, v_xidx] == np.inf).any():
                        # 存在しない頂点はスルー
                        continue
                    
                    vv = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])]
                    # v = model.vertex_dict[vertex_map[v_yidx, v_xidx]]
                    v_xno = total_v_xidx + 1
                    v_yno = v_yidx + 1

                    # ボーン仮登録
                    bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                    bone = Bone(bone_name, bone_name, vv.position(), root_bone.index, 0, 0x0000 | 0x0002)
                    bone.local_z_vector = vv.normal().copy()
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
                            model.display_slots[display_name].references.append((0, parent_bone.index))

                        model.bones[bone.name] = bone
                        model.bone_indexes[bone.index] = bone.name
                        tmp_all_bones[bone.name]["regist"] = True

                        registed_bone_indexs[v_yidx][v_xidx] = total_v_xidx

                        # 前ボーンとして設定
                        prev_xidx = v_xidx
            
            logger.debug(f"registed_bone_indexs: {registed_bone_indexs}")

            all_registed_bone_indexs[base_map_idx] = registed_bone_indexs

        return root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances

    def create_bone2(self, model: PmxModel, param_option: dict, vertex_map_orders: list, vertex_maps: dict, vertex_connecteds: dict):
        # 中心ボーン生成

        # 略称
        abb_name = param_option['abb_name']
        # 材質名
        material_name = param_option['material_name']
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"

        # 表示枠定義
        model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

        root_bone = self.create_root_bone(model, param_option)
        
        # 表示枠
        model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

        tmp_all_bones = {}
        all_yidxs = {}
        all_bone_indexes = {}
        all_registed_bone_indexs = {}

        all_bone_horizonal_distances = {}
        all_bone_vertical_distances = {}

        for base_map_idx, vertex_map in enumerate(vertex_maps):
            bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1] + 1))
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
                if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx, 0] >= 0:
                    # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
                    now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
                    prev_v_vec = model.vertex_dict[vertex_map[v_yidx, 0]].position
                    bone_horizonal_distances[v_yidx, v_xidx + 1] = now_v_vec.distanceToPoint(prev_v_vec)

            all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
            all_bone_vertical_distances[base_map_idx] = bone_vertical_distances

        for base_map_idx in vertex_map_orders:
            vertex_map = vertex_maps[base_map_idx]
            vertex_connected = vertex_connecteds[base_map_idx]

            bone_vertical_distances = all_bone_vertical_distances[base_map_idx]
            full_xs = np.arange(0, vertex_map.shape[1])[np.count_nonzero(bone_vertical_distances, axis=0) == max(np.count_nonzero(bone_vertical_distances, axis=0))]
            median_x = int(np.median(full_xs))
            median_y_distance = np.mean(bone_vertical_distances[:, median_x][np.nonzero(bone_vertical_distances[:, median_x])])

            prev_yi = 0
            v_yidxs = []
            for yi, bh in enumerate(bone_vertical_distances[1:, median_x]):
                if yi == 0 or np.sum(bone_vertical_distances[prev_yi:(yi + 1), median_x]) >= median_y_distance * param_option["vertical_bone_density"] * 0.8:
                    v_yidxs.append(yi)
                    prev_yi = yi + 1
            if v_yidxs[-1] < vertex_map.shape[0] - 1:
                # 最下段は必ず登録
                v_yidxs = v_yidxs + [vertex_map.shape[0] - 1]
            all_yidxs[base_map_idx] = v_yidxs

            # 中央あたりの横幅中央値ベースで横の割りを決める
            bone_horizonal_distances = all_bone_horizonal_distances[base_map_idx]
            full_ys = [y for i, y in enumerate(v_yidxs) if np.count_nonzero(bone_horizonal_distances[i, :]) == max(np.count_nonzero(bone_horizonal_distances, axis=1))]
            if not full_ys:
                full_ys = v_yidxs
            median_y = int(np.median(full_ys))
            median_x_distance = np.median(bone_horizonal_distances[median_y, :][np.nonzero(bone_horizonal_distances[median_y, :])])

            prev_xi = 0
            base_v_xidxs = []
            if param_option["density_type"] == logger.transtext('距離'):
                # 距離ベースの場合、中間距離で割りを決める
                for xi, bh in enumerate(bone_horizonal_distances[median_y, 1:]):
                    if xi == 0 or np.sum(bone_horizonal_distances[median_y, prev_xi:(xi + 1)]) >= median_x_distance * param_option["horizonal_bone_density"] * 0.8:
                        base_v_xidxs.append(xi)
                        prev_xi = xi + 1
            else:
                base_v_xidxs = list(range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]))

            if base_v_xidxs[-1] < vertex_map.shape[1] - param_option["horizonal_bone_density"]:
                # 右端は必ず登録
                base_v_xidxs = base_v_xidxs + [vertex_map.shape[1] - param_option["horizonal_bone_density"]]

            all_bone_indexes[base_map_idx] = {}
            for yi in range(vertex_map.shape[0]):
                all_bone_indexes[base_map_idx][yi] = {}
                v_xidxs = copy.deepcopy(base_v_xidxs)
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
            v_yidxs = all_yidxs[base_map_idx]
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
                            model.display_slots[display_name].references.append((0, parent_bone.index))

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
    
    def get_saved_bone_names(self, model: PmxModel):
        saved_bone_names = []
        # 準標準ボーンまでは削除対象外
        saved_bone_names.extend(SEMI_STANDARD_BONE_NAMES)

        for pidx, param_option in enumerate(self.options.param_options):
            if param_option['exist_physics_clear'] == logger.transtext('そのまま'):
                continue

            edge_material_name = param_option['edge_material_name']
            back_material_name = param_option['back_material_name']
            weighted_bone_indexes = {}

            # 頂点CSVが指定されている場合、対象頂点リスト生成
            if param_option['vertices_csv']:
                target_vertices = []
                with open(param_option['vertices_csv'], encoding='cp932', mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)            # ヘッダーを読み飛ばす
                    for row in reader:
                        if len(row) > 1 and int(row[1]) in model.material_vertices[param_option['material_name']]:
                            target_vertices.append(int(row[1]))
            else:
                target_vertices = list(model.material_vertices[param_option['material_name']])
            
            if edge_material_name:
                target_vertices = list(set(target_vertices) | set(model.material_vertices[edge_material_name]))
            
            if back_material_name:
                target_vertices = list(set(target_vertices) | set(model.material_vertices[back_material_name]))

            if param_option['exist_physics_clear'] == logger.transtext('再利用'):
                # 再利用の場合、指定されている全ボーンを対象とする
                bone_grid = param_option["bone_grid"]
                bone_grid_cols = param_option["bone_grid_cols"]
                bone_grid_rows = param_option["bone_grid_rows"]

                for r in range(bone_grid_rows):
                    for c in range(bone_grid_cols):
                        if bone_grid[r][c]:
                            weighted_bone_indexes[bone_grid[r][c]] = model.bones[bone_grid[r][c]].index
            else:
                for vertex_idx in target_vertices:
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
            
            if param_option['exist_physics_clear'] == logger.transtext('再利用'):
                # 再利用する場合、ボーンは全部残す
                saved_bone_names.extend(list(model.bones.keys()))
            else:
                # 他の材質で該当ボーンにウェイト割り当てられている場合、ボーンの削除だけは避ける
                for bone_idx, vertices in model.vertices.items():
                    is_not_delete = False
                    if bone_idx in list(weighted_bone_indexes.values()) and len(vertices) > 0:
                        is_not_delete = False
                        for vertex in vertices:
                            if vertex.index not in target_vertices:
                                is_not_delete = True
                                for material_name, material_vertices in model.material_vertices.items():
                                    if vertex.index in material_vertices:
                                        break
                                logger.info("削除対象外ボーン: %s(%s), 対象外頂点: %s, 所属材質: %s", \
                                            model.bone_indexes[bone_idx], bone_idx, vertex.index, material_name)
                                break
                    if is_not_delete:
                        saved_bone_names.append(model.bone_indexes[bone_idx])

            # 非表示子ボーンも削除する
            for bone in model.bones.values():
                if not bone.getVisibleFlag() and bone.parent_index in model.bone_indexes and model.bone_indexes[bone.parent_index] in weighted_bone_indexes \
                        and model.bone_indexes[bone.parent_index] not in saved_bone_names:
                    weighted_bone_indexes[bone.name] = bone.index
            
            logger.debug('weighted_bone_indexes: %s', ", ".join(list(weighted_bone_indexes.keys())))
            logger.debug('saved_bone_names: %s', ", ".join(saved_bone_names))

        return saved_bone_names

    def clear_exist_physics(self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list, saved_bone_names: list):
        logger.info("%s: 削除対象抽出", material_name)

        weighted_bone_indexes = {}
        if param_option['exist_physics_clear'] == logger.transtext('再利用'):
            # 再利用の場合、指定されている全ボーンを対象とする
            bone_grid = param_option["bone_grid"]
            bone_grid_cols = param_option["bone_grid_cols"]
            bone_grid_rows = param_option["bone_grid_rows"]

            for r in range(bone_grid_rows):
                for c in range(bone_grid_cols):
                    if bone_grid[r][c]:
                        weighted_bone_indexes[bone_grid[r][c]] = model.bones[bone_grid[r][c]].index
        else:
            for vertex_idx in target_vertices:
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
        
        # 非表示子ボーンも削除する
        for bone in model.bones.values():
            if not bone.getVisibleFlag() and bone.parent_index in model.bone_indexes and model.bone_indexes[bone.parent_index] in weighted_bone_indexes \
                    and model.bone_indexes[bone.parent_index] not in SEMI_STANDARD_BONE_NAMES:
                weighted_bone_indexes[bone.name] = bone.index
        
        for bone in model.bones.values():
            is_target = True
            if bone.name in saved_bone_names and bone.name in weighted_bone_indexes:
                # 保存済みボーン名に入ってても対象外
                logger.warning("他の材質のウェイトボーンとして設定されているため、ボーン「%s」を削除対象外とします。", bone.name)
                is_target = False

            if is_target:
                for vertex in model.vertices.get(bone.name, []):
                    for vertex_weight_bone_index in vertex.get_idx_list():
                        if vertex_weight_bone_index not in weighted_bone_indexes.values():
                            # 他のボーンのウェイトが乗ってたら対象外
                            logger.warning("削除対象外ボーンにウェイトが乗っているため、ボーン「%s」を削除対象外とします。\n調査対象インデックス：%s", bone.name, vertex.index)
                            is_target = False
                            break

            if not is_target and bone.name in weighted_bone_indexes:
                logger.debug("他ウェイト対象外: %s", bone.name)
                del weighted_bone_indexes[bone.name]

        for bone_name, bone_index in weighted_bone_indexes.items():
            bone = model.bones[bone_name]
            for morph in model.org_morphs.values():
                if morph.morph_type == 2:
                    for offset in morph.offsets:
                        if type(offset) is BoneMorphData:
                            if offset.bone_index == bone_index:
                                logger.error("削除対象ボーンがボーンモーフとして登録されているため、削除出来ません。\n" \
                                             + "事前にボーンモーフから外すか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s), モーフ名: %s", \
                                             bone_name, morph.name, decoration=MLogger.DECORATION_BOX)
                                return None
            for bidx, bone in enumerate(model.bones.values()):
                if bone.parent_index == bone_index and bone.index not in weighted_bone_indexes.values():
                    logger.error("削除対象ボーンが削除対象外ボーンの親ボーンとして登録されているため、削除出来ません。\n" \
                                 + "事前に親子関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外子ボーン: %s(%s)", \
                                 bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
                    return None

                if (bone.getExternalRotationFlag() or bone.getExternalTranslationFlag()) \
                   and bone.effect_index == bone_index and bone.index not in weighted_bone_indexes.values():
                    logger.error("削除対象ボーンが削除対象外ボーンの付与親ボーンとして登録されているため、削除出来ません。\n" \
                                 + "事前に付与関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外付与子ボーン: %s(%s)", \
                                 bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
                    return None
                    
                if bone.getIkFlag():
                    if bone.ik.target_index == bone_index and bone.index not in weighted_bone_indexes.values():
                        logger.error("削除対象ボーンが削除対象外ボーンのリンクターゲットボーンとして登録されているため、削除出来ません。\n" \
                                     + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)", \
                                     bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
                        return None

                    for link in bone.ik.link:
                        if link.bone_index == bone_index and bone.index not in weighted_bone_indexes.values():
                            logger.error("削除対象ボーンが削除対象外ボーンのリンクボーンとして登録されているため、削除出来ません。\n" \
                                         + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)", \
                                         bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
                            return None
                           
        weighted_rigidbody_indexes = {}
        for rigidbody in model.rigidbodies.values():
            if rigidbody.index not in list(weighted_rigidbody_indexes.values()) and rigidbody.bone_index in list(weighted_bone_indexes.values()) \
               and model.bone_indexes[rigidbody.bone_index] not in SEMI_STANDARD_BONE_NAMES:
                weighted_rigidbody_indexes[rigidbody.name] = rigidbody.index

        weighted_joint_indexes = {}
        for joint in model.joints.values():
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_a in list(weighted_rigidbody_indexes.values()):
                weighted_joint_indexes[joint.name] = joint.name
            if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_b in list(weighted_rigidbody_indexes.values()):
                weighted_joint_indexes[joint.name] = joint.name

        logger.info("%s: 削除実行", material_name)

        logger.info('削除対象ボーンリスト: %s', ", ".join(list(weighted_bone_indexes.keys())))
        logger.info('削除対象剛体リスト: %s', ", ".join(list(weighted_rigidbody_indexes.keys())))
        logger.info('削除対象ジョイントリスト: %s', ", ".join((weighted_joint_indexes.keys())))

        # 削除
        for joint_name in weighted_joint_indexes.keys():
            del model.joints[joint_name]

        for rigidbody_name in weighted_rigidbody_indexes.keys():
            del model.rigidbodies[rigidbody_name]

        for bone_name in weighted_bone_indexes.keys():
            if bone_name not in saved_bone_names:
                del model.bones[bone_name]

        logger.info("%s: INDEX振り直し", material_name)

        reset_rigidbodies = {}
        for ridx, (rigidbody_name, rigidbody) in enumerate(model.rigidbodies.items()):
            reset_rigidbodies[rigidbody.index] = {'name': rigidbody_name, 'index': ridx}
            model.rigidbodies[rigidbody_name].index = ridx

        reset_bones = {}
        for bidx, (bone_name, bone) in enumerate(model.bones.items()):
            reset_bones[bone.index] = {'name': bone_name, 'index': bidx}
            model.bones[bone_name].index = bidx
            model.bone_indexes[bidx] = bone_name

        logger.info("%s: INDEX再割り当て", material_name)

        for jidx, (joint_name, joint) in enumerate(model.joints.items()):
            if joint.rigidbody_index_a in reset_rigidbodies:
                joint.rigidbody_index_a = reset_rigidbodies[joint.rigidbody_index_a]['index']
            if joint.rigidbody_index_b in reset_rigidbodies:
                joint.rigidbody_index_b = reset_rigidbodies[joint.rigidbody_index_b]['index']
        for rigidbody in model.rigidbodies.values():
            if rigidbody.bone_index in reset_bones:
                rigidbody.bone_index = reset_bones[rigidbody.bone_index]['index']
            else:
                rigidbody.bone_index = -1

        for display_slot in model.display_slots.values():
            new_references = []
            for display_type, bone_idx in display_slot.references:
                if display_type == 0:
                    if bone_idx in reset_bones:
                        new_references.append((display_type, reset_bones[bone_idx]['index']))
                else:
                    new_references.append((display_type, bone_idx))
            display_slot.references = new_references

        for morph in model.org_morphs.values():
            if morph.morph_type == 2:
                new_offsets = []
                for offset in morph.offsets:
                    if type(offset) is BoneMorphData:
                        if offset.bone_index in reset_bones:
                            offset.bone_index = reset_bones[offset.bone_index]['index']
                            new_offsets.append(offset)
                        else:
                            offset.bone_index = -1
                            new_offsets.append(offset)
                    else:
                        new_offsets.append(offset)
                morph.offsets = new_offsets

        for bidx, bone in enumerate(model.bones.values()):
            if bone.parent_index in reset_bones:
                bone.parent_index = reset_bones[bone.parent_index]['index']
            else:
                bone.parent_index = -1

            if bone.getConnectionFlag():
                if bone.tail_index in reset_bones:
                    bone.tail_index = reset_bones[bone.tail_index]['index']
                else:
                    bone.tail_index = -1

            if bone.getExternalRotationFlag() or bone.getExternalTranslationFlag():
                if bone.effect_index in reset_bones:
                    bone.effect_index = reset_bones[bone.effect_index]['index']
                else:
                    bone.effect_index = -1

            if bone.getIkFlag():
                if bone.ik.target_index in reset_bones:
                    bone.ik.target_index = reset_bones[bone.ik.target_index]['index']
                    for link in bone.ik.link:
                        link.bone_index = reset_bones[link.bone_index]['index']
                else:
                    bone.ik.target_index = -1
                    for link in bone.ik.link:
                        link.bone_index = -1

        for vidx, vertex in enumerate(model.vertex_dict.values()):
            if type(vertex.deform) is Bdef1:
                vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
            elif type(vertex.deform) is Bdef2:
                vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
                vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1
            elif type(vertex.deform) is Bdef4:
                vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
                vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1
                vertex.deform.index2 = reset_bones[vertex.deform.index2]['index'] if vertex.deform.index2 in reset_bones else -1
                vertex.deform.index3 = reset_bones[vertex.deform.index3]['index'] if vertex.deform.index3 in reset_bones else -1
            elif type(vertex.deform) is Sdef:
                vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
                vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1

        return model

    # 頂点を展開した図を作成
    def create_vertex_map(self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list):
        logger.info("%s: 面の抽出", material_name)
        logger.info("%s: 面の抽出準備", material_name)

        non_target_iidxs = []
        virtual_vertices = {}
        edge_pair_lkeys = {}
        for index_idx in model.material_indices[material_name]:
            # 頂点の組み合わせから面INDEXを引く
            if model.indices[index_idx][0] not in target_vertices or model.indices[index_idx][1] not in target_vertices or model.indices[index_idx][2] not in target_vertices:
                # 3つ揃ってない場合、スルー
                non_target_iidxs.append(index_idx)
                continue

            for v0_idx, v1_idx, v2_idx in zip(model.indices[index_idx], model.indices[index_idx][1:] + [model.indices[index_idx][0]], [model.indices[index_idx][2]] + model.indices[index_idx][:-1]):
                v0 = model.vertex_dict[v0_idx]
                v1 = model.vertex_dict[v1_idx]
                v2 = model.vertex_dict[v2_idx]

                v0_key = v0.position.to_key()
                v1_key = v1.position.to_key()

                if v0_key not in virtual_vertices:
                    virtual_vertices[v0_key] = VirtualVertex(v0_key)
                
                # 仮想頂点登録
                virtual_vertices[v0_key].append(v0, v1, v2, index_idx)

                # 一旦ルートボーンにウェイトを一括置換
                v0.deform = Bdef1(model.bones[param_option['parent_bone_name']].index)

                # 辺キー生成
                lkey = (min(v0_key, v1_key), max(v0_key, v1_key))
                if lkey not in edge_pair_lkeys:
                    edge_pair_lkeys[lkey] = []
                edge_pair_lkeys[lkey].append(index_idx)

        if len(virtual_vertices.keys()) == 0:
            logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None, None, None

        for key, virtual_vertex in virtual_vertices.items():
            logger.debug(f'[{key}] {virtual_vertex}')

        logger.info("%s: エッジの抽出準備", material_name)

        edge_line_pairs = {}
        for (min_vkey, max_vkey), line_iidxs in edge_pair_lkeys.items():
            if len(line_iidxs) == 1:
                # 1つの面にしか紐付いてない辺を抽出
                if min_vkey not in edge_line_pairs:
                    edge_line_pairs[min_vkey] = []
                if max_vkey not in edge_line_pairs:
                    edge_line_pairs[max_vkey] = []
                
                edge_line_pairs[min_vkey].append(max_vkey)
                edge_line_pairs[max_vkey].append(min_vkey)

        # 方向に応じて判定値を変える
        # デフォルトは下
        base_vertical_axis = MVector3D(0, -1, 0)
        if param_option['direction'] == '上':
            base_vertical_axis = MVector3D(0, 1, 0)
        elif param_option['direction'] == '右':
            base_vertical_axis = MVector3D(-1, 0, 0)
        elif param_option['direction'] == '左':
            base_vertical_axis = MVector3D(1, 0, 0)
        
        logger.info("%s: エッジの判定", material_name)

        # エッジを繋いでいく
        tmp_all_edge_lines = []
        edge_vkeys = []
        while len(edge_vkeys) < len(edge_line_pairs.keys()):
            _, tmp_all_edge_lines, edge_vkeys = self.get_edge_lines(edge_line_pairs, virtual_vertices, None, tmp_all_edge_lines, edge_vkeys)

        logger.info("%s: エッジの抽出", material_name)

        logger.debug('--------------------------------------')
        all_edge_lines = []
        for n, edge_line in enumerate(tmp_all_edge_lines):
            all_edge_lines.append([[]])
            for m, edge_vkey in enumerate(edge_line):
                if len(all_edge_lines[n][-1]) < 2:
                    logger.debug(f'{edge_vkey}({virtual_vertices[edge_vkey].vidxs()}) - ***')
                else:
                    prev_vkey = all_edge_lines[n][-1][-2]
                    now_vkey = all_edge_lines[n][-1][-1]
                    next_vkey = edge_vkey
                    prev_pos = virtual_vertices[prev_vkey].position()
                    now_pos = virtual_vertices[now_vkey].position()
                    next_pos = virtual_vertices[next_vkey].position()
                    prev_direction = (now_pos - prev_pos).normalized()
                    now_direction = (next_pos - now_pos).normalized()
                    dot = MVector3D.dotProduct(prev_direction, now_direction)
                    if dot < 0.5:
                        logger.debug('-------------------')
                        # ある程度角がある場合、別辺と見なす
                        all_edge_lines[n].append([copy.deepcopy(now_vkey)])
                        logger.debug(f'{now_vkey}({virtual_vertices[now_vkey].vidxs()}) - **')
                    logger.debug(f'{edge_vkey}({virtual_vertices[edge_vkey].vidxs()}) - {dot}')
                    
                all_edge_lines[n][-1].append(edge_vkey)
        
        for n, edge_lines in enumerate(all_edge_lines):
            if len(edge_lines) > 1 and edge_lines[0][0] in edge_line_pairs[edge_lines[-1][-1]]:
                is_connect = True
                if len(edge_lines[0]) > 1 and len(edge_lines[-1]) > 1:
                    end_prev_vkey = edge_lines[-1][-2]
                    end_now_vkey = edge_lines[-1][-1]
                    end_prev_pos = virtual_vertices[end_prev_vkey].position()
                    end_now_pos = virtual_vertices[end_now_vkey].position()
                    end_direction = (end_now_pos - end_prev_pos).normalized()

                    start_prev_vkey = edge_lines[0][0]
                    start_now_vkey = edge_lines[0][1]
                    start_prev_pos = virtual_vertices[start_prev_vkey].position()
                    start_now_pos = virtual_vertices[start_now_vkey].position()
                    start_direction = (start_now_pos - start_prev_pos).normalized()

                    is_connect = (MVector3D.dotProduct(end_direction, start_direction) > 0.8)
                    
                if is_connect:
                    # 最後と最初が繋がってる場合、一個ずらして削除
                    edge_lines[-1].extend(copy.deepcopy(edge_lines[0]))
                    del all_edge_lines[n][0]
                # elif virtual_vertices[end_now_vkey].vidxs()[0] in [lv.index for lv in virtual_vertices[start_prev_vkey].line_vertices]:
                #     # 最初と最後が辺として繋がっている場合、最後の一点を最初に追加
                #     edge_lines[0].insert(0, copy.deepcopy(edge_lines[-1][-1]))

        logger.info("%s: エッジの方向抽出", material_name)

        logger.debug('--------------------------------------')
        all_vertical_edge_lines = []
        vertical_edge_lines = []
        horizonal_edge_lines = []
        for i, edge_lines in enumerate(all_edge_lines):
            edge_dots = []
            for n, edge_line in enumerate(edge_lines):
                edge_dots.append([])
                for m, edge_vkey in enumerate(edge_line):
                    if m == 0:
                        logger.debug(f'[{n:02d}-{m:03d}] {edge_vkey}({virtual_vertices[edge_vkey].vidxs()}) **')
                    else:
                        now_vkey = edge_line[m - 1]
                        next_vkey = edge_line[m]
                        now_pos = virtual_vertices[now_vkey].position()
                        next_pos = virtual_vertices[next_vkey].position()
                        now_direction = (next_pos - now_pos).normalized()

                        # ローカル軸のvertical方向を求める
                        mat = MMatrix4x4()
                        mat.setToIdentity()
                        mat.translate(now_pos)
                        mat.rotate(MQuaternion.rotationTo(now_direction, base_vertical_axis))

                        vertical_pos = mat.inverted() * base_vertical_axis
                        vertical_direction = (vertical_pos - now_pos).normalized()

                        dot = MVector3D.dotProduct(vertical_direction, now_direction)
                        edge_dots[-1].append(dot)
                        logger.debug(f'[{n:02d}-{m:03d}] {edge_vkey}({virtual_vertices[edge_vkey].vidxs()}) - {now_direction.to_log()} ({dot})')
                logger.debug('--------------')

            # 全体のmeanで切り分ける
            mean_dot = np.mean([np.mean(np.abs(np.array(ed))) for ed in edge_dots])
            for n, (edge_line, edge_dot) in enumerate(zip(edge_lines, edge_dots)):
                edge_mean = np.mean(np.abs(edge_dot))
                if round(edge_mean, 3) <= round(mean_dot, 3):
                    # 進行方向と同じ向きの場合、水平方向
                    horizonal_edge_lines.append(edge_line)
                    logger.debug(f'[{n:02d}-horizonal] {edge_mean} - {mean_dot}')
                else:
                    vertical_edge_lines.append(edge_line)
                    all_vertical_edge_lines.extend(edge_line)
                    logger.debug(f'[{n:02d}-vertical] {edge_mean} - {mean_dot}')

        logger.info("%s: 水平エッジの上下判定", material_name)

        # 各頂点の位置との差分から距離を測る
        hel_distances = []
        for hel in horizonal_edge_lines:
            hepos = []
            for he in hel:
                hepos.append(virtual_vertices[he].position().data())
            he_distances = np.linalg.norm(np.array(hepos) - model.bones[param_option['parent_bone_name']].position.data(), ord=2, axis=1)
            hel_distances.append(np.mean(he_distances))
        
        mean_distance = np.mean(hel_distances)

        # horizonalを上下に分ける
        top_horizonal_edge_lines = []
        bottom_horizonal_edge_lines = []
        for n, (held, hel) in enumerate(zip(hel_distances, horizonal_edge_lines)):
            if held > mean_distance:
                # 遠い方が下
                bottom_horizonal_edge_lines.append(hel)
                logger.debug(f'[{n:02d}-horizonal-bottom] {hel}')
            else:
                # 近い方が上
                top_horizonal_edge_lines.append(hel)
                logger.debug(f'[{n:02d}-horizonal-top] {hel}')

        if len(tmp_all_edge_lines) == 2:
            # 元から二辺に分かれていた場合、4等分した時の方向で決める
            logger.info("%s: 水平エッジの向き判定", material_name)
            
            # 末端の平行方向の向きを確認する
            top_0_4_distance = virtual_vertices[top_horizonal_edge_lines[0][0]].position().distanceToPoint( \
                                virtual_vertices[bottom_horizonal_edge_lines[0][0]].position())   # noqa
            top_1_4_distance = virtual_vertices[top_horizonal_edge_lines[0][int(len(top_horizonal_edge_lines[0]) * 0.25)]].position().distanceToPoint( \
                                virtual_vertices[bottom_horizonal_edge_lines[0][int(len(bottom_horizonal_edge_lines[0]) * 0.25)]].position())   # noqa
            top_2_4_distance = virtual_vertices[top_horizonal_edge_lines[0][int(len(top_horizonal_edge_lines[0]) * 0.5)]].position().distanceToPoint( \
                                virtual_vertices[bottom_horizonal_edge_lines[0][int(len(bottom_horizonal_edge_lines[0]) * 0.5)]].position())   # noqa
            top_3_4_distance = virtual_vertices[top_horizonal_edge_lines[0][int(len(top_horizonal_edge_lines[0]) * 0.75)]].position().distanceToPoint( \
                                virtual_vertices[bottom_horizonal_edge_lines[0][int(len(bottom_horizonal_edge_lines[0]) * 0.75)]].position())   # noqa
            
            if np.round(np.mean([top_0_4_distance, top_2_4_distance])) < np.round(np.mean([top_1_4_distance, top_3_4_distance])):
                # 方向が逆転してたら、1/4のところで距離が大きくなるはず
                logger.debug(f'[reverse] {bottom_horizonal_edge_lines}')

                # TOPの開始地点に最短な頂点がBOTTOMの後の方である場合、反転
                bottom_horizonal_edge_lines.reverse()
                for bhel in bottom_horizonal_edge_lines:
                    bhel.reverse()

            logger.debug(f'[bottom_horizonal_edge_lines] {bottom_horizonal_edge_lines}')

            logger.info("%s: 水平エッジの開始位置判定", material_name)

            logger.debug('--------------------------------------')
            direction_dots = []
            for n, bottom_edge_vkey in enumerate(bottom_horizonal_edge_lines[0]):
                top_vv = virtual_vertices[top_horizonal_edge_lines[0][0]]
                bottom_vv = virtual_vertices[bottom_edge_vkey]
                top_pos = top_vv.position()
                bottom_pos = bottom_vv.position()
                now_direction = (bottom_pos - top_pos).normalized()

                # ローカル軸のvertical方向を求める
                mat = MMatrix4x4()
                mat.setToIdentity()
                mat.translate(top_pos)
                mat.rotate(MQuaternion.rotationTo(base_vertical_axis, now_direction))

                vertical_pos = mat.inverted() * base_vertical_axis
                vertical_direction = (vertical_pos - top_pos).normalized()

                dot = MVector3D.dotProduct(vertical_direction, now_direction)
                direction_dots.append(dot)
                logger.debug(f'[{n:03d}] {bottom_edge_vkey}({bottom_vv.vidxs()}) - [{now_direction.to_log()}] ({dot})')

            logger.debug('--------------')

            # 先頭をverticalなINDEXにするため、後ろにずらす
            for n in range(np.argmax(direction_dots)):
                bottom_horizonal_edge_lines[0].append(copy.deepcopy(bottom_horizonal_edge_lines[0][n]))

            # ズラし終わったら削除する
            for n in range(np.argmax(direction_dots)):
                del bottom_horizonal_edge_lines[0][0]

            logger.debug(f'[vertical bottom_horizonal_edge_lines] {bottom_horizonal_edge_lines}')
        else:
            logger.info("%s: 終端エッジの反転", material_name)

            # 一筆書きの場合、TOPとBOTTOMで向きが反転してるはずなので、BOTTOMの向きを反転させる
            logger.debug(f'[reverse] {bottom_horizonal_edge_lines}')

            bottom_horizonal_edge_lines.reverse()
            for bhel in bottom_horizonal_edge_lines:
                bhel.reverse()

            logger.debug(f'[bottom_horizonal_edge_lines] {bottom_horizonal_edge_lines}')

            logger.info("%s: 終端エッジの並べ替え", material_name)
            
            # 始端から最も近い終端から始める
            top_start_distances = []
            bottom_edge_start_idxs = []
            start_idx = 0
            for bhel in bottom_horizonal_edge_lines:
                bottom_edge_start_idxs.append(start_idx)
                for bhe in bhel:
                    top_start_distances.append(virtual_vertices[top_horizonal_edge_lines[0][0]].position().distanceToPoint(virtual_vertices[bhe].position()))
                    start_idx += 1

            logger.debug(f'[top_start_distances] {top_start_distances}')

            for start_idx in bottom_edge_start_idxs:
                if start_idx < np.argmin(top_start_distances):
                    # 最も近い終端より前の終端エッジは後ろに回す
                    bottom_horizonal_edge_lines.append(copy.deepcopy(bottom_horizonal_edge_lines[0]))
                    del bottom_horizonal_edge_lines[0]

            logger.debug(f'[bottom_horizonal_edge_lines] {bottom_horizonal_edge_lines}')

        logger.info("%s: 水平エッジの距離測定", material_name)

        # 上端の距離間隔
        top_distances = []
        top_keys = []
        top_poses = []
        for thel in top_horizonal_edge_lines:
            for m, the in enumerate(thel):
                if m == 0:
                    top_distances.append(0)
                else:
                    top_distances.append(virtual_vertices[thel[m - 1]].position().distanceToPoint(virtual_vertices[the].position()))
                top_keys.append(the)
                top_poses.append((virtual_vertices[the].position()).data())
        top_distance_ratios = np.array([np.sum(top_distances[:(m + 1)]) for m in range(len(top_distances))]) / np.sum(top_distances)

        # 末端の距離間隔
        bottom_distances = []
        bottom_keys = []
        bottom_poses = []
        for bhel in bottom_horizonal_edge_lines:
            for m, bhe in enumerate(bhel):
                if m == 0:
                    bottom_distances.append(0)
                else:
                    bottom_distances.append(virtual_vertices[bhel[m - 1]].position().distanceToPoint(virtual_vertices[bhe].position()))
                bottom_keys.append(bhe)
                bottom_poses.append((virtual_vertices[bhe].position()).data())
        bottom_distance_ratios = np.array([np.sum(bottom_distances[:(m + 1)]) for m in range(len(bottom_distances))]) / np.sum(bottom_distances)
        
        logger.debug(f'[top_distances] {top_distances}')
        logger.debug(f'[bottom_distances] {bottom_distances}')

        logger.debug(f'[top_distance_ratios ({len(top_distance_ratios)})] {top_distance_ratios}')
        logger.debug(f'[bottom_distance_ratios ({len(bottom_distance_ratios)})] {bottom_distance_ratios}')

        vertex_coordinate_maps = []
        bx = 0
        xx = 0
        for bhel in bottom_horizonal_edge_lines:
            vertex_coordinate_maps.append([])
            for bhe in bhel:
                if len(top_distances) == len(bottom_distances):
                    # 同じリング数の場合、同じINDEXのを選ぶ
                    tid = bx
                    tkeys = [top_keys[bx]]
                else:
                    direction_dots = []
                    for thel in top_horizonal_edge_lines:
                        for m, the in enumerate(thel):
                            top_vv = virtual_vertices[the]
                            bottom_vv = virtual_vertices[bhe]
                            top_pos = top_vv.position()
                            bottom_pos = bottom_vv.position()
                            now_direction = (bottom_pos - top_pos).normalized()

                            # ローカル軸のvertical方向を求める
                            mat = MMatrix4x4()
                            mat.setToIdentity()
                            mat.translate(top_pos)
                            mat.rotate(MQuaternion.rotationTo(base_vertical_axis, now_direction))

                            vertical_pos = mat.inverted() * base_vertical_axis
                            vertical_direction = (vertical_pos - top_pos).normalized()

                            # 出来るだけ直進している始端を選ぶ
                            dot = abs(MVector3D.dotProduct(vertical_direction, now_direction))
                            direction_dots.append(dot)

                    # # できるだけ直進しているのをチェック(近似値が複数取れる可能性)
                    # similarity = param_option['similarity'] * 150
                    # tids = np.where(direction_dots >= np.max(np.fix(np.array(direction_dots) * similarity) / similarity))[0]
                    # tkeys = np.array(top_keys)[tids]

                    # target_tids = np.where(diff_ratios <= np.min(np.ceil(np.array(diff_ratios) * similarity) / similarity))[0]
                    # tkeys = np.array(top_keys)[tids]

                    # # 最も直進しているのをチェック
                    # tids = [np.argmax(direction_dots)]
                    # tkeys = [top_keys[np.argmax(direction_dots)]]

                    # # 距離間隔比率のキー(近似値が複数取れる可能性)
                    # diff_ratios = np.abs(top_distance_ratios - bottom_distance_ratios[xx])
                    # tid = np.argmin(diff_ratios)

                    # 最も確率が高いの前後も含めてチェック
                    tid = np.argmax(direction_dots)
                    similarity = int((1 - param_option['similarity']) * 10)
                    tids = list(range(tid - similarity, tid + similarity + 1))
                    tkeys = np.array(top_keys + top_keys + top_keys)[np.array(tids)]

                if bx > 0 and tid > 0 and top_distances[tid] == 0:
                    # 上端の切れ目の場合、グループを変える
                    vertex_coordinate_maps.append([])

                for ci, tkey in enumerate(tkeys):
                    logger.debug(f'** start: bx: {bx}, top: {virtual_vertices[tuple(tkey)].vidxs()}, bottom: {virtual_vertices[bhe].vidxs()}, dots: [{np.round(direction_dots, decimals=3).tolist()}]')
                    logger.info('頂点ルート走査[%s-%s]: 始端: %s, 終端: %s', f'{(bx + 1):04d}', f'{(ci + 1):02d}', virtual_vertices[tuple(tkey)].vidxs(), virtual_vertices[bhe].vidxs())
                    vertex_coordinate_maps[-1].append(self.create_vertex_coordinate_map(bx, tuple(tkey), bhe, virtual_vertices, top_keys, bottom_keys, MVector3D(1, 0, 0)))
                    bx += 1
                xx += 1

        logger.info("%s: 絶対頂点マップの生成", material_name)
        vertex_maps = []
        vertex_connecteds = []

        midx = 0
        for vertex_coordinate_map in vertex_coordinate_maps:
            logger.info("-- 絶対頂点マップ: %s個目: ---------", midx + 1)

            vertex_connecteds.append([])
            vertex_tmp_dots = []
            top_keys = []

            logger.info("-- 絶対頂点マップ[%s]: 頂点ルート決定", midx + 1)

            for vcm in vertex_coordinate_map:
                vertex_tmp_dots.append([])
                vcm_reverse_list = list(reversed(list(vcm.values())))
                for y, vc in enumerate(vcm_reverse_list):
                    if y == 0:
                        vertex_tmp_dots[-1].append(1)
                        top_keys.append(vc['vv'])
                    elif y <= 1:
                        continue
                    else:
                        top_vv = virtual_vertices[vcm_reverse_list[0]['vv']]
                        bottom_vv = virtual_vertices[vcm_reverse_list[-1]['vv']]
                        top_pos = top_vv.position()
                        bottom_pos = bottom_vv.position()
                        total_direction = (bottom_pos - top_pos).normalized()

                        # ローカル軸のvertical方向を求める
                        mat = MMatrix4x4()
                        mat.setToIdentity()
                        mat.translate(top_pos)
                        mat.rotate(MQuaternion.rotationTo(base_vertical_axis, total_direction))

                        vertical_pos = mat.inverted() * base_vertical_axis
                        vertical_direction = (vertical_pos - top_pos).normalized()

                        prev_prev_vv = virtual_vertices[vcm_reverse_list[y - 2]['vv']]
                        prev_vv = virtual_vertices[vcm_reverse_list[y - 1]['vv']]
                        now_vv = virtual_vertices[vc['vv']]
                        prev_prev_pos = prev_prev_vv.position()
                        prev_pos = prev_vv.position()
                        now_pos = now_vv.position()
                        prev_direction = (prev_pos - prev_prev_pos).normalized()
                        now_direction = (now_pos - prev_pos).normalized()

                        # dot = MVector3D.dotProduct(now_direction, prev_direction) * (now_pos.distanceToPoint(prev_pos) / bottom_pos.distanceToPoint(top_pos))
                        dot = MVector3D.dotProduct(now_direction, prev_direction) * now_pos.distanceToPoint(prev_pos)
                        logger.debug(f"target top: [{virtual_vertices[vcm_reverse_list[0]['vv']].vidxs()}], bottom: [{virtual_vertices[vcm_reverse_list[-1]['vv']].vidxs()}], dot({y}): {round(dot, 3)}")   # noqa

                        vertex_tmp_dots[-1].append(dot)

                logger.info("-- 絶対頂点マップ[%s]: 頂点ルート確認[%s] 始端: %s, 終端: %s, 近似値: %s", midx + 1, len(top_keys), virtual_vertices[vcm_reverse_list[0]['vv']].vidxs(), \
                            virtual_vertices[vcm_reverse_list[-1]['vv']].vidxs(), round(np.mean(vertex_tmp_dots[-1]), 3))

            logger.debug('------------------')
            top_key_cnts = dict(Counter(top_keys))
            target_regists = [False for _ in range(len(vertex_coordinate_map))]
            if np.max(list(top_key_cnts.values())) > 1:
                # 同じ始端から2つ以上の末端に繋がっている場合
                for top_key, cnt in top_key_cnts.items():
                    vertex_mean_dots = {}
                    for x, vcm in enumerate(vertex_coordinate_map):
                        vcm_reverse_list = list(reversed(list(vcm.values())))
                        if vcm_reverse_list[0]['vv'] == top_key:
                            if cnt > 1:
                                # 2個以上同じ始端から出ている場合は内積の平均値を取る
                                vertex_mean_dots[x] = np.mean(vertex_tmp_dots[x])
                                logger.debug(f"target top: [{virtual_vertices[vcm_reverse_list[0]['vv']].vidxs()}], bottom: [{virtual_vertices[vcm_reverse_list[-1]['vv']].vidxs()}], dot: {round(vertex_mean_dots[x], 3)}")   # noqa
                            else:
                                # 1個の場合はそのまま登録
                                vertex_mean_dots[x] = 1
                    # 最も内積平均値が大きい列を登録対象とする
                    target_regists[list(vertex_mean_dots.keys())[np.argmax(list(vertex_mean_dots.values()))]] = True
            else:
                # 全部1個ずつ繋がっている場合はそのまま登録
                target_regists = [True for _ in range(len(vertex_coordinate_map))]

            logger.debug(f'target_regists: {target_regists}')

            logger.info("-- 絶対頂点マップ[%s]: マップ生成", midx + 1)

            # XYの最大と最小の抽出
            start_x = vertex_coordinate_map[0][list(vertex_coordinate_map[0].keys())[0]]['x']
            xs = [v["x"] for vcm in vertex_coordinate_map for v in vcm.values() if target_regists[v["x"] - start_x]]
            ys = [v["y"] for vcm in vertex_coordinate_map for v in vcm.values() if target_regists[v["x"] - start_x]]

            xu = np.unique(xs)
            yu = np.unique(ys)
            
            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((len(yu), len(xu), 3), (np.inf, np.inf, np.inf))
            vertex_display_map = np.full((len(yu), len(xu)), ' None ')

            vertical_break_x = 0
            for s, vcm in enumerate(vertex_coordinate_map):
                for t, vc_key in enumerate(list(vcm.keys())[:-1]):
                    if t > 0 and vcm[vc_key]['vv'] in all_vertical_edge_lines:
                        # 途中に切れ目がある場合、ずらす
                        vertical_break_x = s
                        break
                if vertical_break_x > 0:
                    break
            
            break_x = np.count_nonzero(target_regists[:(vertical_break_x + 1)])

            xx = 0
            for r, vcm in enumerate(vertex_coordinate_map):
                if not target_regists[r]:
                    continue

                vertex_tmp_dots.append([])
                vcm_reverse_list = list(reversed(list(vcm.values())))
                yy = 0
                for y, vc in enumerate(vcm_reverse_list):
                    logger.debug(f'x: {vc["x"]}, y: {vc["y"]}, vv: {vc["vv"]}, idxs: {virtual_vertices[vc["vv"]].vidxs()}')

                    vertex_map[yy, xx - break_x] = vc['vv']
                    vertex_display_map[yy, xx - break_x] = ':'.join([str(v) for v in virtual_vertices[vc["vv"]].vidxs()])
                    
                    yy += 1
                xx += 1

                logger.debug('-------')

            # 左端と右端で面が連続しているかチェック
            for yi in range(vertex_map.shape[0]):
                if tuple(vertex_map[yi, -1]) in virtual_vertices and tuple(vertex_map[yi, 0]) in virtual_vertices[tuple(vertex_map[yi, -1])].lines():
                    vertex_connecteds[-1].append(True)
                else:
                    vertex_connecteds[-1].append(False)
                
            vertex_maps.append(vertex_map)

            logger.info('\n'.join([', '.join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]), translate=False)
            logger.debug(f'vertex_connected: {vertex_connecteds[-1]}')

            logger.info("-- 絶対頂点マップ: %s個目:終了 ---------", midx + 1)

            midx += 1
            logger.debug('-----------------------')

        return vertex_maps, vertex_connecteds, virtual_vertices

    def create_vertex_coordinate_map(self, tx: int, top_edge_key: tuple, bottom_edge_key: tuple, virtual_vertices: dict, top_keys: list, bottom_keys: list, local_next_axis: MVector3D):
        vertex_coordinate_map = {}

        vertex_coordinate_map[bottom_edge_key] = {'x': tx, 'y': 0, 'vv': bottom_edge_key}

        return self.create_vertex_direction(vertex_coordinate_map, tx, -1, bottom_edge_key, top_edge_key, bottom_edge_key, virtual_vertices, \
                                            top_keys, bottom_keys, local_next_axis)

    def create_vertex_direction(self, vertex_coordinate_map: list, tx: int, ty: int, from_key: tuple, top_edge_key: tuple, bottom_edge_key: tuple, \
                                virtual_vertices: dict, top_keys: list, bottom_keys: list, local_next_axis: MVector3D):

        top_edge_vv = virtual_vertices[top_edge_key]
        bottom_edge_vv = virtual_vertices[bottom_edge_key]

        logger.debug('-----------')
        logger.debug(f'create_vertex_direction: tx: {tx}, ty: {ty}, top: {top_edge_vv.vidxs()}, bottom: {bottom_edge_vv.vidxs()}')

        # http://www.sousakuba.com/Programming/gs_near_pos_on_line.html
        # 始端と終端を繋いだ線分
        edge_line = top_edge_vv.position() - bottom_edge_vv.position()
        edge_normal_line = edge_line.normalized()

        # FROMから繋がる辺
        from_vv = virtual_vertices[from_key]
        from_edge_lines = from_vv.lines()

        if not from_edge_lines:
            return vertex_coordinate_map

        to_vv_dots = []
        to_vv_distances = []
        for tvv_key in from_edge_lines:
            if tvv_key not in bottom_keys + list(vertex_coordinate_map.keys()):
                tvv = virtual_vertices[tvv_key]

                # 終端とTO候補点を繋いだ線分
                to_line = tvv.position() - bottom_edge_vv.position()
                
                # 終端から線上最近点までの距離
                to_dot = MVector3D.dotProduct(edge_normal_line, to_line)

                # 線上最近点
                to_nearest_pos = bottom_edge_vv.position() + (edge_normal_line * to_dot)

                # 線上最近点とTO候補点との距離
                distance = tvv.position().distanceToPoint(to_nearest_pos)
                to_vv_distances.append(distance)

                # 始端から終端の線分との内積
                dot = MVector3D.dotProduct(to_line.normalized(), (to_nearest_pos - bottom_edge_vv.position()).normalized())
                to_vv_dots.append(dot)

                logger.debug(f'to_vv: vidx: [{tvv.vidxs()}], pos: [{tvv.position().to_log()}], local: [{to_nearest_pos.to_log()}], distance: [{round(distance, 5)}], dot: [{round(dot, 5)}]')    # noqa
            else:
                to_vv_dots.append(0)
                to_vv_distances.append(np.inf)

        if (np.array(to_vv_dots) == 0).all():
            return vertex_coordinate_map

        # 進行方向のキー(近似値が複数取れる可能性)
        direction_idxs = np.where(to_vv_dots >= np.max(np.fix(np.array(to_vv_dots) * 10000) / 10000))

        if len(direction_idxs) > 0:
            if len(direction_idxs[0]) == 1:
                # 最高値が1つしかない場合、それを採用
                direction_keys = np.array(from_edge_lines)[direction_idxs][0]
                direction_key = (direction_keys[0], direction_keys[1], direction_keys[2])
            else:
                # 最高値が複数ある場合、距離が小さい方を優先
                direction_keys = np.array(from_edge_lines)[np.where(to_vv_distances == np.min(np.array(to_vv_distances)[direction_idxs]))][0]
                direction_key = (direction_keys[0], direction_keys[1], direction_keys[2])
        else:
            return vertex_coordinate_map

        logger.debug(f'* direction: vidx: [{virtual_vertices[direction_key].vidxs()}]')
        vertex_coordinate_map[direction_key] = {'x': tx, 'y': ty, 'vv': direction_key}

        if direction_key in top_keys:
            # 上端に辿り着いたら終了
            return vertex_coordinate_map

        return self.create_vertex_direction(vertex_coordinate_map, tx, ty - 1, direction_key, top_edge_key, bottom_edge_key, virtual_vertices, \
                                            top_keys, bottom_keys, local_next_axis)

    # エッジを繋いでいく
    def get_edge_lines(self, edge_line_pairs: dict, virtual_vertices: dict, start_vkey: tuple, edge_lines: list, edge_vkeys: list):
        if len(edge_vkeys) >= len(edge_line_pairs.keys()):
            return start_vkey, edge_lines, edge_vkeys
        
        if not start_vkey:
            # Z方向に最も＋（最も奥）のものを選ぶ（ソートしやすいよう、Z-Yの順番に並べる）
            for vkey in list(set(edge_line_pairs.keys()) - set(edge_vkeys)):
                if not start_vkey or (virtual_vertices[start_vkey].position().z() < virtual_vertices[vkey].position().z()):
                    start_vkey = vkey
            edge_lines.append([start_vkey])
            edge_vkeys.append(start_vkey)
        
        for next_vkey in edge_line_pairs[start_vkey]:
            if next_vkey not in edge_vkeys:
                edge_lines[-1].append(next_vkey)
                edge_vkeys.append(next_vkey)
                start_vkey, edge_lines, edge_vkeys = self.get_edge_lines(edge_line_pairs, virtual_vertices, next_vkey, edge_lines, edge_vkeys)

        return None, edge_lines, edge_vkeys


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def randomname(n) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


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
