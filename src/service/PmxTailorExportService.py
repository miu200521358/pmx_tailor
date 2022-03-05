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
        # 対象頂点に対するボーン情報
        self.bone = None
        self.parent_bone = None
        self.bone_regist = False
    
    def append(self, real_vertices: list, connected_vvs: list, indexes: list, out_indexes: list):
        for rv in real_vertices:
            if rv not in self.real_vertices:
                self.real_vertices.append(rv)
                self.positions.append(rv.position.data())
                if len(indexes) > 0:
                    # 対象面である場合のみ法線保持
                    self.normals.append(rv.normal.data())
        
        for lv in connected_vvs:
            if lv not in self.connected_vvs:
                self.connected_vvs.append(lv)
        
        for i in indexes:
            if i not in self.indexes:
                self.indexes.append(i)

        for i in out_indexes:
            if i not in self.out_indexes:
                self.out_indexes.append(i)

    def vidxs(self):
        return [v.index for v in self.real_vertices]
    
    def position(self):
        return MVector3D(np.mean(self.positions, axis=0))
    
    def normal(self):
        return MVector3D(np.mean(self.normals, axis=0))

    def __str__(self):
        return f"v[{','.join([str(v.index) for v in self.real_vertices])}] pos[{self.position().to_log()}] nor[{self.normal().to_log()}], lines[{self.connected_vvs}], indexes[{self.indexes}], out_indexes[{self.out_indexes}]"    # noqa


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
            model.comment += f"\r\n\r\n{logger.transtext('物理')}: PmxTailor"

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

    def get_saved_bone_names(self, model: PmxModel):
        # TODO
        return []

    def create_physics(self, model: PmxModel, param_option: dict, saved_bone_names: list):
        model.comment += f"\r\n{logger.transtext('材質')}: {param_option['material_name']} --------------"    # noqa
        model.comment += f"\r\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"    # noqa
        model.comment += f", {logger.transtext('細かさ')}: {param_option['fineness']}"    # noqa
        model.comment += f", {logger.transtext('質量')}: {param_option['mass']}"    # noqa
        model.comment += f", {logger.transtext('柔らかさ')}: {param_option['air_resistance']}"    # noqa
        model.comment += f", {logger.transtext('張り')}: {param_option['shape_maintenance']}"    # noqa

        material_name = param_option['material_name']

        logger.info("%s: 対象頂点リストの生成", material_name)
        
        # 頂点CSVが指定されている場合、対象頂点リスト生成
        if param_option['vertices_csv']:
            target_vertices = []
            try:
                with open(param_option['vertices_csv'], encoding='cp932', mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)            # ヘッダーを読み飛ばす
                    for row in reader:
                        if len(row) > 1 and int(row[1]) in model.material_vertices[material_name]:
                            target_vertices.append(int(row[1]))
            except Exception:
                logger.warning("頂点CSVが正常に読み込めなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
                return None, None
        else:
            target_vertices = list(model.material_vertices[material_name])

        if param_option['exist_physics_clear'] == logger.transtext('再利用'):
            # TODO
            pass
        else:
            logger.info("【%s】頂点マップ生成", material_name, decoration=MLogger.DECORATION_LINE)

            vertex_maps, virtual_vertices \
                = self.create_vertex_map(model, param_option, material_name, target_vertices)
            
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
            
            root_bone, virtual_vertices, all_regist_bones, all_bone_vertical_distances, all_bone_horizonal_distances, all_bone_connected \
                = self.create_bone(model, param_option, material_name, virtual_vertices, vertex_maps, vertex_map_orders)
            
            self.create_weight(model, param_option, material_name, virtual_vertices, vertex_maps, \
                               all_regist_bones, all_bone_vertical_distances, all_bone_horizonal_distances, all_bone_connected)

        return True
    
    def create_weight(self, model: PmxModel, param_option: dict, material_name: str, virtual_vertices: dict, vertex_maps: dict, \
                      all_regist_bones: dict, all_bone_vertical_distances: dict, all_bone_horizonal_distances: dict, all_bone_connected: dict):
        logger.info("【%s】ウェイト生成", material_name, decoration=MLogger.DECORATION_LINE)

        for base_map_idx, regist_bones in all_regist_bones.items():
            logger.info("--【No.%s】ウェイト分布判定", base_map_idx + 1)

            # ウェイト分布
            prev_weight_cnt = 0
            weight_cnt = 0

            for v_yidx in range(regist_bones.shape[0]):
                for v_xidx in range(regist_bones.shape[1]):
                    vv = virtual_vertices[tuple(vertex_maps[base_map_idx][v_yidx, v_xidx])]

                    if regist_bones[v_yidx, v_xidx]:
                        if v_yidx < regist_bones.shape[0] - 1:
                            target_v_yidx = v_yidx
                        else:
                            # Y末端は登録対象外なので、ひとつ上のをそのまま割り当てる
                            target_v_yidx = v_yidx if v_yidx < regist_bones.shape[0] - 1 else np.max(np.where(regist_bones[:v_yidx, :]), axis=1)[0]

                        # 頂点位置にボーンが登録されている場合、BDEF1登録対象
                        for rv in vv.real_vertices:
                            rv.deform = Bdef1(virtual_vertices[tuple(vertex_maps[base_map_idx][target_v_yidx, v_xidx])].bone.index)
                    elif regist_bones[v_yidx, :].any():
                        # 同じY位置にボーンがある場合、横のBDEF2登録対象
                        # 末端ボーンにはウェイトを割り当てない
                        target_v_yidx = v_yidx if v_yidx < regist_bones.shape[0] - 1 else np.max(
                            np.where(regist_bones[:v_yidx, :]), axis=1)[0]
                        prev_xidx = np.max(
                            np.where(regist_bones[target_v_yidx, :(v_xidx + 1)]))
                        if v_xidx < regist_bones.shape[1] - 1 and regist_bones[target_v_yidx, (v_xidx + 1):].any():
                            next_xidx = v_xidx + 1 + \
                                np.min(
                                    np.where(regist_bones[target_v_yidx, (v_xidx + 1):]))
                            regist_next_xidx = next_xidx
                        else:
                            next_xidx = v_xidx + 1
                            regist_next_xidx = 0

                        prev_weight = (np.sum(all_bone_horizonal_distances[base_map_idx][target_v_yidx, prev_xidx:v_xidx]) / \
                                       np.sum(all_bone_horizonal_distances[base_map_idx][target_v_yidx, prev_xidx:next_xidx]))

                        for rv in vv.real_vertices:
                            rv.deform = Bdef2(virtual_vertices[tuple(vertex_maps[base_map_idx][target_v_yidx, prev_xidx])].bone.index,
                                              virtual_vertices[tuple(
                                                  vertex_maps[base_map_idx][target_v_yidx, regist_next_xidx])].bone.index, 1 - prev_weight)

                    elif regist_bones[:, v_xidx].any():
                        # 同じX位置にボーンがある場合、縦のBDEF2登録対象
                        # Y末端は登録対象外
                        above_yidx = np.max(
                            np.where(regist_bones[:v_yidx, v_xidx]))
                        below_yidx = np.min(
                            np.where(regist_bones[v_yidx:, v_xidx])) + v_yidx

                        if below_yidx == regist_bones.shape[0] - 1:
                            # 末端がある場合、上のボーンでBDEF1
                            for rv in vv.real_vertices:
                                rv.deform = Bdef1(virtual_vertices[tuple(
                                    vertex_maps[base_map_idx][above_yidx, v_xidx])].bone.index)
                        else:
                            above_weight = (np.sum(all_bone_vertical_distances[base_map_idx][(above_yidx + 1):(v_yidx + 1), (v_xidx - 1)]) / \
                                            np.sum(all_bone_vertical_distances[base_map_idx][(above_yidx + 1):(below_yidx + 1), (v_xidx - 1)]))

                            for rv in vv.real_vertices:
                                rv.deform = Bdef2(virtual_vertices[tuple(vertex_maps[base_map_idx][above_yidx, v_xidx])].bone.index,
                                                  virtual_vertices[tuple(
                                                      vertex_maps[base_map_idx][below_yidx, v_xidx])].bone.index, 1 - above_weight)
                    else:
                        if regist_bones[:v_yidx, :v_xidx].shape == (1, 1):
                            prev_xidx = np.where(
                                regist_bones[:v_yidx, :v_xidx])[1][0]
                            above_yidx = np.where(
                                regist_bones[:v_yidx, :v_xidx])[0][0]
                        else:
                            prev_xidx = np.max(
                                np.where(regist_bones[:v_yidx, :v_xidx]), axis=1)[1]
                            above_yidx = np.max(
                                np.where(regist_bones[:v_yidx, :v_xidx])[0])

                        below_yidx = np.min(
                            np.max(np.where(regist_bones[v_yidx:, :v_xidx]), axis=0)) + v_yidx

                        if regist_bones[:v_yidx, v_xidx:].any():
                            target_next_xidx = next_xidx = np.min(
                                np.max(np.where(regist_bones[:v_yidx, v_xidx:]), axis=0)) + v_xidx
                        else:
                            # 最後の頂点の場合、とりあえず次の距離を対象とする
                            next_xidx = v_xidx + 1
                            target_next_xidx = 0

                        prev_above_weight = (np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, (v_xidx - 1)]) / \
                                             np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)])) \
                            * (np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx]) / \
                               np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx]))

                        next_above_weight = (np.sum(all_bone_vertical_distances[base_map_idx][v_yidx:below_yidx, (v_xidx - 1)]) / \
                                             np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)])) \
                            * (np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx]) / \
                               np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx]))

                        prev_below_weight = (np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, (v_xidx - 1)]) / \
                                             np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)])) \
                            * (np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, v_xidx:next_xidx]) / \
                               np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx]))

                        next_below_weight = (np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:v_yidx, (v_xidx - 1)]) / \
                                             np.sum(all_bone_vertical_distances[base_map_idx][above_yidx:below_yidx, (v_xidx - 1)])) \
                            * (np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:v_xidx]) / \
                               np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, prev_xidx:next_xidx]))

                        if below_yidx == regist_bones.shape[0] - 1:
                            prev_above_weight += prev_below_weight
                            next_above_weight += next_below_weight

                            # ほぼ0のものは0に置換（円周用）
                            total_weights = np.array(
                                [prev_above_weight, next_above_weight])
                            total_weights[np.isclose(
                                total_weights, 0, equal_nan=True)] = 0

                            if np.count_nonzero(total_weights):
                                deform_weights = total_weights / \
                                    total_weights.sum(axis=0, keepdims=1)

                                for rv in vv.real_vertices:
                                    rv.deform = Bdef2(virtual_vertices[tuple(vertex_maps[base_map_idx][above_yidx, prev_xidx])].bone.index,
                                                      virtual_vertices[tuple(
                                                          vertex_maps[base_map_idx][above_yidx, target_next_xidx])].bone.index,
                                                      deform_weights[0])
                        else:
                            # ほぼ0のものは0に置換（円周用）
                            total_weights = np.array(
                                [prev_above_weight, next_above_weight, prev_below_weight, next_below_weight])
                            total_weights[np.isclose(
                                total_weights, 0, equal_nan=True)] = 0

                            if np.count_nonzero(total_weights):
                                deform_weights = total_weights / \
                                    total_weights.sum(axis=0, keepdims=1)

                                for rv in vv.real_vertices:
                                    rv.deform = Bdef4(virtual_vertices[tuple(vertex_maps[base_map_idx][above_yidx, prev_xidx])].bone.index,
                                                      virtual_vertices[tuple(
                                                          vertex_maps[base_map_idx][above_yidx, target_next_xidx])].bone.index,
                                                      virtual_vertices[tuple(
                                                          vertex_maps[base_map_idx][below_yidx, prev_xidx])].bone.index,
                                                      virtual_vertices[tuple(
                                                          vertex_maps[base_map_idx][below_yidx, target_next_xidx])].bone.index,
                                                      deform_weights[0], deform_weights[1], deform_weights[2], deform_weights[3])

                    weight_cnt += len(vv.real_vertices)
                    if weight_cnt > 0 and weight_cnt // 1000 > prev_weight_cnt:
                        logger.info("-- --【No.%s】頂点ウェイト: %s個目:終了", base_map_idx + 1, weight_cnt)
                        prev_weight_cnt = weight_cnt // 1000

    def create_bone(self, model: PmxModel, param_option: dict, material_name: str, virtual_vertices: dict, vertex_maps: dict, vertex_map_orders: dict):
        logger.info("【%s】ボーン生成", material_name, decoration=MLogger.DECORATION_LINE)
            
        # 中心ボーン生成

        # 略称
        abb_name = param_option['abb_name']
        # 表示枠名
        display_name = f"{abb_name}:{material_name}"
        # 親ボーン
        parent_bone = model.bones[param_option['parent_bone_name']]

        # 表示枠定義
        model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

        # 中心ボーン
        root_bone = Bone(f'{abb_name}中心', f'{abb_name}Root', parent_bone.position, \
                         parent_bone.index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
        root_bone.index = len(model.bones)
        model.bones[root_bone.name] = root_bone
        model.bone_indexes[root_bone.index] = root_bone.name

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
                    if (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx, v_xidx + 1] != np.inf).all():
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])].position()
                        bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)
                        
                        if tuple(vertex_map[v_yidx, v_xidx]) in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])].connected_vvs:
                            # 前の仮想頂点と繋がっている場合、True
                            bone_connected[v_yidx, v_xidx] = True

                    if v_yidx < vertex_map.shape[0] - 1 and (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx + 1, v_xidx] != np.inf).all():
                        now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                        next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx + 1, v_xidx])].position()
                        bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)

                    vertex_cnt += 1
                    if vertex_cnt > 0 and vertex_cnt // 1000 > prev_vertex_cnt:
                        logger.info("-- --【No.%s】頂点距離算出: %s個目:終了", base_map_idx + 1, vertex_cnt)
                        prev_vertex_cnt = vertex_cnt // 1000

                v_xidx += 1
                if (vertex_map[v_yidx, v_xidx] != np.inf).all() and (vertex_map[v_yidx, 0] != np.inf).all():
                    # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
                    if tuple(vertex_map[v_yidx, v_xidx]) in virtual_vertices[tuple(vertex_map[v_yidx, 0])].connected_vvs:
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

        # 全体通してのX番号
        prev_xs = [0]
        all_regist_bones = {}
        for base_map_idx in vertex_map_orders:
            logger.info("--【No.%s】ボーン生成", base_map_idx + 1)

            prev_bone_cnt = 0
            bone_cnt = 0

            vertex_map = vertex_maps[base_map_idx]

            # ボーン登録有無
            regist_bones = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)
            all_regist_bones[base_map_idx] = regist_bones

            if param_option["density_type"] == logger.transtext('距離'):
                median_vertical_distance = np.median(all_bone_vertical_distances[base_map_idx][:, int(vertex_map.shape[1] / 2)])
                median_horizonal_distance = np.median(all_bone_horizonal_distances[base_map_idx][int(vertex_map.shape[0] / 2), :])

                logger.debug(f'median_horizonal_distance: {round(median_horizonal_distance, 4)}, median_vertical_distance: {round(median_vertical_distance, 4)}')

                # 間隔が距離タイプの場合、均等になるように間を空ける
                y_regists = np.zeros(vertex_map.shape[0], dtype=np.int)
                prev_y_regist = 0
                for v_yidx in range(vertex_map.shape[0]):
                    if v_yidx in [0, vertex_map.shape[0] - 1]:
                        # 最初は必ず登録
                        y_regists[v_yidx] = True
                        continue
                    
                    if np.sum(all_bone_vertical_distances[base_map_idx][(prev_y_regist + 1):(v_yidx + 1), int(vertex_map.shape[1] / 2)]) > \
                       median_vertical_distance * param_option["vertical_bone_density"]:
                        # 前の登録ボーンから一定距離離れたら登録対象
                        y_regists[v_yidx] = True
                        prev_y_regist = v_yidx

                x_regists = np.zeros(vertex_map.shape[1], dtype=np.int)
                prev_x_regist = 0
                for v_xidx in range(vertex_map.shape[1]):
                    if v_xidx in [0, vertex_map.shape[1] - 1]:
                        # 最初は必ず登録
                        x_regists[v_xidx] = True
                        continue
                    
                    if np.sum(all_bone_horizonal_distances[base_map_idx][int(vertex_map.shape[0] / 2), (prev_x_regist + 1):(v_xidx + 1)]) > \
                       median_horizonal_distance * param_option["horizonal_bone_density"]:
                        # 前の登録ボーンから一定距離離れたら登録対象
                        x_regists[v_xidx] = True
                        prev_x_regist = v_xidx

                for v_yidx, y_regist in enumerate(y_regists):
                    for v_xidx, x_regist in enumerate(x_regists):
                        regist_bones[v_yidx, v_xidx] = (y_regist and x_regist)

            else:
                # 間隔が頂点タイプの場合、規則的に間を空ける
                for v_yidx in list(range(0, vertex_map.shape[0], param_option["vertical_bone_density"])) + [vertex_map.shape[0] - 1]:
                    for v_xidx in range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]):
                        regist_bones[v_yidx, v_xidx] = True
                    if not all_bone_connected[base_map_idx][v_yidx, vertex_map.shape[1] - 1]:
                        # 繋がってない場合、最後に追加する
                        regist_bones[v_yidx, vertex_map.shape[1] - 1] = True

            for v_yidx in range(vertex_map.shape[0]):
                for v_xidx in range(vertex_map.shape[1]):
                    if (vertex_map[v_yidx, v_xidx] == np.inf).any():
                        continue

                    v_yno = v_yidx + 1
                    v_xno = v_xidx + max(prev_xs) + 1

                    vkey = tuple(vertex_map[v_yidx, v_xidx])
                    vv = virtual_vertices[vkey]

                    # 親は既にモデルに登録済みのものを選ぶ
                    parent_bone = None
                    for parent_v_yidx in range(v_yidx - 1, -1, -1):
                        parent_bone = virtual_vertices[tuple(vertex_map[parent_v_yidx, v_xidx])].bone
                        if parent_bone and parent_bone.name in model.bones:
                            # 登録されていたら終了
                            break
                        else:
                            parent_bone = None
                    if not parent_bone:
                        # 最後まで登録されている親ボーンが見つからなければ、ルート
                        parent_bone = root_bone

                    # ひとつ前も既にモデルに登録済みのものを選ぶ
                    prev_bone = None
                    for prev_v_xidx in range(v_xidx - 1, -1, -1):
                        if (vertex_map[v_yidx, prev_v_xidx] == np.inf).any():
                            continue

                        prev_bone = virtual_vertices[tuple(vertex_map[v_yidx, prev_v_xidx])].bone
                        if prev_bone and prev_bone.name in model.bones:
                            # 登録されていたら終了
                            break

                    if not vv.bone:
                        # ボーン仮登録
                        bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                        bone = Bone(bone_name, bone_name, vv.position().copy(), parent_bone.index, 0, 0x0000 | 0x0002)
                        bone.index = len(model.bones)
                        bone.local_z_vector = vv.normal().copy()

                        bone.parent_index = parent_bone.index
                        bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                        bone.local_z_vector *= MVector3D(-1, 1, -1)
                        bone.flag |= 0x0800

                        if v_yidx > 0:
                            # 親ボーンの表示先も同時設定
                            if parent_bone != root_bone:
                                parent_bone.tail_index = bone.index
                                parent_bone.local_x_vector = (bone.position - parent_bone.position).normalized()
                                parent_bone.flag |= 0x0001

                            # 表示枠
                            parent_bone.flag |= 0x0008 | 0x0010
                            model.display_slots[display_name].references.append((0, parent_bone.index))

                        vv.bone = bone

                        if regist_bones[v_yidx, v_xidx]:
                            # 登録対象である場合
                            model.bones[bone.name] = bone
                            model.bone_indexes[bone.index] = bone.name

                        logger.debug(f"tmp_all_bones: {bone.name}, pos: {bone.position.to_log()}")

                        bone_cnt += 1
                        if bone_cnt > 0 and bone_cnt // 1000 > prev_bone_cnt:
                            logger.info("-- --【No.%s】ボーン生成: %s個目:終了", base_map_idx + 1, bone_cnt)
                            prev_bone_cnt = bone_cnt // 1000

            prev_xs.extend(max(prev_xs) + np.array(list(range(vertex_map.shape[1]))) + 1)

        return root_bone, virtual_vertices, all_regist_bones, all_bone_vertical_distances, all_bone_horizonal_distances, all_bone_connected

    def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
        return f'{abb_name}-{(v_yno):03d}-{(v_xno):03d}'

    def create_vertex_map(self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list):
        # 閾値（元の検出度ベースで求め直す）
        threshold = param_option['similarity'] / 7.5

        logger.info("%s: 厚み閾値[%s]", material_name, threshold)

        # 方向に応じて判定値を変える
        # デフォルトは下
        base_vertical_axis = MVector3D(0, -1, 0)
        if param_option['direction'] == '上':
            base_vertical_axis = MVector3D(0, 1, 0)
        elif param_option['direction'] == '右':
            base_vertical_axis = MVector3D(-1, 0, 0)
        elif param_option['direction'] == '左':
            base_vertical_axis = MVector3D(1, 0, 0)
        
        logger.info("%s: 仮想頂点リストの生成", material_name)

        parent_bone = model.bones[param_option['parent_bone_name']]

        virtual_vertices = {}
        edge_pair_lkeys = {}
        for index_idx in model.material_indices[material_name]:
            # 頂点の組み合わせから面INDEXを引く
            if model.indices[index_idx][0] not in target_vertices or model.indices[index_idx][1] not in target_vertices or model.indices[index_idx][2] not in target_vertices:
                # 3つ揃ってない場合、スルー
                continue

            for v0_idx, v1_idx, v2_idx in zip(model.indices[index_idx], model.indices[index_idx][1:] + [model.indices[index_idx][0]], [model.indices[index_idx][2]] + model.indices[index_idx][:-1]):
                v0 = model.vertex_dict[v0_idx]
                v1 = model.vertex_dict[v1_idx]
                v2 = model.vertex_dict[v2_idx]

                v0_key = v0.position.to_key(threshold)
                v1_key = v1.position.to_key(threshold)
                v2_key = v2.position.to_key(threshold)

                if v0_key not in virtual_vertices:
                    virtual_vertices[v0_key] = VirtualVertex(v0_key)
                
                # 一旦ルートボーンにウェイトを一括置換
                v0.deform = Bdef1(parent_bone.index)

                # 面垂線（Yは潰す）
                vv1 = (v1.position - v0.position).normalized()
                vv2 = (v2.position - v1.position).normalized()
                surface_normal = MVector3D.crossProduct(vv1, vv2).normalized()
                surface_normal.setY(0)

                # 親ボーンに対する向き（Yは潰す）
                parent_direction = (v0.position - parent_bone.position)
                parent_direction.setY(0)
                parent_direction.normalize()

                # 親ボーンの向きとの内積
                normal_dot = MVector3D.dotProduct(surface_normal, parent_direction)
                logger.debug(f'index[{index_idx}], v0[{v0.index}:{v0_key}], sn[{surface_normal.to_log()}], pd[{parent_direction.to_log()}], dot[{round(normal_dot, 5)}]')

                # 面法線と同じ向き場合、辺キー生成（表面のみを対象とする）
                if normal_dot > 0.1:
                    lkey = (min(v0_key, v1_key), max(v0_key, v1_key))
                    if lkey not in edge_pair_lkeys:
                        edge_pair_lkeys[lkey] = []
                    if index_idx not in edge_pair_lkeys[lkey]:
                        edge_pair_lkeys[lkey].append(index_idx)

                    # 仮想頂点登録（該当頂点対象）
                    virtual_vertices[v0_key].append([v0], [v1_key, v2_key], [index_idx], [])
                else:
                    # 仮想頂点登録（該当頂点非対象）
                    virtual_vertices[v0_key].append([v0], [v1_key, v2_key], [], [index_idx])

        if not virtual_vertices:
            logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return None, None

        if not edge_pair_lkeys:
            logger.warning("対象範囲にエッジが見つけられなかった為、処理を終了します。\n面が表裏反転してないかご確認ください。", decoration=MLogger.DECORATION_BOX)
            return None, None

        if logger.is_debug_level():
            logger.debug('--------------------------')
            for key, virtual_vertex in virtual_vertices.items():
                logger.debug(f'[{key}] {virtual_vertex}')

            logger.debug('--------------------------')
            for (min_key, max_key), indexes in edge_pair_lkeys.items():
                logger.debug(f'[{min_key}:{virtual_vertices[min_key].vidxs()}, {max_key}:{virtual_vertices[max_key].vidxs()}] {indexes}')

        edge_line_pairs = {}
        for (min_vkey, max_vkey), line_iidxs in edge_pair_lkeys.items():
            if len(line_iidxs) == 1:
                if min_vkey not in edge_line_pairs:
                    edge_line_pairs[min_vkey] = []
                if max_vkey not in edge_line_pairs:
                    edge_line_pairs[max_vkey] = []
                
                edge_line_pairs[min_vkey].append(max_vkey)
                edge_line_pairs[max_vkey].append(min_vkey)

        logger.info("%s: エッジの抽出準備", material_name)

        # エッジを繋いでいく
        all_edge_lines = []
        edge_vkeys = []
        while len(edge_vkeys) < len(edge_line_pairs.keys()):
            _, all_edge_lines, edge_vkeys = self.get_edge_lines(edge_line_pairs, None, all_edge_lines, edge_vkeys)

        if logger.is_debug_level():
            for n, edge_lines in enumerate(all_edge_lines):
                logger.debug(f'[{n:02d}] {[f"{ekey}:{virtual_vertices[ekey].vidxs()}" for ekey in edge_lines]}')

        logger.info("%s: エッジの抽出", material_name)

        horizonal_edge_lines = []
        vertical_edge_lines = []
        for n, edge_lines in enumerate(all_edge_lines):
            if 1 < len(all_edge_lines):
                horizonal_edge_lines.append([])
                vertical_edge_lines.append([])

            edge_dots = []
            direction_dots = []
            for prev_edge_key, now_edge_key, next_edge_key in zip(list(edge_lines[-1:]) + list(edge_lines[:-1]), edge_lines, list(edge_lines[1:]) + list(edge_lines[:1])):
                prev_edge_pos = virtual_vertices[prev_edge_key].position()
                now_edge_pos = virtual_vertices[now_edge_key].position()
                next_edge_pos = virtual_vertices[next_edge_key].position()

                edge_dots.append(MVector3D.dotProduct((now_edge_pos - prev_edge_pos).normalized(), (next_edge_pos - now_edge_pos).normalized()))
                direction_dots.append(MVector3D.dotProduct((now_edge_pos - prev_edge_pos).normalized(), base_vertical_axis))

            # 方向の中央値
            # direction_dot_mean = np.mean([np.min(np.abs(direction_dots)), np.mean(np.abs(direction_dots))])
            direction_dot_mean = np.mean(np.abs(direction_dots))
            # 最大の内積差
            edge_dot_mean = np.mean(np.abs(edge_dots))
            edge_dot_diff_max = np.max(np.abs(np.abs(edge_dots) - edge_dot_mean))

            logger.debug(f'[{n:02d}] direction[{np.round(direction_dot_mean, 4)}], dot[{np.round(direction_dots, 4)}], edge_dot_diff_max[{round(edge_dot_diff_max, 4)}]')
            logger.debug(f'[{n:02d}] mean[{np.round(edge_dot_mean, 4)}], dot[{np.round(edge_dots, 4)}] diff[{np.round(np.array(edge_dots) - edge_dot_mean, 4)}] diffmax[{np.round(edge_dot_diff_max, 4)}]')     # noqa

            if edge_dot_diff_max > 0.5:
                # 内積差が大きい場合、エッジが分断されてるとみなす
                logger.debug(f'[{n:02d}] corner[{np.where(np.array(edge_dots) < 0.5)[0].tolist()}]')
                slice_idxs = np.where(np.array(edge_dots) < 0.5)[0].tolist()
                slice_idxs += [slice_idxs[0]]
                # is_prev_horizonal = True
                for ssi, esi in zip(slice_idxs, slice_idxs[1:]):
                    target_edge_lines = edge_lines[ssi:(esi + 1)] if 0 <= ssi < esi else edge_lines[ssi:] + edge_lines[:(esi + 1)]
                    target_direction_dots = direction_dots[(ssi + 1):(esi + 1)] if 0 <= ssi < esi else direction_dots[(ssi + 1):] + direction_dots[:(esi + 1)]
                    
                    if np.round(np.mean(np.abs(target_direction_dots)), 3) <= np.round(direction_dot_mean, 3):
                        # 同一方向の傾きがdirectionと垂直っぽければ、水平方向
                        if 1 == len(all_edge_lines):
                            horizonal_edge_lines.append([])
                        horizonal_edge_lines[-1].append(target_edge_lines)
                        # is_prev_horizonal = True
                    else:
                        # 同一方向の傾きがdirectionと同じっぽければ、垂直方向

                        # # 垂直が2回続いている場合、スリットとみなして、切替の一点を水平に入れておく
                        # if not is_prev_horizonal:
                        #     if 1 == len(all_edge_lines):
                        #         horizonal_edge_lines.append([])
                        #     horizonal_edge_lines[-1].append([target_edge_lines[0]])

                        if 1 == len(all_edge_lines):
                            vertical_edge_lines.append([])
                        vertical_edge_lines[-1].append(target_edge_lines)
                        # is_prev_horizonal = False
                        
            else:
                # 内積差が小さい場合、エッジが均一に水平に繋がってるとみなす(一枚物は有り得ない)
                horizonal_edge_lines[-1].append(edge_lines)
    
        logger.debug(f'horizonal[{horizonal_edge_lines}]')
        logger.debug(f'vertical[{vertical_edge_lines}]')

        logger.info("%s: 水平エッジの上下判定", material_name)

        # 親ボーンとの距離
        horizonal_distances = []
        for edge_lines in horizonal_edge_lines:
            line_horizonal_distances = []
            for edge_line in edge_lines:
                horizonal_poses = []
                for edge_key in edge_line:
                    horizonal_poses.append(virtual_vertices[edge_key].position().data())
                line_horizonal_distances.append(np.mean(np.linalg.norm(np.array(horizonal_poses) - parent_bone.position.data(), ord=2, axis=1), axis=0))
            horizonal_distances.append(np.mean(line_horizonal_distances))

        # 水平方向を上下に分ける
        horizonal_total_mean_distance = np.mean(horizonal_distances)
        logger.debug(f'distance[{horizonal_total_mean_distance}], [{horizonal_distances}]')

        bottom_horizonal_edge_lines = []
        top_horizonal_edge_lines = []
        for n, (hd, hel) in enumerate(zip(horizonal_distances, horizonal_edge_lines)):
            if hd > horizonal_total_mean_distance:
                # 遠い方が下(BOTTOM)
                # 一枚物は反転
                if 1 == len(all_edge_lines):
                    bottom_horizonal_edge_lines.append([])
                    for he in hel:
                        bottom_horizonal_edge_lines[-1].insert(0, list(reversed(he)))
                else:
                    bottom_horizonal_edge_lines.append(hel)
                logger.debug(f'[{n:02d}-horizonal-bottom] {hel}')
            else:
                # 近い方が上(TOP)
                top_horizonal_edge_lines.append(hel)
                logger.debug(f'[{n:02d}-horizonal-top] {hel}')

        if not top_horizonal_edge_lines:
            logger.warning("物理方向に対して水平な上部エッジが見つけられなかった為、処理を終了します。", decoration=MLogger.DECORATION_BOX)
            return None, None

        top_keys = []
        top_degrees = {}
        top_edge_poses = []
        for ti, thel in enumerate(top_horizonal_edge_lines):
            for hi, the in enumerate(thel):
                for ei, thkey in enumerate(the):
                    top_edge_poses.append(virtual_vertices[thkey].position().data())

        top_edge_mean_pos = MVector3D(np.mean(top_edge_poses, axis=0))
        # 真後ろに最も近い位置
        top_edge_start_pos = MVector3D(list(sorted(top_edge_poses, key=lambda x: (abs(x[0]), -x[2], -x[1])))[0])

        for ti, thel in enumerate(top_horizonal_edge_lines):
            for hi, the in enumerate(thel):
                top_keys.extend(the)
                for ei, thkey in enumerate(the):
                    top_degrees[thkey] = self.calc_arc_degree(top_edge_start_pos, top_edge_mean_pos, virtual_vertices[thkey].position(), base_vertical_axis)
                    logger.info("%s: 水平エッジ上部(%s-%s-%s): %s -> %s", material_name, ti + 1, hi + 1, ei + 1, virtual_vertices[thkey].vidxs(), round(top_degrees[thkey], 3))

        if not bottom_horizonal_edge_lines:
            logger.warning("物理方向に対して水平な下部エッジが見つけられなかった為、処理を終了します。", decoration=MLogger.DECORATION_BOX)
            return None, None

        logger.info('--------------')
        bottom_keys = []
        bottom_degrees = {}
        bottom_edge_poses = []
        for bi, bhel in enumerate(bottom_horizonal_edge_lines):
            for hi, bhe in enumerate(bhel):
                for ei, bhkey in enumerate(bhe):
                    bottom_edge_poses.append(virtual_vertices[bhkey].position().data())

        bottom_edge_mean_pos = MVector3D(np.mean(bottom_edge_poses, axis=0))
        bottom_edge_start_pos = MVector3D(list(sorted(bottom_edge_poses, key=lambda x: (abs(x[0]), -x[2], -x[1])))[0])

        for bi, bhel in enumerate(bottom_horizonal_edge_lines):
            for hi, bhe in enumerate(bhel):
                bottom_keys.extend(bhe)
                for ei, bhkey in enumerate(bhe):
                    bottom_degrees[bhkey] = self.calc_arc_degree(bottom_edge_start_pos, bottom_edge_mean_pos, virtual_vertices[bhkey].position(), base_vertical_axis)
                    logger.info("%s: 水平エッジ下部(%s-%s-%s): %s -> %s", material_name, bi + 1, hi + 1, ei + 1, virtual_vertices[bhkey].vidxs(), round(bottom_degrees[bhkey], 3))

        logger.info('--------------------------')
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
                    if len(top_degrees) == len(bottom_degrees):
                        # 同じ列数の場合、そのまま適用
                        top_edge_key = list(top_degrees.keys())[ki]
                    else:
                        bottom_degree = bottom_degrees[bottom_edge_key]
                        # # 途中の切れ目である場合、前後の中間角度で見る
                        # bottom_idx = [i for i, k in enumerate(bottom_degrees.keys()) if k == bottom_edge_key][0]
                        # if ki == len(bhe) - 1 and ((len(bhel) > 0 and hi < len(bhel) - 1) or \
                        #    (len(bottom_horizonal_edge_lines) > 0 and hi < len(bottom_horizonal_edge_lines) - 1)):
                        #     # 末端の場合、次との中間角度
                        #     bottom_degree = np.mean([bottom_degrees[bottom_edge_key], bottom_degrees[list(bottom_degrees.keys())[bottom_idx + 1]]])
                        # elif ki == 0 and ((len(bhel) > 0 and hi > 0) or (len(bottom_horizonal_edge_lines) > 0 and bi > 0)):
                        #     # 開始の場合、前との中間角度
                        #     bottom_degree = np.mean([bottom_degrees[bottom_edge_key], bottom_degrees[list(bottom_degrees.keys())[bottom_idx - 1]]])

                        # 近いdegreeのものを選ぶ
                        top_idx = np.argmin(np.abs(np.array(list(top_degrees.values())) - bottom_degree))
                        top_edge_key = list(top_degrees.keys())[top_idx]
                    logger.debug(f'** start: ({bi:02d}-{hi:02d}), top[{top_edge_key}({virtual_vertices[top_edge_key].vidxs()})][{round(top_degrees[top_edge_key], 3)}], bottom[{bottom_edge_key}({virtual_vertices[bottom_edge_key].vidxs()})][{round(bottom_degrees[bottom_edge_key], 3)}]')   # noqa

                    vkeys, vscores = self.create_vertex_line_map(top_edge_key, bottom_edge_key, bottom_edge_key, virtual_vertices, \
                                                                 top_keys, bottom_keys, base_vertical_axis, [bottom_edge_key], [1])
                    logger.info('頂点ルート走査[%s-%s-%s]: 終端: %s -> 始端: %s, スコア: %s', f'{(bi + 1):04d}', f'{(hi + 1):02d}', f'{(ki + 1):02d}', \
                                virtual_vertices[vkeys[-1]].vidxs(), virtual_vertices[vkeys[0]].vidxs() if vkeys else 'NG', round(np.sum(vscores), 4) if vscores else '-')
                    if vkeys:
                        all_vkeys_list[-1].append(vkeys)
                        all_scores[-1].append(vscores)

        logger.info("%s: 絶対頂点マップの生成", material_name)
        vertex_maps = []

        midx = 0
        for li, (vkeys_list, scores) in enumerate(zip(all_vkeys_list, all_scores)):
            logger.info("-- 絶対頂点マップ: %s個目: ---------", midx + 1)

            # top_keys = []
            # line_dots = []

            logger.info("-- 絶対頂点マップ[%s]: 頂点ルート決定", midx + 1)

            # top_vv = virtual_vertices[vkeys[0]]
            # bottom_vv = virtual_vertices[vkeys[-1]]
            # top_pos = top_vv.position()
            # bottom_pos = bottom_vv.position()

            # for vkeys in vkeys_list:
            #     line_dots.append([])

            #     for y, vkey in enumerate(vkeys):
            #         if y == 0:
            #             line_dots[-1].append(1)
            #             top_keys.append(vkey)
            #         elif y <= 1:
            #             continue
            #         else:
            #             prev_prev_vv = virtual_vertices[vkeys[y - 2]]
            #             prev_vv = virtual_vertices[vkeys[y - 1]]
            #             now_vv = virtual_vertices[vkey]
            #             prev_prev_pos = prev_prev_vv.position()
            #             prev_pos = prev_vv.position()
            #             now_pos = now_vv.position()
            #             prev_direction = (prev_pos - prev_prev_pos).normalized()
            #             now_direction = (now_pos - prev_pos).normalized()

            #             dot = MVector3D.dotProduct(now_direction, prev_direction)   # * now_pos.distanceToPoint(prev_pos)   # * MVector3D.dotProduct(now_direction, total_direction)   #
            #             line_dots[-1].append(dot)

            #             logger.debug(f"target top: [{virtual_vertices[vkeys[0]].vidxs()}], bottom: [{virtual_vertices[vkeys[-1]].vidxs()}], dot({y}): {round(dot, 5)}")   # noqa

            #     logger.info("-- 絶対頂点マップ[%s]: 頂点ルート確認[%s] 始端: %s, 終端: %s, 近似値: %s", midx + 1, len(top_keys), top_vv.vidxs(), bottom_vv.vidxs(), round(np.mean(line_dots[-1]), 4))

            logger.debug('------------------')
            top_key_cnts = dict(Counter([vkeys[0] for vkeys in vkeys_list]))
            target_regists = [False for _ in range(len(vkeys_list))]
            if np.max(list(top_key_cnts.values())) > 1:
                # 同じ始端から2つ以上の末端に繋がっている場合
                for top_key, cnt in top_key_cnts.items():
                    total_scores = {}
                    for x, (vkeys, ss) in enumerate(zip(vkeys_list, scores)):
                        if vkeys[0] == top_key:
                            if cnt > 1:
                                # 2個以上同じ始端から出ている場合はスコアの合計を取る
                                total_scores[x] = np.sum(ss)
                                logger.debug(f"target top: [{virtual_vertices[vkeys[0]].vidxs()}], bottom: [{virtual_vertices[vkeys[-1]].vidxs()}], total: {round(total_scores[x], 3)}")   # noqa
                            else:
                                # 1個の場合はそのまま登録
                                total_scores[x] = 1
                    # 最も内積平均値が大きい列を登録対象とする
                    target_regists[list(total_scores.keys())[np.argmax(list(total_scores.values()))]] = True
            else:
                # 全部1個ずつ繋がっている場合はそのまま登録
                target_regists = [True for _ in range(len(vkeys_list))]

            logger.debug(f'target_regists: {target_regists}')

            logger.info("-- 絶対頂点マップ[%s]: マップ生成", midx + 1)

            # XYの最大と最小の抽出
            xu = np.unique([i for i, vks in enumerate(vkeys_list) if target_regists[i]])
            yu = np.unique([i for vks in vkeys_list for i, vk in enumerate(vks)])
            
            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((len(yu), len(xu), 3), (np.inf, np.inf, np.inf))
            vertex_display_map = np.full((len(yu), len(xu)), ' None ')

            xx = 0
            for x, vkeys in enumerate(vkeys_list):
                if not target_regists[x]:
                    continue

                for y, vkey in enumerate(vkeys):
                    logger.debug(f'x: {x}, y: {y}, vv: {vkey}, vidxs: {virtual_vertices[vkey].vidxs()}')

                    vertex_map[y, xx] = vkey
                    vertex_display_map[y, xx] = ':'.join([str(v) for v in virtual_vertices[vkey].vidxs()])

                xx += 1
                logger.debug('-------')

            vertex_maps.append(vertex_map)

            logger.info('\n'.join([', '.join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]), translate=False)
            logger.info("-- 絶対頂点マップ: %s個目:終了 ---------", midx + 1)

            midx += 1
            logger.debug('-----------------------')
                
        return vertex_maps, virtual_vertices
    
    def calc_arc_degree(self, start_pos: MVector3D, mean_pos: MVector3D, target_pos: MVector3D, base_vertical_axis: MVector3D):
        start_normal_pos = (start_pos - mean_pos).normalized()
        target_normal_pos = (target_pos - mean_pos).normalized()
        qq = MQuaternion.rotationTo(start_normal_pos, target_normal_pos)
        degree = qq.toDegreeSign(base_vertical_axis)
        if np.isclose(MVector3D.dotProduct(start_normal_pos, target_normal_pos), -1):
            # ほぼ真後ろを向いてる場合、固定で180度を入れておく
            degree = 180
        if degree < 0:
            # マイナスになった場合、360を足しておく
            degree += 360
        
        return degree
    
    def create_vertex_line_map(self, top_edge_key: tuple, bottom_edge_key: tuple, from_key: tuple, virtual_vertices: dict, \
                               top_keys: list, bottom_keys: list, base_vertical_axis: MVector3D, vkeys: list, vscores: list, loop=0):

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
        logger.debug(f' - top({top_vv.vidxs()}): x[{top_x_pos.to_log()}], y[{top_y_pos.to_log()}], z[{top_z_pos.to_log()}]')

        # int_max = np.iinfo(np.int32).max
        scores = []
        for n, to_key in enumerate(from_vv.connected_vvs):
            to_vv = virtual_vertices[to_key]
            to_pos = to_vv.position()

            direction_dot = MVector3D.dotProduct((from_pos - bottom_pos).normalized(), (to_pos - from_pos).normalized())
            if to_key in vkeys or to_key in bottom_keys or (from_key not in bottom_keys and direction_dot <= 0):
                # 到達済み、最下層、反対方向のベクトルには行かせない
                scores.append(0)
                logger.debug(f' - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], 対象外')
                continue

            # if to_key == top_edge_key:
            #     # TOPに到達するときには必ずそこに向く
            #     scores.append(int_max)
            #     logger.debug(f' - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], TOP到達')
            #     continue

            # # ボーン進行方向(x)
            # to_x_pos = (to_pos - bottom_pos).normalized()
            # # ボーン進行方向に対しての縦軸(y)
            # to_y_pos = to_vv.normal(base_vertical_axis).normalized()
            # # ボーン進行方向に対しての横軸(z)
            # to_z_pos = MVector3D.crossProduct(to_x_pos, to_y_pos)
            # to_qq = MQuaternion.fromDirection(to_z_pos, to_y_pos)

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
            # local_dot = MVector3D.dotProduct(base_vertical_axis, local_next_vpos)
            # prev_dot = MVector3D.dotProduct((from_pos - prev_pos).normalized(), (to_pos - from_pos).normalized()) if prev_pos else 1

            scores.append(score)

            # dots.append(local_dot * prev_dot)

            logger.debug(f' - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], local_next_vpos[{local_next_vpos.to_log()}], score: [{score}], yaw_score: {round(yaw_score, 5)}, pitch_score: {round(pitch_score, 5)}, roll_score: {round(roll_score, 5)}')   # noqa

            # to_degrees.append(self.calc_arc_degree(bottom_edge_start_pos, bottom_edge_mean_pos, to_pos, base_vertical_axis))
            # to_lengths.append(to_pos.distanceToPoint(top_pos))

            # logger.debug(f' - get_vertical_key({n}) : to[{to_vv.vidxs()}], pos[{to_pos.to_log()}], degree[{round(to_degrees[-1], 4)}]')

        if np.count_nonzero(scores) == 0:
            # スコアが付けられなくなったら終了
            return vkeys, vscores
        
        # nearest_idx = np.where(np.array(scores) == int_max)[0]
        # if len(nearest_idx) > 0:
        #     # TOP到達した場合、採用
        #     nearest_idx = nearest_idx[0]
        #     vscores.append(1)
        # else:

        # 最もスコアの高いINDEXを採用
        nearest_idx = np.argmax(scores)
        vscores.append(np.max(scores))
        vertical_key = from_vv.connected_vvs[nearest_idx]

        logger.debug(f'direction: from: [{virtual_vertices[from_key].vidxs()}], to: [{virtual_vertices[vertical_key].vidxs()}]')

        vkeys.insert(0, vertical_key)

        if vertical_key in top_keys:
            # 上端に辿り着いたら終了
            return vkeys, vscores

        return self.create_vertex_line_map(top_edge_key, bottom_edge_key, vertical_key, virtual_vertices, \
                                           top_keys, bottom_keys, base_vertical_axis, vkeys, vscores, loop + 1)

    def get_edge_lines(self, edge_line_pairs: dict, start_vkey: tuple, edge_lines: list, edge_vkeys: list, loop=0):
        if len(edge_vkeys) >= len(edge_line_pairs.keys()) or loop > 500:
            return start_vkey, edge_lines, edge_vkeys
        
        if not start_vkey:
            # X(中央揃え) - Z(降順) - Y(降順)
            sorted_edge_line_pairs = sorted(list(set(edge_line_pairs.keys()) - set(edge_vkeys)), key=lambda x: (abs(x[0]), -x[2], -x[1]))
            start_vkey = sorted_edge_line_pairs[0]
            edge_lines.append([start_vkey])
            edge_vkeys.append(start_vkey)
        
        for next_vkey in sorted(edge_line_pairs[start_vkey], key=lambda x: (x[0], x[2], -x[1])):
            if next_vkey not in edge_vkeys:
                edge_lines[-1].append(next_vkey)
                edge_vkeys.append(next_vkey)
                start_vkey, edge_lines, edge_vkeys = self.get_edge_lines(edge_line_pairs, next_vkey, edge_lines, edge_vkeys, loop + 1)

        return None, edge_lines, edge_vkeys

# class PmxTailorExportService():
#     def __init__(self, options: MExportOptions):
#         self.options = options

#     def execute(self):
#         logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

#         try:
#             service_data_txt = f"{logger.transtext('PmxTailor変換処理実行')}\n------------------------\n{logger.transtext('exeバージョン')}: {self.options.version_name}\n"
#             service_data_txt = f"{service_data_txt}　{logger.transtext('元モデル')}: {os.path.basename(self.options.pmx_model.path)}\n"

#             for pidx, param_option in enumerate(self.options.param_options):
#                 service_data_txt = f"{service_data_txt}\n　【No.{pidx + 1}】 --------- "    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('材質')}: {param_option['material_name']}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('検出度')}: {param_option['similarity']}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('細かさ')}: {param_option['fineness']}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('質量')}: {param_option['mass']}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('柔らかさ')}: {param_option['air_resistance']}"    # noqa
#                 service_data_txt = f"{service_data_txt}\n　　{logger.transtext('張り')}: {param_option['shape_maintenance']}"    # noqa

#             logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

#             model = self.options.pmx_model
#             model.comment += f"\r\n\r\n{logger.transtext('物理')}: PmxTailor"

#             # 保持ボーンは全設定を確認する
#             saved_bone_names = self.get_saved_bone_names(model)

#             for pidx, param_option in enumerate(self.options.param_options):
#                 if not self.create_physics(model, param_option, saved_bone_names):
#                     return False

#             # 最後に出力
#             logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

#             PmxWriter().write(model, self.options.output_path)

#             logger.info("出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title=logger.transtext("成功"))

#             return True
#         except MKilledException:
#             return False
#         except SizingException as se:
#             logger.error("PmxTailor変換処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
#         except Exception:
#             logger.critical("PmxTailor変換処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
#         finally:
#             logging.shutdown()

#     def create_physics(self, model: PmxModel, param_option: dict, saved_bone_names: list):
#         model.comment += f"\r\n{logger.transtext('材質')}: {param_option['material_name']} --------------"    # noqa
#         model.comment += f"\r\n　　{logger.transtext('剛体グループ')}: {param_option['rigidbody'].collision_group + 1}"    # noqa
#         model.comment += f", {logger.transtext('細かさ')}: {param_option['fineness']}"    # noqa
#         model.comment += f", {logger.transtext('質量')}: {param_option['mass']}"    # noqa
#         model.comment += f", {logger.transtext('柔らかさ')}: {param_option['air_resistance']}"    # noqa
#         model.comment += f", {logger.transtext('張り')}: {param_option['shape_maintenance']}"    # noqa

#         # 頂点CSVが指定されている場合、対象頂点リスト生成
#         if param_option['vertices_csv']:
#             target_vertices = []
#             try:
#                 with open(param_option['vertices_csv'], encoding='cp932', mode='r') as f:
#                     reader = csv.reader(f)
#                     next(reader)            # ヘッダーを読み飛ばす
#                     for row in reader:
#                         if len(row) > 1 and int(row[1]) in model.material_vertices[param_option['material_name']]:
#                             target_vertices.append(int(row[1]))
#             except Exception:
#                 logger.warning("頂点CSVが正常に読み込めなかったため、処理を終了します", decoration=MLogger.DECORATION_BOX)
#                 return False
#         else:
#             target_vertices = list(model.material_vertices[param_option['material_name']])

#         if param_option['exist_physics_clear'] in [logger.transtext('上書き'), logger.transtext('再利用')]:
#             # 既存材質削除フラグONの場合
#             logger.info("【%s】既存材質削除", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#             model = self.clear_exist_physics(model, param_option, param_option['material_name'], target_vertices, saved_bone_names)

#             if not model:
#                 return False

#         if param_option['exist_physics_clear'] == logger.transtext('再利用'):
#             if param_option['physics_type'] in [logger.transtext('髪'), logger.transtext('単一揺'), logger.transtext('胸')]:
#                 logger.info("【%s】ボーンマップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 logger.info("【%s】剛体生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 root_rigidbody, registed_rigidbodies = self.create_vertical_rigidbody(model, param_option)

#                 logger.info("【%s】ジョイント生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 self.create_vertical_joint(model, param_option, root_rigidbody, registed_rigidbodies)
#             else:
#                 logger.info("【%s】ボーンマップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 bone_blocks = self.create_bone_blocks(model, param_option, param_option['material_name'])

#                 if not bone_blocks:
#                     logger.warning("有効なボーンマップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
#                     return False

#                 logger.info("【%s】剛体生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 root_rigidbody, registed_rigidbodies = self.create_rigidbody_by_bone_blocks(model, param_option, bone_blocks)

#                 logger.info("【%s】ジョイント生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#                 self.create_joint_by_bone_blocks(model, param_option, bone_blocks, root_rigidbody, registed_rigidbodies)

#         else:
#             logger.info("【%s】頂点マップ生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#             vertex_maps, duplicate_vertices, registed_iidxs, duplicate_indices, index_combs_by_vpos \
#                 = self.create_vertex_map(model, param_option, param_option['material_name'], target_vertices)
            
#             if not vertex_maps:
#                 return False
            
#             # 各頂点の有効INDEX数が最も多いものをベースとする
#             map_cnt = []
#             for vertex_map in vertex_maps:
#                 map_cnt.append(np.count_nonzero(vertex_map >= 0))
            
#             if len(map_cnt) == 0:
#                 logger.warning("有効な頂点マップが生成できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
#                 return False
            
#             vertex_map_orders = [k for k in np.argsort(-np.array(map_cnt)) if map_cnt[k] > np.max(map_cnt) * 0.5]
            
#             logger.info("【%s】ボーン生成", param_option['material_name'], decoration=MLogger.DECORATION_LINE)

#             root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances \
#                 = self.create_bone(model, param_option, vertex_map_orders, vertex_maps, vertex_connecteds)

#             vertex_remaining_set = set(target_vertices)

#             for base_map_idx in vertex_map_orders:
#                 logger.info("【%s(No.%s)】ウェイト分布", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

#                 self.create_weight(model, param_option, vertex_maps[base_map_idx], vertex_connecteds[base_map_idx], duplicate_vertices, \
#                                    all_registed_bone_indexs[base_map_idx], all_bone_horizonal_distances[base_map_idx], all_bone_vertical_distances[base_map_idx], \
#                                    vertex_remaining_set, target_vertices)

#             if len(list(vertex_remaining_set)) > 0:
#                 logger.info("【%s】残ウェイト分布", param_option['material_name'], decoration=MLogger.DECORATION_LINE)
                
#                 self.create_remaining_weight(model, param_option, vertex_maps, vertex_remaining_set, vertex_map_orders, target_vertices)
    
#             if param_option['edge_material_name']:
#                 logger.info("【%s】裾ウェイト分布", param_option['edge_material_name'], decoration=MLogger.DECORATION_LINE)

#                 edge_vertices = set(model.material_vertices[param_option['edge_material_name']])
#                 self.create_remaining_weight(model, param_option, vertex_maps, edge_vertices, vertex_map_orders, edge_vertices)
        
#             if param_option['back_material_name']:
#                 logger.info("【%s】裏面ウェイト分布", param_option['back_material_name'], decoration=MLogger.DECORATION_LINE)

#                 self.create_back_weight(model, param_option)
    
#             for base_map_idx in vertex_map_orders:
#                 logger.info("【%s(No.%s)】剛体生成", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

#                 root_rigidbody, registed_rigidbodies = self.create_rigidbody(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx], root_bone)

#                 logger.info("【%s(No.%s)】ジョイント生成", param_option['material_name'], base_map_idx + 1, decoration=MLogger.DECORATION_LINE)

#                 self.create_joint(model, param_option, vertex_connecteds[base_map_idx], tmp_all_bones, all_registed_bone_indexs[base_map_idx], root_rigidbody, registed_rigidbodies)

#         return True

#     def create_bone_blocks(self, model: PmxModel, param_option: dict, material_name: str):
#         bone_grid = param_option["bone_grid"]
#         bone_grid_cols = param_option["bone_grid_cols"]
#         bone_grid_rows = param_option["bone_grid_rows"]

#         # ウェイトボーンリスト取得（ついでにウェイト正規化）
#         weighted_bone_pairs = []
#         for vertex_idx in model.material_vertices[material_name]:
#             vertex = model.vertex_dict[vertex_idx]
#             if type(vertex.deform) is Bdef2 or type(vertex.deform) is Sdef:
#                 if 0 < vertex.deform.weight0 < 1:
#                     # 2つめのボーンも有効値を持っている場合、判定対象
#                     key = (min(vertex.deform.index0, vertex.deform.index1), max(vertex.deform.index0, vertex.deform.index1))
#                     if key not in weighted_bone_pairs:
#                         weighted_bone_pairs.append(key)
#             elif type(vertex.deform) is Bdef4:
#                 # ウェイト正規化
#                 total_weights = np.array([vertex.deform.weight0, vertex.deform.weight1, vertex.deform.weight2, vertex.deform.weight3])
#                 weights = total_weights / total_weights.sum(axis=0, keepdims=1)

#                 vertex.deform.weight0 = weights[0]
#                 vertex.deform.weight1 = weights[1]
#                 vertex.deform.weight2 = weights[2]
#                 vertex.deform.weight3 = weights[3]

#                 weighted_bone_indexes = []
#                 if vertex.deform.weight0 > 0:
#                     weighted_bone_indexes.append(vertex.deform.index0)
#                 if vertex.deform.weight1 > 0:
#                     weighted_bone_indexes.append(vertex.deform.index1)
#                 if vertex.deform.weight2 > 0:
#                     weighted_bone_indexes.append(vertex.deform.index2)
#                 if vertex.deform.weight3 > 0:
#                     weighted_bone_indexes.append(vertex.deform.index3)

#                 for bi0, bi1 in list(itertools.combinations(weighted_bone_indexes, 2)):
#                     # ボーン2つずつのペアでウェイト繋がり具合を保持する
#                     key = (min(bi0, bi1), max(bi0, bi1))
#                     if key not in weighted_bone_pairs:
#                         weighted_bone_pairs.append(key)
        
#         bone_blocks = {}
#         for pac in range(bone_grid_cols):
#             prev_above_bone_name = None
#             prev_above_bone_position = None
#             for par in range(bone_grid_rows):
#                 prev_above_bone_name = bone_grid[par][pac]
#                 if not prev_above_bone_name:
#                     continue
                
#                 is_above_connected = True
#                 prev_above_bone_position = model.bones[prev_above_bone_name].position
#                 prev_above_bone_index = model.bones[prev_above_bone_name].index
#                 if prev_above_bone_name:
#                     prev_below_bone_name = None
#                     prev_below_bone_position = None
#                     pbr = par + 1
#                     if pbr in bone_grid and pac in bone_grid[pbr]:
#                         prev_below_bone_name = bone_grid[pbr][pac]
#                         if prev_below_bone_name:
#                             prev_below_bone_position = model.bones[prev_below_bone_name].position
#                         if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                             # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
#                             prev_below_bone_name = prev_above_bone_name
#                             prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
#                     elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                         prev_below_bone_name = prev_above_bone_name
#                         prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

#                     next_above_bone_name = None
#                     next_above_bone_position = None
#                     nnac = [k for k, v in bone_grid[par].items() if v][-1]
#                     if pac < nnac:
#                         # 右周りにボーンの連携をチェック
#                         nac = pac + 1
#                         if par in bone_grid and nac in bone_grid[par]:
#                             next_above_bone_name = bone_grid[par][nac]
#                             if next_above_bone_name:
#                                 next_above_bone_position = model.bones[next_above_bone_name].position
#                                 next_above_bone_index = model.bones[next_above_bone_name].index
#                             else:
#                                 # 隣がない場合、1つ前のボーンと結合させる
#                                 nac = pac - 1
#                                 next_above_bone_name = bone_grid[par][nac]
#                                 if next_above_bone_name:
#                                     next_above_bone_position = model.bones[next_above_bone_name].position
#                                     next_above_bone_index = model.bones[next_above_bone_name].index
#                                     is_above_connected = False
#                     else:
#                         # 一旦円周を描いてみる
#                         next_above_bone_name = bone_grid[par][0]
#                         nac = 0
#                         if next_above_bone_name:
#                             next_above_bone_position = model.bones[next_above_bone_name].position
#                             next_above_bone_index = model.bones[next_above_bone_name].index
#                             key = (min(prev_above_bone_index, next_above_bone_index), max(prev_above_bone_index, next_above_bone_index))
#                             if key not in weighted_bone_pairs:
#                                 # ウェイトが乗ってなかった場合、2つ前のボーンと結合させる
#                                 nac = pac - 1
#                                 if par in bone_grid and nac in bone_grid[par]:
#                                     next_above_bone_name = bone_grid[par][nac]
#                                     if next_above_bone_name:
#                                         next_above_bone_position = model.bones[next_above_bone_name].position
#                                         next_above_bone_index = model.bones[next_above_bone_name].index
#                                         is_above_connected = False

#                     next_below_bone_name = None
#                     next_below_bone_position = None
#                     nbr = par + 1
#                     if nbr in bone_grid and nac in bone_grid[nbr]:
#                         next_below_bone_name = bone_grid[nbr][nac]
#                         if next_below_bone_name:
#                             next_below_bone_position = model.bones[next_below_bone_name].position
#                         if next_above_bone_name and not next_below_bone_name and model.bones[next_above_bone_name].tail_position != MVector3D():
#                             # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
#                             next_below_bone_name = next_above_bone_name
#                             next_below_bone_position = next_above_bone_position + model.bones[next_above_bone_name].tail_position
#                     elif next_above_bone_name and model.bones[next_above_bone_name].tail_position != MVector3D():
#                         next_below_bone_name = next_above_bone_name
#                         next_below_bone_position = next_above_bone_position + model.bones[next_above_bone_name].tail_position

#                     prev_prev_above_bone_name = None
#                     prev_prev_above_bone_position = None
#                     if pac > 0:
#                         # 左周りにボーンの連携をチェック
#                         ppac = pac - 1
#                         if par in bone_grid and ppac in bone_grid[par]:
#                             prev_prev_above_bone_name = bone_grid[par][ppac]
#                             if prev_prev_above_bone_name:
#                                 prev_prev_above_bone_position = model.bones[prev_prev_above_bone_name].position
#                             else:
#                                 # 隣がない場合、prev_aboveと同じにする
#                                 prev_prev_above_bone_name = prev_above_bone_name
#                                 prev_prev_above_bone_position = prev_above_bone_position
#                         else:
#                             prev_prev_above_bone_name = prev_above_bone_name
#                             prev_prev_above_bone_position = prev_above_bone_position
#                     else:
#                         # 一旦円周を描いてみる
#                         ppac = [k for k, v in bone_grid[par].items() if v][-1]
#                         prev_prev_above_bone_name = bone_grid[par][ppac]
#                         if prev_prev_above_bone_name:
#                             prev_prev_above_bone_position = model.bones[prev_prev_above_bone_name].position
#                             prev_prev_above_bone_index = model.bones[prev_prev_above_bone_name].index
#                             key = (min(prev_above_bone_index, prev_prev_above_bone_index), max(prev_above_bone_index, prev_prev_above_bone_index))
#                             if key not in weighted_bone_pairs:
#                                 # ウェイトが乗ってなかった場合、prev_aboveと同じにする
#                                 prev_prev_above_bone_name = prev_above_bone_name
#                                 prev_prev_above_bone_position = prev_above_bone_position
#                         else:
#                             prev_prev_above_bone_name = prev_above_bone_name
#                             prev_prev_above_bone_position = prev_above_bone_position

#                     if prev_above_bone_name and prev_below_bone_name and next_above_bone_name and next_below_bone_name:
#                         bone_blocks[prev_above_bone_name] = {'prev_above': prev_above_bone_name, 'prev_below': prev_below_bone_name, \
#                                                              'next_above': next_above_bone_name, 'next_below': next_below_bone_name, \
#                                                              'prev_above_pos': prev_above_bone_position, 'prev_below_pos': prev_below_bone_position, \
#                                                              'next_above_pos': next_above_bone_position, 'next_below_pos': next_below_bone_position, \
#                                                              'prev_prev_above': prev_prev_above_bone_name, 'prev_prev_above_pos': prev_prev_above_bone_position, \
#                                                              'yi': par, 'xi': pac, 'is_above_connected': is_above_connected}
#                         logger.debug(f'prev_above: {prev_above_bone_name}, [{prev_above_bone_position.to_log()}], ' \
#                                      + f'next_above: {next_above_bone_name}, [{next_above_bone_position.to_log()}], ' \
#                                      + f'prev_below: {prev_below_bone_name}, [{prev_below_bone_position.to_log()}], ' \
#                                      + f'next_below: {next_below_bone_name}, [{next_below_bone_position.to_log()}], ' \
#                                      + f'prev_prev_above: {prev_prev_above_bone_name}, [{prev_prev_above_bone_position.to_log()}], ' \
#                                      + f'yi: {par}, xi: {pac}, is_above_connected: {is_above_connected}')

#         return bone_blocks
    
#     def create_vertical_joint(self, model: PmxModel, param_option: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
#         bone_grid_cols = param_option["bone_grid_cols"]
#         bone_grid_rows = param_option["bone_grid_rows"]
#         bone_grid = param_option["bone_grid"]

#         # ジョイント生成
#         created_joints = {}

#         # 縦ジョイント情報
#         param_vertical_joint = param_option['vertical_joint']

#         prev_joint_cnt = 0

#         for pac in range(bone_grid_cols):
#             # ジョイント生成
#             created_joints = {}

#             valid_rows = [par for par in range(bone_grid_rows) if par]
#             if len(valid_rows) == 0:
#                 continue
            
#             max_vy = valid_rows[-1]
#             min_vy = 0
#             xs = np.arange(min_vy, max_vy, step=1)
        
#             if param_vertical_joint:
#                 coefficient = param_option['vertical_joint_coefficient']

#                 vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
#                 vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
#                 vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

#                 vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
#                 vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
#                 vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

#                 vertical_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x()]])), xs)             # noqa
#                 vertical_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y()]])), xs)             # noqa
#                 vertical_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z()]])), xs)             # noqa

#                 vertical_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x()]])), xs)             # noqa
#                 vertical_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y()]])), xs)             # noqa
#                 vertical_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z()]])), xs)             # noqa

#                 vertical_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x()]])), xs)             # noqa
#                 vertical_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y()]])), xs)             # noqa
#                 vertical_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z()]])), xs)             # noqa

#                 vertical_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x()]])), xs)             # noqa
#                 vertical_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y()]])), xs)             # noqa
#                 vertical_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z()]])), xs)             # noqa
            
#             prev_above_bone_name = None
#             prev_above_bone_position = None
#             for par in range(bone_grid_rows):
#                 prev_above_bone_name = bone_grid[par][pac]
#                 if not prev_above_bone_name or prev_above_bone_name not in model.bones:
#                     continue

#                 prev_above_bone_position = model.bones[prev_above_bone_name].position
#                 prev_below_bone_name = None
#                 prev_below_bone_position = None
#                 prev_below_below_bone_name = None
#                 prev_below_below_bone_position = None
#                 if prev_above_bone_name:
#                     pbr = par + 1
#                     if pbr in bone_grid and pac in bone_grid[pbr]:
#                         prev_below_bone_name = bone_grid[pbr][pac]
#                         if prev_below_bone_name:
#                             prev_below_bone_position = model.bones[prev_below_bone_name].position

#                             pbbr = pbr + 1
#                             if pbbr in bone_grid and pac in bone_grid[pbbr]:
#                                 prev_below_below_bone_name = bone_grid[pbbr][pac]
#                                 if prev_below_below_bone_name:
#                                     prev_below_below_bone_position = model.bones[prev_below_below_bone_name].position
#                                 if not prev_below_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                                     # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
#                                     prev_below_below_bone_name = prev_above_bone_name
#                                     prev_below_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
#                             elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                                 prev_below_below_bone_name = prev_above_bone_name
#                                 prev_below_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

#                         if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                             # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
#                             prev_below_bone_name = prev_above_bone_name
#                             prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
#                     elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                         prev_below_bone_name = prev_above_bone_name
#                         prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

#                 if prev_above_bone_position and prev_below_bone_position:
#                     if not prev_below_below_bone_position:
#                         prev_below_below_bone_position = prev_below_bone_position

#                     if par == 0 and prev_above_bone_name in registed_rigidbodies:
#                         # ルート剛体と根元剛体を繋ぐジョイント
#                         joint_name = f'↓|{root_rigidbody.name}|{registed_rigidbodies[prev_above_bone_name]}'

#                         # 縦ジョイント
#                         joint_vec = prev_above_bone_position

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (root_rigidbody.shape_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         joint = Joint(joint_name, joint_name, 0, root_rigidbody.index, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[par], vertical_limit_min_mov_ys[par], vertical_limit_min_mov_zs[par]), \
#                                       MVector3D(vertical_limit_max_mov_xs[par], vertical_limit_max_mov_ys[par], vertical_limit_max_mov_zs[par]),
#                                       MVector3D(math.radians(vertical_limit_min_rot_xs[par]), math.radians(vertical_limit_min_rot_ys[par]), math.radians(vertical_limit_min_rot_zs[par])),
#                                       MVector3D(math.radians(vertical_limit_max_rot_xs[par]), math.radians(vertical_limit_max_rot_ys[par]), math.radians(vertical_limit_max_rot_zs[par])),
#                                       MVector3D(vertical_spring_constant_mov_xs[par], vertical_spring_constant_mov_ys[par], vertical_spring_constant_mov_zs[par]), \
#                                       MVector3D(vertical_spring_constant_rot_xs[par], vertical_spring_constant_rot_ys[par], vertical_spring_constant_rot_zs[par]))
#                         created_joints[f'0:{root_rigidbody.index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'] = joint

#                         # バランサー剛体が必要な場合
#                         if param_option["rigidbody_balancer"]:
#                             balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
#                             joint_name = f'B|{prev_above_bone_name}|{balancer_prev_above_bone_name}'
#                             joint_key = f'8:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}'

#                             joint_vec = model.rigidbodies[prev_above_bone_name].shape_position

#                             # 回転量
#                             joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                             joint_axis = (model.rigidbodies[balancer_prev_above_bone_name].shape_position - prev_above_bone_position).normalized()
#                             joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                             joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                             joint_euler = joint_rotation_qq.toEulerAngles()
#                             joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[balancer_prev_above_bone_name].index,
#                                           joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                           MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
#                             created_joints[joint_key] = joint
            
#                     if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
#                         # 縦ジョイント
#                         joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{prev_below_bone_name}'
#                         joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'   # noqa

#                         if joint_key not in created_joints:
#                             # 未登録のみ追加
                            
#                             # 縦ジョイント
#                             joint_vec = prev_below_bone_position

#                             # 回転量
#                             joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                             joint_axis = (prev_below_below_bone_position - prev_above_bone_position).normalized()
#                             joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                             joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                             joint_euler = joint_rotation_qq.toEulerAngles()
#                             joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                           model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
#                                           joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[par], vertical_limit_min_mov_ys[par], vertical_limit_min_mov_zs[par]), \
#                                           MVector3D(vertical_limit_max_mov_xs[par], vertical_limit_max_mov_ys[par], vertical_limit_max_mov_zs[par]),
#                                           MVector3D(math.radians(vertical_limit_min_rot_xs[par]), math.radians(vertical_limit_min_rot_ys[par]), math.radians(vertical_limit_min_rot_zs[par])),
#                                           MVector3D(math.radians(vertical_limit_max_rot_xs[par]), math.radians(vertical_limit_max_rot_ys[par]), math.radians(vertical_limit_max_rot_zs[par])),
#                                           MVector3D(vertical_spring_constant_mov_xs[par], vertical_spring_constant_mov_ys[par], vertical_spring_constant_mov_zs[par]), \
#                                           MVector3D(vertical_spring_constant_rot_xs[par], vertical_spring_constant_rot_ys[par], vertical_spring_constant_rot_zs[par]))
#                             created_joints[joint_key] = joint

#                             # バランサー剛体が必要な場合
#                             if param_option["rigidbody_balancer"]:
#                                 balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
#                                 joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
#                                 joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

#                                 joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

#                                 # 回転量
#                                 joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                                 joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
#                                 joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                                 joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                                 joint_euler = joint_rotation_qq.toEulerAngles()
#                                 joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                                 joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
#                                               model.rigidbodies[balancer_prev_below_bone_name].index,
#                                               joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                               MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
#                                 created_joints[joint_key] = joint

#                                 # バランサー補助剛体
#                                 balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
#                                 joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
#                                 joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'  # noqa
#                                 joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
#                                               model.rigidbodies[balancer_prev_below_bone_name].index,
#                                               MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
#                                               MVector3D(),MVector3D(), MVector3D())   # noqa
#                                 created_joints[joint_key] = joint
                                
#             for joint_key in sorted(created_joints.keys()):
#                 # ジョイントを登録
#                 joint = created_joints[joint_key]
#                 joint.index = len(model.joints)

#                 if joint.name in model.joints:
#                     logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
#                     joint.name += randomname(3)

#                 model.joints[joint.name] = joint

#             prev_joint_cnt += len(created_joints)

#         logger.info("-- ジョイント: %s個目:終了", prev_joint_cnt)
                            
#         return root_rigidbody
    
#     def create_joint_by_bone_blocks(self, model: PmxModel, param_option: dict, bone_blocks: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
#         bone_grid_rows = param_option["bone_grid_rows"]
#         # bone_grid_cols = param_option["bone_grid_cols"]

#         # ジョイント生成
#         created_joints = {}

#         # # 略称
#         # abb_name = param_option['abb_name']
#         # 縦ジョイント情報
#         param_vertical_joint = param_option['vertical_joint']
#         # 横ジョイント情報
#         param_horizonal_joint = param_option['horizonal_joint']
#         # 斜めジョイント情報
#         param_diagonal_joint = param_option['diagonal_joint']
#         # 逆ジョイント情報
#         param_reverse_joint = param_option['reverse_joint']

#         prev_joint_cnt = 0

#         max_vy = bone_grid_rows
#         middle_vy = (bone_grid_rows) * 0.3
#         min_vy = 0
#         xs = np.arange(min_vy, max_vy, step=1)
    
#         if param_vertical_joint:
#             coefficient = param_option['vertical_joint_coefficient']

#             vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
#             vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
#             vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

#             vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
#             vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
#             vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

#             vertical_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x()]])), xs)             # noqa
#             vertical_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y()]])), xs)             # noqa
#             vertical_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z()]])), xs)             # noqa

#             vertical_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x()]])), xs)             # noqa
#             vertical_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y()]])), xs)             # noqa
#             vertical_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z()]])), xs)             # noqa

#             vertical_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x()]])), xs)             # noqa
#             vertical_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y()]])), xs)             # noqa
#             vertical_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z()]])), xs)             # noqa

#             vertical_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             vertical_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             vertical_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_horizonal_joint:
#             coefficient = param_option['horizonal_joint_coefficient']
            
#             horizonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x()]])), xs)             # noqa
#             horizonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y()]])), xs)             # noqa
#             horizonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z()]])), xs)             # noqa

#             horizonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x()]])), xs)             # noqa
#             horizonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y()]])), xs)             # noqa
#             horizonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z()]])), xs)             # noqa

#             horizonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x()]])), xs)             # noqa
#             horizonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y()]])), xs)             # noqa
#             horizonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z()]])), xs)             # noqa

#             horizonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x()]])), xs)             # noqa
#             horizonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y()]])), xs)             # noqa
#             horizonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z()]])), xs)             # noqa

#             horizonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x()]])), xs)             # noqa
#             horizonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y()]])), xs)             # noqa
#             horizonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z()]])), xs)             # noqa

#             horizonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             horizonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             horizonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_diagonal_joint:
#             coefficient = param_option['diagonal_joint_coefficient']

#             diagonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x()]])), xs)             # noqa
#             diagonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y()]])), xs)             # noqa
#             diagonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z()]])), xs)             # noqa

#             diagonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x()]])), xs)             # noqa
#             diagonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y()]])), xs)             # noqa
#             diagonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z()]])), xs)             # noqa

#             diagonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x()]])), xs)             # noqa
#             diagonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y()]])), xs)             # noqa
#             diagonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z()]])), xs)             # noqa

#             diagonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x()]])), xs)             # noqa
#             diagonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y()]])), xs)             # noqa
#             diagonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z()]])), xs)             # noqa

#             diagonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x()]])), xs)             # noqa
#             diagonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y()]])), xs)             # noqa
#             diagonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z()]])), xs)             # noqa

#             diagonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             diagonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             diagonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_reverse_joint:
#             coefficient = param_option['reverse_joint_coefficient']

#             reverse_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x()]])), xs)             # noqa
#             reverse_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y()]])), xs)             # noqa
#             reverse_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z()]])), xs)             # noqa

#             reverse_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x()]])), xs)             # noqa
#             reverse_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y()]])), xs)             # noqa
#             reverse_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z()]])), xs)             # noqa

#             reverse_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x()]])), xs)             # noqa
#             reverse_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y()]])), xs)             # noqa
#             reverse_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z()]])), xs)             # noqa

#             reverse_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x()]])), xs)             # noqa
#             reverse_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y()]])), xs)             # noqa
#             reverse_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z()]])), xs)             # noqa

#             reverse_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x()]])), xs)             # noqa
#             reverse_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y()]])), xs)             # noqa
#             reverse_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z()]])), xs)             # noqa

#             reverse_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             reverse_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             reverse_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z()]])), xs)             # noqa
        
#         for bone_block in bone_blocks.values():
#             prev_above_bone_name = bone_block['prev_above']
#             prev_above_bone_position = bone_block['prev_above_pos']
#             prev_below_bone_name = bone_block['prev_below']
#             prev_below_bone_position = bone_block['prev_below_pos']
#             next_above_bone_name = bone_block['next_above']
#             next_above_bone_position = bone_block['next_above_pos']
#             next_below_bone_name = bone_block['next_below']
#             next_below_bone_position = bone_block['next_below_pos']
#             yi = bone_block['yi']
#             # xi = bone_block['xi']

#             if yi == 0 and prev_above_bone_name in registed_rigidbodies:
#                 # ルート剛体と根元剛体を繋ぐジョイント
#                 joint_name = f'↓|{root_rigidbody.name}|{registed_rigidbodies[prev_above_bone_name]}'

#                 # 縦ジョイント
#                 joint_vec = prev_above_bone_position

#                 # 回転量
#                 joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                 joint_axis = (root_rigidbody.shape_position - prev_above_bone_position).normalized()
#                 joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                 joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                 joint_euler = joint_rotation_qq.toEulerAngles()
#                 joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                 joint = Joint(joint_name, joint_name, 0, root_rigidbody.index, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                               joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yi], vertical_limit_min_mov_ys[yi], vertical_limit_min_mov_zs[yi]), \
#                               MVector3D(vertical_limit_max_mov_xs[yi], vertical_limit_max_mov_ys[yi], vertical_limit_max_mov_zs[yi]),
#                               MVector3D(math.radians(vertical_limit_min_rot_xs[yi]), math.radians(vertical_limit_min_rot_ys[yi]), math.radians(vertical_limit_min_rot_zs[yi])),
#                               MVector3D(math.radians(vertical_limit_max_rot_xs[yi]), math.radians(vertical_limit_max_rot_ys[yi]), math.radians(vertical_limit_max_rot_zs[yi])),
#                               MVector3D(vertical_spring_constant_mov_xs[yi], vertical_spring_constant_mov_ys[yi], vertical_spring_constant_mov_zs[yi]), \
#                               MVector3D(vertical_spring_constant_rot_xs[yi], vertical_spring_constant_rot_ys[yi], vertical_spring_constant_rot_zs[yi]))   # noqa
#                 created_joints[f'0:{root_rigidbody.index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'] = joint

#                 # バランサー剛体が必要な場合
#                 if param_option["rigidbody_balancer"]:
#                     balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
#                     joint_name = f'B|{prev_above_bone_name}|{balancer_prev_above_bone_name}'
#                     joint_key = f'8:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}'

#                     joint_vec = model.rigidbodies[prev_above_bone_name].shape_position

#                     # 回転量
#                     joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                     joint_axis = (model.rigidbodies[balancer_prev_above_bone_name].shape_position - prev_above_bone_position).normalized()
#                     joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                     joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                     joint_euler = joint_rotation_qq.toEulerAngles()
#                     joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                     joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, model.rigidbodies[balancer_prev_above_bone_name].index,
#                                   joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                   MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
#                     created_joints[joint_key] = joint
    
#             if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
#                 # 縦ジョイント
#                 joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[prev_below_bone_name]}'
#                 joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

#                 if joint_key not in created_joints:
#                     # 未登録のみ追加
                    
#                     # 縦ジョイント
#                     joint_vec = prev_below_bone_position

#                     # 回転量
#                     joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                     joint_axis = (next_above_bone_position - prev_above_bone_position).normalized()
#                     joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                     joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                     joint_euler = joint_rotation_qq.toEulerAngles()
#                     joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                     joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                   model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
#                                   joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yi], vertical_limit_min_mov_ys[yi], vertical_limit_min_mov_zs[yi]), \
#                                   MVector3D(vertical_limit_max_mov_xs[yi], vertical_limit_max_mov_ys[yi], vertical_limit_max_mov_zs[yi]),
#                                   MVector3D(math.radians(vertical_limit_min_rot_xs[yi]), math.radians(vertical_limit_min_rot_ys[yi]), math.radians(vertical_limit_min_rot_zs[yi])),
#                                   MVector3D(math.radians(vertical_limit_max_rot_xs[yi]), math.radians(vertical_limit_max_rot_ys[yi]), math.radians(vertical_limit_max_rot_zs[yi])),
#                                   MVector3D(vertical_spring_constant_mov_xs[yi], vertical_spring_constant_mov_ys[yi], vertical_spring_constant_mov_zs[yi]), \
#                                   MVector3D(vertical_spring_constant_rot_xs[yi], vertical_spring_constant_rot_ys[yi], vertical_spring_constant_rot_zs[yi]))   # noqa
#                     created_joints[joint_key] = joint

#                     if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                         logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                         prev_joint_cnt = len(created_joints) // 200
                    
#                     if param_reverse_joint and prev_below_bone_name in registed_rigidbodies and prev_above_bone_name in registed_rigidbodies:
#                         # 逆ジョイント
#                         joint_name = f'↑|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
#                         joint_key = f'1:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

#                         if joint_key not in created_joints:
#                             # 未登録のみ追加
#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
#                                           model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                                           joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yi], reverse_limit_min_mov_ys[yi], reverse_limit_min_mov_zs[yi]), \
#                                           MVector3D(reverse_limit_max_mov_xs[yi], reverse_limit_max_mov_ys[yi], reverse_limit_max_mov_zs[yi]),
#                                           MVector3D(math.radians(reverse_limit_min_rot_xs[yi]), math.radians(reverse_limit_min_rot_ys[yi]), math.radians(reverse_limit_min_rot_zs[yi])),
#                                           MVector3D(math.radians(reverse_limit_max_rot_xs[yi]), math.radians(reverse_limit_max_rot_ys[yi]), math.radians(reverse_limit_max_rot_zs[yi])),
#                                           MVector3D(reverse_spring_constant_mov_xs[yi], reverse_spring_constant_mov_ys[yi], reverse_spring_constant_mov_zs[yi]), \
#                                           MVector3D(reverse_spring_constant_rot_xs[yi], reverse_spring_constant_rot_ys[yi], reverse_spring_constant_rot_zs[yi]))
#                             created_joints[joint_key] = joint

#                             if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                                 logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                                 prev_joint_cnt = len(created_joints) // 200

#                     # バランサー剛体が必要な場合
#                     if param_option["rigidbody_balancer"]:
#                         balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
#                         joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
#                         joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

#                         joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
#                                       model.rigidbodies[balancer_prev_below_bone_name].index,
#                                       joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                       MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))
#                         created_joints[joint_key] = joint

#                         # バランサー補助剛体
#                         balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
#                         joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
#                         joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'
#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
#                                       model.rigidbodies[balancer_prev_below_bone_name].index,
#                                       MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
#                                       MVector3D(), MVector3D(), MVector3D())
#                         created_joints[joint_key] = joint
                                                    
#             if param_horizonal_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
#                 # 横ジョイント
#                 if prev_above_bone_name != next_above_bone_name:
#                     joint_name = f'→|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
#                     joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                        
#                         joint_vec = np.mean([prev_above_bone_position, prev_below_bone_position, \
#                                              next_above_bone_position, next_below_bone_position])

#                         # 回転量
#                         joint_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                       model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi], horizonal_limit_min_mov_ys[yi], horizonal_limit_min_mov_zs[yi]), \
#                                       MVector3D(horizonal_limit_max_mov_xs[yi], horizonal_limit_max_mov_ys[yi], horizonal_limit_max_mov_zs[yi]),
#                                       MVector3D(math.radians(horizonal_limit_min_rot_xs[yi]), math.radians(horizonal_limit_min_rot_ys[yi]), math.radians(horizonal_limit_min_rot_zs[yi])),
#                                       MVector3D(math.radians(horizonal_limit_max_rot_xs[yi]), math.radians(horizonal_limit_max_rot_ys[yi]), math.radians(horizonal_limit_max_rot_zs[yi])),
#                                       MVector3D(horizonal_spring_constant_mov_xs[yi], horizonal_spring_constant_mov_ys[yi], horizonal_spring_constant_mov_zs[yi]), \
#                                       MVector3D(horizonal_spring_constant_rot_xs[yi], horizonal_spring_constant_rot_ys[yi], horizonal_spring_constant_rot_zs[yi]))    # noqa
#                         created_joints[joint_key] = joint

#                         if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                             logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                             prev_joint_cnt = len(created_joints) // 200
                        
#                     if param_reverse_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
#                         # 横逆ジョイント
#                         joint_name = f'←|{registed_rigidbodies[next_above_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
#                         joint_key = f'3:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

#                         if joint_key not in created_joints:
#                             # 未登録のみ追加
                            
#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index, \
#                                           model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                                           joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yi], reverse_limit_min_mov_ys[yi], reverse_limit_min_mov_zs[yi]), \
#                                           MVector3D(reverse_limit_max_mov_xs[yi], reverse_limit_max_mov_ys[yi], reverse_limit_max_mov_zs[yi]),
#                                           MVector3D(math.radians(reverse_limit_min_rot_xs[yi]), math.radians(reverse_limit_min_rot_ys[yi]), math.radians(reverse_limit_min_rot_zs[yi])),
#                                           MVector3D(math.radians(reverse_limit_max_rot_xs[yi]), math.radians(reverse_limit_max_rot_ys[yi]), math.radians(reverse_limit_max_rot_zs[yi])),
#                                           MVector3D(reverse_spring_constant_mov_xs[yi], reverse_spring_constant_mov_ys[yi], reverse_spring_constant_mov_zs[yi]), \
#                                           MVector3D(reverse_spring_constant_rot_xs[yi], reverse_spring_constant_rot_ys[yi], reverse_spring_constant_rot_zs[yi]))      # noqa
#                             created_joints[joint_key] = joint

#                             if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                                 logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                                 prev_joint_cnt = len(created_joints) // 200
                            
#             if param_diagonal_joint and prev_above_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies and \
#                     prev_below_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:                                # noqa
#                 # ＼ジョイント
#                 joint_name = f'＼|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
#                 joint_key = f'4:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

#                 if joint_key not in created_joints:
#                     # 未登録のみ追加
                    
#                     # ＼ジョイント
#                     joint_vec = np.mean([prev_below_bone_position, next_below_bone_position])

#                     # 回転量
#                     joint_axis_up = (next_below_bone_position - prev_above_bone_position).normalized()
#                     joint_axis = (prev_below_bone_position - next_above_bone_position).normalized()
#                     joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                     joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                     joint_euler = joint_rotation_qq.toEulerAngles()
#                     joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                     joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                   model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
#                                   joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yi], diagonal_limit_min_mov_ys[yi], diagonal_limit_min_mov_zs[yi]), \
#                                   MVector3D(diagonal_limit_max_mov_xs[yi], diagonal_limit_max_mov_ys[yi], diagonal_limit_max_mov_zs[yi]),
#                                   MVector3D(math.radians(diagonal_limit_min_rot_xs[yi]), math.radians(diagonal_limit_min_rot_ys[yi]), math.radians(diagonal_limit_min_rot_zs[yi])),
#                                   MVector3D(math.radians(diagonal_limit_max_rot_xs[yi]), math.radians(diagonal_limit_max_rot_ys[yi]), math.radians(diagonal_limit_max_rot_zs[yi])),
#                                   MVector3D(diagonal_spring_constant_mov_xs[yi], diagonal_spring_constant_mov_ys[yi], diagonal_spring_constant_mov_zs[yi]), \
#                                   MVector3D(diagonal_spring_constant_rot_xs[yi], diagonal_spring_constant_rot_ys[yi], diagonal_spring_constant_rot_zs[yi]))   # noqa
#                     created_joints[joint_key] = joint

#                     if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                         logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                         prev_joint_cnt = len(created_joints) // 200
                    
#                 # ／ジョイント ---------------
#                 joint_name = f'／|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
#                 joint_key = f'5:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

#                 if joint_key not in created_joints:
#                     # 未登録のみ追加
                
#                     # ／ジョイント

#                     # 回転量
#                     joint_axis_up = (prev_below_bone_position - next_above_bone_position).normalized()
#                     joint_axis = (next_below_bone_position - prev_above_bone_position).normalized()
#                     joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                     joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                     joint_euler = joint_rotation_qq.toEulerAngles()
#                     joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                     joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
#                                   model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
#                                   joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yi], diagonal_limit_min_mov_ys[yi], diagonal_limit_min_mov_zs[yi]), \
#                                   MVector3D(diagonal_limit_max_mov_xs[yi], diagonal_limit_max_mov_ys[yi], diagonal_limit_max_mov_zs[yi]),
#                                   MVector3D(math.radians(diagonal_limit_min_rot_xs[yi]), math.radians(diagonal_limit_min_rot_ys[yi]), math.radians(diagonal_limit_min_rot_zs[yi])),
#                                   MVector3D(math.radians(diagonal_limit_max_rot_xs[yi]), math.radians(diagonal_limit_max_rot_ys[yi]), math.radians(diagonal_limit_max_rot_zs[yi])),
#                                   MVector3D(diagonal_spring_constant_mov_xs[yi], diagonal_spring_constant_mov_ys[yi], diagonal_spring_constant_mov_zs[yi]), \
#                                   MVector3D(diagonal_spring_constant_rot_xs[yi], diagonal_spring_constant_rot_ys[yi], diagonal_spring_constant_rot_zs[yi]))   # noqa
#                     created_joints[joint_key] = joint

#                     if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                         logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                         prev_joint_cnt = len(created_joints) // 200
                    
#         logger.info("-- ジョイント: %s個目:終了", len(created_joints))

#         for joint_key in sorted(created_joints.keys()):
#             # ジョイントを登録
#             joint = created_joints[joint_key]
#             joint.index = len(model.joints)

#             if joint.name in model.joints:
#                 logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
#                 joint.name += randomname(3)

#             model.joints[joint.name] = joint

#     def create_joint(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict, root_rigidbody: RigidBody, registed_rigidbodies: dict):
#         # ジョイント生成
#         created_joints = {}

#         # 略称
#         abb_name = param_option['abb_name']
#         # 縦ジョイント情報
#         param_vertical_joint = param_option['vertical_joint']
#         # 横ジョイント情報
#         param_horizonal_joint = param_option['horizonal_joint']
#         # 斜めジョイント情報
#         param_diagonal_joint = param_option['diagonal_joint']
#         # 逆ジョイント情報
#         param_reverse_joint = param_option['reverse_joint']

#         v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
#         prev_joint_cnt = 0

#         max_vy = max(v_yidxs)
#         middle_vy = (max(v_yidxs)) * 0.3
#         min_vy = 0
#         xs = np.arange(min_vy, max_vy, step=1)
    
#         if param_vertical_joint:
#             coefficient = param_option['vertical_joint_coefficient']

#             vertical_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.x() / coefficient, param_vertical_joint.translation_limit_min.x()]])), xs)             # noqa
#             vertical_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.y() / coefficient, param_vertical_joint.translation_limit_min.y()]])), xs)             # noqa
#             vertical_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_min.z() / coefficient, param_vertical_joint.translation_limit_min.z()]])), xs)             # noqa

#             vertical_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.x() / coefficient, param_vertical_joint.translation_limit_max.x()]])), xs)             # noqa
#             vertical_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.y() / coefficient, param_vertical_joint.translation_limit_max.y()]])), xs)             # noqa
#             vertical_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_vertical_joint.translation_limit_max.z() / coefficient, param_vertical_joint.translation_limit_max.z()]])), xs)             # noqa

#             vertical_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x() / coefficient, param_vertical_joint.rotation_limit_min.x()]])), xs)             # noqa
#             vertical_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y() / coefficient, param_vertical_joint.rotation_limit_min.y()]])), xs)             # noqa
#             vertical_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z() / coefficient, param_vertical_joint.rotation_limit_min.z()]])), xs)             # noqa

#             vertical_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x() / coefficient, param_vertical_joint.rotation_limit_max.x()]])), xs)             # noqa
#             vertical_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y() / coefficient, param_vertical_joint.rotation_limit_max.y()]])), xs)             # noqa
#             vertical_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z() / coefficient, param_vertical_joint.rotation_limit_max.z()]])), xs)             # noqa

#             vertical_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x() / coefficient, param_vertical_joint.spring_constant_translation.x()]])), xs)             # noqa
#             vertical_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y() / coefficient, param_vertical_joint.spring_constant_translation.y()]])), xs)             # noqa
#             vertical_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z() / coefficient, param_vertical_joint.spring_constant_translation.z()]])), xs)             # noqa

#             vertical_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x() / coefficient, param_vertical_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             vertical_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y() / coefficient, param_vertical_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             vertical_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z() / coefficient, param_vertical_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_horizonal_joint:
#             coefficient = param_option['horizonal_joint_coefficient']

#             if param_option['bone_thinning_out']:
#                 horizonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_min.x() / coefficient, param_horizonal_joint.translation_limit_min.x()]])), xs)             # noqa
#                 horizonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_min.y() / coefficient, param_horizonal_joint.translation_limit_min.y()]])), xs)             # noqa
#                 horizonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_min.z() / coefficient, param_horizonal_joint.translation_limit_min.z()]])), xs)             # noqa

#                 horizonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_max.x() / coefficient, param_horizonal_joint.translation_limit_max.x()]])), xs)             # noqa
#                 horizonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_max.y() / coefficient, param_horizonal_joint.translation_limit_max.y()]])), xs)             # noqa
#                 horizonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                     [param_horizonal_joint.translation_limit_max.z() / coefficient, param_horizonal_joint.translation_limit_max.z()]])), xs)             # noqa
#             else:
#                 max_x = 0
#                 for yi, v_yidx in enumerate(v_yidxs):
#                     v_xidxs = list(registed_bone_indexs[v_yidx].keys())
#                     max_x = len(v_xidxs) if max_x < len(v_xidxs) else max_x

#                 x_distances = np.zeros((len(registed_bone_indexs), max_x + 1))
#                 for yi, v_yidx in enumerate(v_yidxs):
#                     v_xidxs = list(registed_bone_indexs[v_yidx].keys())
#                     if v_yidx < len(vertex_connected) and vertex_connected[v_yidx]:
#                         # 繋がってる場合、最後に最初のボーンを追加する
#                         v_xidxs += [list(registed_bone_indexs[v_yidx].keys())[0]]
#                     elif len(registed_bone_indexs[v_yidx]) > 2:
#                         # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
#                         v_xidxs += [list(registed_bone_indexs[v_yidx].keys())[-2]]

#                     for xi, (prev_v_xidx, next_v_xidx) in enumerate(zip(v_xidxs[:-1], v_xidxs[1:])):
#                         prev_v_xidx_diff = np.array(list(registed_bone_indexs[v_yidx].values())) - registed_bone_indexs[v_yidx][prev_v_xidx]
#                         prev_v_xidx = list(registed_bone_indexs[v_yidx].values())[(0 if prev_v_xidx == 0 else np.argmax(prev_v_xidx_diff))]
#                         prev_bone_name = self.get_bone_name(abb_name, v_yidx + 1, prev_v_xidx + 1)

#                         next_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidx].values())) - registed_bone_indexs[v_yidx][next_v_xidx])
#                         next_v_xidx = list(registed_bone_indexs[v_yidx].values())[(0 if next_v_xidx == 0 else np.argmin(next_v_xidx_diff))]
#                         next_bone_name = self.get_bone_name(abb_name, v_yidx + 1, next_v_xidx + 1)
                        
#                         x_distances[yi, xi] = tmp_all_bones[prev_bone_name]["bone"].position.distanceToPoint(tmp_all_bones[next_bone_name]["bone"].position)
#                 x_ratio_distances = np.array(x_distances) / (np.min(x_distances, axis=0) * 2)

#                 horizonal_limit_min_mov_xs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.x())
#                 horizonal_limit_min_mov_ys = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.y())
#                 horizonal_limit_min_mov_zs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_min.z())

#                 horizonal_limit_max_mov_xs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.x())
#                 horizonal_limit_max_mov_ys = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.y())
#                 horizonal_limit_max_mov_zs = np.nan_to_num(x_ratio_distances * param_horizonal_joint.translation_limit_max.z())

#             horizonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x() / coefficient, param_horizonal_joint.rotation_limit_min.x()]])), xs)             # noqa
#             horizonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y() / coefficient, param_horizonal_joint.rotation_limit_min.y()]])), xs)             # noqa
#             horizonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z() / coefficient, param_horizonal_joint.rotation_limit_min.z()]])), xs)             # noqa

#             horizonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x() / coefficient, param_horizonal_joint.rotation_limit_max.x()]])), xs)             # noqa
#             horizonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y() / coefficient, param_horizonal_joint.rotation_limit_max.y()]])), xs)             # noqa
#             horizonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z() / coefficient, param_horizonal_joint.rotation_limit_max.z()]])), xs)             # noqa

#             horizonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x() / coefficient, param_horizonal_joint.spring_constant_translation.x()]])), xs)             # noqa
#             horizonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y() / coefficient, param_horizonal_joint.spring_constant_translation.y()]])), xs)             # noqa
#             horizonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z() / coefficient, param_horizonal_joint.spring_constant_translation.z()]])), xs)             # noqa

#             horizonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x() / coefficient, param_horizonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             horizonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y() / coefficient, param_horizonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             horizonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z() / coefficient, param_horizonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_diagonal_joint:
#             coefficient = param_option['diagonal_joint_coefficient']

#             diagonal_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.x() / coefficient, param_diagonal_joint.translation_limit_min.x()]])), xs)             # noqa
#             diagonal_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.y() / coefficient, param_diagonal_joint.translation_limit_min.y()]])), xs)             # noqa
#             diagonal_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_min.z() / coefficient, param_diagonal_joint.translation_limit_min.z()]])), xs)             # noqa

#             diagonal_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.x() / coefficient, param_diagonal_joint.translation_limit_max.x()]])), xs)             # noqa
#             diagonal_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.y() / coefficient, param_diagonal_joint.translation_limit_max.y()]])), xs)             # noqa
#             diagonal_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_diagonal_joint.translation_limit_max.z() / coefficient, param_diagonal_joint.translation_limit_max.z()]])), xs)             # noqa

#             diagonal_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x() / coefficient, param_diagonal_joint.rotation_limit_min.x()]])), xs)             # noqa
#             diagonal_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y() / coefficient, param_diagonal_joint.rotation_limit_min.y()]])), xs)             # noqa
#             diagonal_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z() / coefficient, param_diagonal_joint.rotation_limit_min.z()]])), xs)             # noqa

#             diagonal_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x() / coefficient, param_diagonal_joint.rotation_limit_max.x()]])), xs)             # noqa
#             diagonal_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y() / coefficient, param_diagonal_joint.rotation_limit_max.y()]])), xs)             # noqa
#             diagonal_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z() / coefficient, param_diagonal_joint.rotation_limit_max.z()]])), xs)             # noqa

#             diagonal_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x() / coefficient, param_diagonal_joint.spring_constant_translation.x()]])), xs)             # noqa
#             diagonal_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y() / coefficient, param_diagonal_joint.spring_constant_translation.y()]])), xs)             # noqa
#             diagonal_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z() / coefficient, param_diagonal_joint.spring_constant_translation.z()]])), xs)             # noqa

#             diagonal_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x() / coefficient, param_diagonal_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             diagonal_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y() / coefficient, param_diagonal_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             diagonal_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z() / coefficient, param_diagonal_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         if param_reverse_joint:
#             coefficient = param_option['reverse_joint_coefficient']

#             reverse_limit_min_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.x() / coefficient, param_reverse_joint.translation_limit_min.x()]])), xs)             # noqa
#             reverse_limit_min_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.y() / coefficient, param_reverse_joint.translation_limit_min.y()]])), xs)             # noqa
#             reverse_limit_min_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_min.z() / coefficient, param_reverse_joint.translation_limit_min.z()]])), xs)             # noqa

#             reverse_limit_max_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.x() / coefficient, param_reverse_joint.translation_limit_max.x()]])), xs)             # noqa
#             reverse_limit_max_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.y() / coefficient, param_reverse_joint.translation_limit_max.y()]])), xs)             # noqa
#             reverse_limit_max_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, max_vy],
#                 [param_reverse_joint.translation_limit_max.z() / coefficient, param_reverse_joint.translation_limit_max.z()]])), xs)             # noqa

#             reverse_limit_min_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x() / coefficient, param_reverse_joint.rotation_limit_min.x()]])), xs)             # noqa
#             reverse_limit_min_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y() / coefficient, param_reverse_joint.rotation_limit_min.y()]])), xs)             # noqa
#             reverse_limit_min_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z() / coefficient, param_reverse_joint.rotation_limit_min.z()]])), xs)             # noqa

#             reverse_limit_max_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x() / coefficient, param_reverse_joint.rotation_limit_max.x()]])), xs)             # noqa
#             reverse_limit_max_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y() / coefficient, param_reverse_joint.rotation_limit_max.y()]])), xs)             # noqa
#             reverse_limit_max_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z() / coefficient, param_reverse_joint.rotation_limit_max.z()]])), xs)             # noqa

#             reverse_spring_constant_mov_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x() / coefficient, param_reverse_joint.spring_constant_translation.x()]])), xs)             # noqa
#             reverse_spring_constant_mov_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y() / coefficient, param_reverse_joint.spring_constant_translation.y()]])), xs)             # noqa
#             reverse_spring_constant_mov_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z() / coefficient, param_reverse_joint.spring_constant_translation.z()]])), xs)             # noqa

#             reverse_spring_constant_rot_xs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x() / coefficient, param_reverse_joint.spring_constant_rotation.x()]])), xs)             # noqa
#             reverse_spring_constant_rot_ys = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y() / coefficient, param_reverse_joint.spring_constant_rotation.y()]])), xs)             # noqa
#             reverse_spring_constant_rot_zs = MBezierUtils.intersect_by_x(bezier.Curve.from_nodes(np.asfortranarray([[min_vy, middle_vy, max_vy],
#                 [param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z() / coefficient, param_reverse_joint.spring_constant_rotation.z()]])), xs)             # noqa

#         for yi, (below_below_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[-2:-1], v_yidxs[-1:])):
#             # ルート剛体と先頭剛体を繋ぐジョイント
#             below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())

#             if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
#                 # 繋がってる場合、最後に最初のボーンを追加する
#                 below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
#             elif len(registed_bone_indexs[below_v_yidx]) > 2:
#                 # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
#                 below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]

#             for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
#                 prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
#                 next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
#                 below_v_yno = below_v_yidx + 1

#                 prev_above_bone_name = root_rigidbody.name
#                 prev_above_bone_position = root_rigidbody.shape_position
#                 prev_below_bone_name = self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)
#                 prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
#                 next_below_bone_name = self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)
#                 next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

#                 prev_below_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_below_v_yidx].values())) - registed_bone_indexs[below_v_yidx][prev_below_v_xidx])
#                 prev_below_below_v_xidx = list(registed_bone_indexs[below_below_v_yidx].values())[(0 if prev_below_v_xidx == 0 else np.argmin(prev_below_below_v_xidx_diff))]
#                 prev_below_below_bone_name = self.get_bone_name(abb_name, below_below_v_yidx + 1, prev_below_below_v_xidx + 1)
#                 prev_below_below_bone_position = tmp_all_bones[prev_below_below_bone_name]["bone"].position
                
#                 next_below_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_below_v_yidx].values())) - registed_bone_indexs[below_v_yidx][next_below_v_xidx])
#                 next_below_below_v_xidx = list(registed_bone_indexs[below_below_v_yidx].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_below_below_v_xidx_diff))]
#                 next_below_below_bone_name = self.get_bone_name(abb_name, below_below_v_yidx + 1, next_below_below_v_xidx + 1)
#                 next_below_below_bone_position = tmp_all_bones[next_below_below_bone_name]["bone"].position

#                 if prev_above_bone_name in model.rigidbodies and prev_below_bone_name in registed_rigidbodies:
#                     joint_name = f'↓|{prev_above_bone_name}|{registed_rigidbodies[prev_below_bone_name]}'
#                     joint_key = f'0:{model.rigidbodies[prev_above_bone_name].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                        
#                         # 縦ジョイント
#                         joint_vec = prev_below_bone_position

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (next_below_bone_position - prev_below_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         yidx = 0
#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_above_bone_name].index, \
#                                       model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yidx], vertical_limit_min_mov_ys[yidx], vertical_limit_min_mov_zs[yidx]), \
#                                       MVector3D(vertical_limit_max_mov_xs[yidx], vertical_limit_max_mov_ys[yidx], vertical_limit_max_mov_zs[yidx]),
#                                       MVector3D(math.radians(vertical_limit_min_rot_xs[yidx]), math.radians(vertical_limit_min_rot_ys[yidx]), math.radians(vertical_limit_min_rot_zs[yidx])),
#                                       MVector3D(math.radians(vertical_limit_max_rot_xs[yidx]), math.radians(vertical_limit_max_rot_ys[yidx]), math.radians(vertical_limit_max_rot_zs[yidx])),
#                                       MVector3D(vertical_spring_constant_mov_xs[yidx], vertical_spring_constant_mov_ys[yidx], vertical_spring_constant_mov_zs[yidx]), \
#                                       MVector3D(vertical_spring_constant_rot_xs[yidx], vertical_spring_constant_rot_ys[yidx], vertical_spring_constant_rot_zs[yidx]))   # noqa
#                         created_joints[joint_key] = joint

#                     # バランサー剛体が必要な場合
#                     if param_option["rigidbody_balancer"]:
#                         balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
#                         joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
#                         joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'

#                         joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
#                                       model.rigidbodies[balancer_prev_below_bone_name].index,
#                                       joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                       MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))   # noqa
#                         created_joints[joint_key] = joint

#                 # 横ジョイント
#                 if prev_below_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:
#                     joint_name = f'→|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
#                     joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                        
#                         joint_vec = np.mean([prev_below_below_bone_position, prev_below_bone_position, \
#                                              next_below_below_bone_position, next_below_bone_position])

#                         # 回転量
#                         joint_axis_up = (next_below_bone_position - prev_below_bone_position).normalized()
#                         joint_axis = (prev_below_below_bone_position - prev_below_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         yidx = 0
#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
#                                       model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi, xi], horizonal_limit_min_mov_ys[yi, xi], horizonal_limit_min_mov_zs[yi, xi]), \
#                                       MVector3D(horizonal_limit_max_mov_xs[yi, xi], horizonal_limit_max_mov_ys[yi, xi], horizonal_limit_max_mov_zs[yi, xi]),
#                                       MVector3D(math.radians(horizonal_limit_min_rot_xs[yidx]), math.radians(horizonal_limit_min_rot_ys[yidx]), math.radians(horizonal_limit_min_rot_zs[yidx])),
#                                       MVector3D(math.radians(horizonal_limit_max_rot_xs[yidx]), math.radians(horizonal_limit_max_rot_ys[yidx]), math.radians(horizonal_limit_max_rot_zs[yidx])),
#                                       MVector3D(horizonal_spring_constant_mov_xs[yidx], horizonal_spring_constant_mov_ys[yidx], horizonal_spring_constant_mov_zs[yidx]), \
#                                       MVector3D(horizonal_spring_constant_rot_xs[yidx], horizonal_spring_constant_rot_ys[yidx], horizonal_spring_constant_rot_zs[yidx]))    # noqa
#                         created_joints[joint_key] = joint

#         for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
#             below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
#             logger.debug(f"before yi: {yi}, below_v_xidxs: {below_v_xidxs}")

#             if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
#                 # 繋がってる場合、最後に最初のボーンを追加する
#                 below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]
#             elif len(registed_bone_indexs[below_v_yidx]) > 2:
#                 # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
#                 below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[-2]]
#             logger.debug(f"after yi: {yi}, below_v_xidxs: {below_v_xidxs}")

#             for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
#                 prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
#                 next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
#                 below_v_yno = below_v_yidx + 1

#                 prev_below_bone_name = self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)
#                 prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
#                 next_below_bone_name = self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)
#                 next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

#                 prev_above_bone_name = tmp_all_bones[prev_below_bone_name]["parent"]
#                 prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
                
#                 next_above_bone_name = tmp_all_bones[next_below_bone_name]["parent"]
#                 next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position
                
#                 # next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - registed_bone_indexs[below_v_yidx][next_below_v_xidx])
#                 # next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
#                 # next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
#                 # next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

#                 if param_vertical_joint and prev_above_bone_name != prev_below_bone_name and prev_above_bone_name in registed_rigidbodies and prev_below_bone_name in registed_rigidbodies:
#                     # 縦ジョイント
#                     joint_name = f'↓|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[prev_below_bone_name]}'
#                     joint_key = f'0:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                        
#                         # 縦ジョイント
#                         joint_vec = prev_below_bone_position

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (next_above_bone_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
#                         yidx = min(len(vertical_limit_min_mov_xs) - 1, yidx)

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                       model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(vertical_limit_min_mov_xs[yidx], vertical_limit_min_mov_ys[yidx], vertical_limit_min_mov_zs[yidx]), \
#                                       MVector3D(vertical_limit_max_mov_xs[yidx], vertical_limit_max_mov_ys[yidx], vertical_limit_max_mov_zs[yidx]),
#                                       MVector3D(math.radians(vertical_limit_min_rot_xs[yidx]), math.radians(vertical_limit_min_rot_ys[yidx]), math.radians(vertical_limit_min_rot_zs[yidx])),
#                                       MVector3D(math.radians(vertical_limit_max_rot_xs[yidx]), math.radians(vertical_limit_max_rot_ys[yidx]), math.radians(vertical_limit_max_rot_zs[yidx])),
#                                       MVector3D(vertical_spring_constant_mov_xs[yidx], vertical_spring_constant_mov_ys[yidx], vertical_spring_constant_mov_zs[yidx]), \
#                                       MVector3D(vertical_spring_constant_rot_xs[yidx], vertical_spring_constant_rot_ys[yidx], vertical_spring_constant_rot_zs[yidx]))   # noqa
#                         created_joints[joint_key] = joint

#                         if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                             logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                             prev_joint_cnt = len(created_joints) // 200
                        
#                         if param_reverse_joint:
#                             # 逆ジョイント
#                             joint_name = f'↑|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
#                             joint_key = f'1:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

#                             if not (joint_key in created_joints or prev_below_bone_name not in registed_rigidbodies or prev_above_bone_name not in registed_rigidbodies):
#                                 # 未登録のみ追加
#                                 joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
#                                               model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                                               joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
#                                               MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
#                                               MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
#                                               MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
#                                               MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
#                                               MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))  # noqa
#                                 created_joints[joint_key] = joint

#                                 if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                                     logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                                     prev_joint_cnt = len(created_joints) // 200

#                         # バランサー剛体が必要な場合
#                         if param_option["rigidbody_balancer"]:
#                             balancer_prev_below_bone_name = f'B-{prev_below_bone_name}'
#                             joint_name = f'B|{prev_below_bone_name}|{balancer_prev_below_bone_name}'
#                             joint_key = f'8:{model.rigidbodies[prev_below_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'   # noqa

#                             joint_vec = model.rigidbodies[prev_below_bone_name].shape_position

#                             # 回転量
#                             joint_axis_up = (prev_below_bone_position - prev_above_bone_position).normalized()
#                             joint_axis = (model.rigidbodies[balancer_prev_below_bone_name].shape_position - prev_above_bone_position).normalized()
#                             joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                             joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                             joint_euler = joint_rotation_qq.toEulerAngles()
#                             joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[prev_below_bone_name].index, \
#                                           model.rigidbodies[balancer_prev_below_bone_name].index,
#                                           joint_vec, joint_radians, MVector3D(), MVector3D(), MVector3D(), MVector3D(),
#                                           MVector3D(100000, 100000, 100000), MVector3D(100000, 100000, 100000))
#                             created_joints[joint_key] = joint

#                             # バランサー補助剛体
#                             balancer_prev_above_bone_name = f'B-{prev_above_bone_name}'
#                             joint_name = f'BS|{balancer_prev_above_bone_name}|{balancer_prev_below_bone_name}'
#                             joint_key = f'9:{model.rigidbodies[balancer_prev_above_bone_name].index:05d}:{model.rigidbodies[balancer_prev_below_bone_name].index:05d}'
#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[balancer_prev_above_bone_name].index, \
#                                           model.rigidbodies[balancer_prev_below_bone_name].index,
#                                           MVector3D(), MVector3D(), MVector3D(-50, -50, -50), MVector3D(50, 50, 50), MVector3D(math.radians(1), math.radians(1), math.radians(1)), \
#                                           MVector3D(), MVector3D(), MVector3D())
#                             created_joints[joint_key] = joint
                                                                            
#                 if param_horizonal_joint and prev_above_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:
#                     # 横ジョイント
#                     if xi < len(below_v_xidxs) - 1 and prev_above_bone_name != next_above_bone_name:
#                         joint_name = f'→|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
#                         joint_key = f'2:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

#                         if joint_key not in created_joints:
#                             # 未登録のみ追加
                            
#                             joint_vec = np.mean([prev_above_bone_position, prev_below_bone_position, \
#                                                  next_above_bone_position, next_below_bone_position])

#                             # 回転量
#                             joint_axis_up = (next_above_bone_position - prev_above_bone_position).normalized()
#                             joint_axis = (prev_below_bone_position - prev_above_bone_position).normalized()
#                             joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                             joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                             joint_euler = joint_rotation_qq.toEulerAngles()
#                             joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                             yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
#                             yidx = min(len(horizonal_limit_min_mov_xs) - 1, yidx)

#                             joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                           model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
#                                           joint_vec, joint_radians, MVector3D(horizonal_limit_min_mov_xs[yi, xi], horizonal_limit_min_mov_ys[yi, xi], horizonal_limit_min_mov_zs[yi, xi]), \
#                                           MVector3D(horizonal_limit_max_mov_xs[yi, xi], horizonal_limit_max_mov_ys[yi, xi], horizonal_limit_max_mov_zs[yi, xi]),
#                                           MVector3D(math.radians(horizonal_limit_min_rot_xs[yidx]), math.radians(horizonal_limit_min_rot_ys[yidx]), math.radians(horizonal_limit_min_rot_zs[yidx])),
#                                           MVector3D(math.radians(horizonal_limit_max_rot_xs[yidx]), math.radians(horizonal_limit_max_rot_ys[yidx]), math.radians(horizonal_limit_max_rot_zs[yidx])),
#                                           MVector3D(horizonal_spring_constant_mov_xs[yidx], horizonal_spring_constant_mov_ys[yidx], horizonal_spring_constant_mov_zs[yidx]), \
#                                           MVector3D(horizonal_spring_constant_rot_xs[yidx], horizonal_spring_constant_rot_ys[yidx], horizonal_spring_constant_rot_zs[yidx]))
#                             created_joints[joint_key] = joint

#                             if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                                 logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                                 prev_joint_cnt = len(created_joints) // 200
                            
#                         if param_reverse_joint:
#                             # 横逆ジョイント
#                             joint_name = f'←|{registed_rigidbodies[next_above_bone_name]}|{registed_rigidbodies[prev_above_bone_name]}'
#                             joint_key = f'3:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}'

#                             if not (joint_key in created_joints or prev_above_bone_name not in registed_rigidbodies or next_above_bone_name not in registed_rigidbodies):
#                                 # 未登録のみ追加
                                
#                                 joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index, \
#                                               model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index,
#                                               joint_vec, joint_radians, MVector3D(reverse_limit_min_mov_xs[yidx], reverse_limit_min_mov_ys[yidx], reverse_limit_min_mov_zs[yidx]), \
#                                               MVector3D(reverse_limit_max_mov_xs[yidx], reverse_limit_max_mov_ys[yidx], reverse_limit_max_mov_zs[yidx]),
#                                               MVector3D(math.radians(reverse_limit_min_rot_xs[yidx]), math.radians(reverse_limit_min_rot_ys[yidx]), math.radians(reverse_limit_min_rot_zs[yidx])),
#                                               MVector3D(math.radians(reverse_limit_max_rot_xs[yidx]), math.radians(reverse_limit_max_rot_ys[yidx]), math.radians(reverse_limit_max_rot_zs[yidx])),
#                                               MVector3D(reverse_spring_constant_mov_xs[yidx], reverse_spring_constant_mov_ys[yidx], reverse_spring_constant_mov_zs[yidx]), \
#                                               MVector3D(reverse_spring_constant_rot_xs[yidx], reverse_spring_constant_rot_ys[yidx], reverse_spring_constant_rot_zs[yidx]))      # noqa
#                                 created_joints[joint_key] = joint

#                                 if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                                     logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                                     prev_joint_cnt = len(created_joints) // 200
                                
#                 if param_diagonal_joint and prev_above_bone_name in registed_rigidbodies and next_below_bone_name in registed_rigidbodies:
#                     # ＼ジョイント
#                     joint_name = f'＼|{registed_rigidbodies[prev_above_bone_name]}|{registed_rigidbodies[next_below_bone_name]}'
#                     joint_key = f'4:{model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                        
#                         # ＼ジョイント
#                         joint_vec = np.mean([prev_below_bone_position, next_below_bone_position])

#                         # 回転量
#                         joint_axis_up = (next_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis = (prev_below_bone_position - next_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
#                         yidx = min(len(diagonal_limit_min_mov_xs) - 1, yidx)

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_above_bone_name]].index, \
#                                       model.rigidbodies[registed_rigidbodies[next_below_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
#                                       MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
#                                       MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
#                                       MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
#                                       MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
#                                       MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
#                         created_joints[joint_key] = joint

#                         if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                             logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                             prev_joint_cnt = len(created_joints) // 200
                        
#                 if param_diagonal_joint and prev_below_bone_name in registed_rigidbodies and next_above_bone_name in registed_rigidbodies:    # noqa
#                     # ／ジョイント ---------------
#                     joint_name = f'／|{registed_rigidbodies[prev_below_bone_name]}|{registed_rigidbodies[next_above_bone_name]}'
#                     joint_key = f'5:{model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index:05d}:{model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index:05d}'

#                     if joint_key not in created_joints:
#                         # 未登録のみ追加
                    
#                         # ／ジョイント

#                         # 回転量
#                         joint_axis_up = (prev_below_bone_position - next_above_bone_position).normalized()
#                         joint_axis = (next_below_bone_position - prev_above_bone_position).normalized()
#                         joint_axis_cross = MVector3D.crossProduct(joint_axis, joint_axis_up).normalized()
#                         joint_rotation_qq = MQuaternion.fromDirection(joint_axis, joint_axis_cross)
#                         joint_euler = joint_rotation_qq.toEulerAngles()
#                         joint_radians = MVector3D(math.radians(joint_euler.x()), math.radians(joint_euler.y()), math.radians(joint_euler.z()))

#                         yidx, _ = self.disassemble_bone_name(prev_below_bone_name)
#                         yidx = min(len(diagonal_limit_min_mov_xs) - 1, yidx)

#                         joint = Joint(joint_name, joint_name, 0, model.rigidbodies[registed_rigidbodies[prev_below_bone_name]].index, \
#                                       model.rigidbodies[registed_rigidbodies[next_above_bone_name]].index,
#                                       joint_vec, joint_radians, MVector3D(diagonal_limit_min_mov_xs[yidx], diagonal_limit_min_mov_ys[yidx], diagonal_limit_min_mov_zs[yidx]), \
#                                       MVector3D(diagonal_limit_max_mov_xs[yidx], diagonal_limit_max_mov_ys[yidx], diagonal_limit_max_mov_zs[yidx]),
#                                       MVector3D(math.radians(diagonal_limit_min_rot_xs[yidx]), math.radians(diagonal_limit_min_rot_ys[yidx]), math.radians(diagonal_limit_min_rot_zs[yidx])),
#                                       MVector3D(math.radians(diagonal_limit_max_rot_xs[yidx]), math.radians(diagonal_limit_max_rot_ys[yidx]), math.radians(diagonal_limit_max_rot_zs[yidx])),
#                                       MVector3D(diagonal_spring_constant_mov_xs[yidx], diagonal_spring_constant_mov_ys[yidx], diagonal_spring_constant_mov_zs[yidx]), \
#                                       MVector3D(diagonal_spring_constant_rot_xs[yidx], diagonal_spring_constant_rot_ys[yidx], diagonal_spring_constant_rot_zs[yidx]))   # noqa
#                         created_joints[joint_key] = joint

#                         if len(created_joints) > 0 and len(created_joints) // 200 > prev_joint_cnt:
#                             logger.info("-- ジョイント: %s個目:終了", len(created_joints))
#                             prev_joint_cnt = len(created_joints) // 200

#         logger.info("-- ジョイント: %s個目:終了", len(created_joints))

#         for joint_key in sorted(created_joints.keys()):
#             # ジョイントを登録
#             joint = created_joints[joint_key]
#             joint.index = len(model.joints)

#             if joint.name in model.joints:
#                 logger.warning("同じジョイント名が既に登録されているため、末尾に乱数を追加します。 既存ジョイント名: %s", joint.name)
#                 joint.name += randomname(3)

#             model.joints[joint.name] = joint
#             logger.debug(f"joint: {joint}")
    
#     def create_vertical_rigidbody(self, model: PmxModel, param_option: dict):
#         bone_grid_cols = param_option["bone_grid_cols"]
#         bone_grid_rows = param_option["bone_grid_rows"]
#         bone_grid = param_option["bone_grid"]

#         prev_rigidbody_cnt = 0

#         registed_rigidbodies = {}

#         # 剛体情報
#         param_rigidbody = param_option['rigidbody']
#         # 剛体係数
#         coefficient = param_option['rigidbody_coefficient']

#         # 親ボーンに紐付く剛体がある場合、それを利用
#         parent_bone = model.bones[param_option['parent_bone_name']]
#         parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
#         if not parent_bone_rigidbody:
#             # 親ボーンに紐付く剛体がない場合、自前で作成
#             parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                               0, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
#             parent_bone_rigidbody.index = len(model.rigidbodies)

#             if parent_bone_rigidbody.name in model.rigidbodies:
#                 logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
#                 parent_bone_rigidbody.name += randomname(3)

#             # 登録したボーン名と剛体の対比表を保持
#             registed_rigidbodies[model.bone_indexes[parent_bone_rigidbody.bone_index]] = parent_bone_rigidbody.name

#             model.rigidbodies[parent_bone.name] = parent_bone_rigidbody

#         target_rigidbodies = {}
#         for pac in range(bone_grid_cols):
#             target_rigidbodies[pac] = []
#             # 剛体生成
#             created_rigidbodies = {}
#             # 剛体の質量
#             created_rigidbody_masses = {}
#             created_rigidbody_linear_dampinges = {}
#             created_rigidbody_angular_dampinges = {}

#             prev_above_bone_name = None
#             prev_above_bone_position = None
#             for par in range(bone_grid_rows):
#                 prev_above_bone_name = bone_grid[par][pac]
#                 if not prev_above_bone_name or prev_above_bone_name not in model.bones:
#                     continue

#                 prev_above_bone_position = model.bones[prev_above_bone_name].position
#                 prev_above_bone_index = model.bones[prev_above_bone_name].index
#                 prev_below_bone_name = None
#                 prev_below_bone_position = None
#                 if prev_above_bone_name:
#                     pbr = par + 1
#                     if pbr in bone_grid and pac in bone_grid[pbr]:
#                         prev_below_bone_name = bone_grid[pbr][pac]
#                         if prev_below_bone_name:
#                             prev_below_bone_position = model.bones[prev_below_bone_name].position
#                         if not prev_below_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                             # 下がない場合、かつ上ボーンの相対位置がある場合、下段があると見なす
#                             prev_below_bone_name = prev_above_bone_name
#                             prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position
#                     elif prev_above_bone_name and model.bones[prev_above_bone_name].tail_position != MVector3D():
#                         prev_below_bone_name = prev_above_bone_name
#                         prev_below_bone_position = prev_above_bone_position + model.bones[prev_above_bone_name].tail_position

#                 if prev_above_bone_position and prev_below_bone_position:
#                     target_rigidbodies[pac].append(prev_above_bone_name)

#                     bone = model.bones[prev_above_bone_name]
#                     if bone.index not in model.vertices:
#                         if bone.tail_index in model.bone_indexes:
#                             # ウェイトが乗っていない場合、ボーンの長さで見る
#                             min_vertex = model.bones[model.bone_indexes[bone.tail_index]].position.data()
#                         else:
#                             min_vertex = np.array([0, 0, 0])
#                         max_vertex = bone.position.data()
#                         max_vertex[0] = 1
#                     else:
#                         # 剛体生成対象の場合のみ作成
#                         vertex_list = []
#                         normal_list = []
#                         for vertex in model.vertices[bone.index]:
#                             vertex_list.append(vertex.position.data().tolist())
#                             normal_list.append(vertex.normal.data().tolist())
#                         vertex_ary = np.array(vertex_list)
#                         min_vertex = np.min(vertex_ary, axis=0)
#                         max_vertex = np.max(vertex_ary, axis=0)

#                     axis_vec = prev_below_bone_position - bone.position
                    
#                     # 回転量
#                     rot = MQuaternion.rotationTo(MVector3D(0, 1, 0), axis_vec.normalized())
#                     shape_euler = rot.toEulerAngles()
#                     shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

#                     # サイズ
#                     diff_size = np.abs(max_vertex - min_vertex)
#                     shape_size = MVector3D(diff_size[0] * 0.3, abs(axis_vec.y() * 0.8), diff_size[2])
#                     shape_position = bone.position + (prev_below_bone_position - bone.position) / 2

#                     # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
#                     mode = 2 if par == 0 else 1
#                     shape_type = param_rigidbody.shape_type
#                     mass = param_rigidbody.param.mass
#                     linear_damping = param_rigidbody.param.linear_damping
#                     angular_damping = param_rigidbody.param.angular_damping
#                     rigidbody = RigidBody(prev_above_bone_name, prev_above_bone_name, prev_above_bone_index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                           shape_type, shape_size, shape_position, shape_rotation_radians, \
#                                           mass, linear_damping, angular_damping, param_rigidbody.param.restitution, param_rigidbody.param.friction, mode)
#                     # 別途保持しておく
#                     created_rigidbodies[rigidbody.name] = rigidbody
#                     created_rigidbody_masses[rigidbody.name] = mass
#                     created_rigidbody_linear_dampinges[rigidbody.name] = linear_damping
#                     created_rigidbody_angular_dampinges[rigidbody.name] = angular_damping
            
#             if len(created_rigidbodies) == 0:
#                 continue

#             min_mass = np.min(list(created_rigidbody_masses.values()))
#             min_linear_damping = np.min(list(created_rigidbody_linear_dampinges.values()))
#             min_angular_damping = np.min(list(created_rigidbody_angular_dampinges.values()))

#             max_mass = np.max(list(created_rigidbody_masses.values()))
#             max_linear_damping = np.max(list(created_rigidbody_linear_dampinges.values()))
#             max_angular_damping = np.max(list(created_rigidbody_angular_dampinges.values()))

#             for rigidbody_name in sorted(created_rigidbodies.keys()):
#                 # 剛体を登録
#                 rigidbody = created_rigidbodies[rigidbody_name]
#                 rigidbody.index = len(model.rigidbodies)

#                 # 質量と減衰は面積に応じた値に変換
#                 if min_mass != max_mass:
#                     rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
#                 if min_linear_damping != max_linear_damping:
#                     rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
#                         min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
#                 if min_angular_damping != max_angular_damping:
#                     rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
#                         min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa
                
#                 if rigidbody.name in model.rigidbodies:
#                     logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                     rigidbody.name += randomname(3)
            
#                 # 登録したボーン名と剛体の対比表を保持
#                 registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name

#                 model.rigidbodies[rigidbody.name] = rigidbody
            
#             prev_rigidbody_cnt += len(created_rigidbodies)

#         # バランサー剛体が必要な場合
#         if param_option["rigidbody_balancer"]:
#             # すべて非衝突対象
#             balancer_no_collision_group = 0
#             # 剛体生成
#             created_rigidbodies = {}
#             # ボーン生成
#             created_bones = {}

#             for rigidbody_params in target_rigidbodies.values():
#                 rigidbody_mass = 0
#                 rigidbody_volume = MVector3D()
#                 for org_rigidbody_name in reversed(rigidbody_params):
#                     org_rigidbody = model.rigidbodies[org_rigidbody_name]
#                     org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
#                     org_tail_position = org_bone.tail_position
#                     if org_bone.tail_index >= 0:
#                         org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
#                     org_axis = (org_tail_position - org_bone.position).normalized()

#                     if rigidbody_mass > 0:
#                         # 中間は子の1.5倍
#                         org_rigidbody.param.mass = rigidbody_mass * 1.5
#                     org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
#                                                                    math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

#                     # 名前にバランサー追加
#                     rigidbody_name = f'B-{org_rigidbody_name}'
#                     # 質量は子の1.5倍
#                     rigidbody_mass = org_rigidbody.param.mass

#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
#                         rigidbody_qq *= org_rigidbody_qq
#                     else:
#                         # 上を向いてたらX方向に反転
#                         rigidbody_qq = org_rigidbody_qq
#                         rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

#                     shape_euler = rigidbody_qq.toEulerAngles()
#                     shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

#                     # 剛体の位置は剛体の上端から反対向き
#                     mat = MMatrix4x4()
#                     mat.setToIdentity()
#                     mat.translate(org_rigidbody.shape_position)
#                     mat.rotate(org_rigidbody_qq)
#                     # X方向に反転
#                     mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     edge_pos = MVector3D()
#                     if org_rigidbody.shape_type == 0:
#                         # 球の場合、半径分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
#                     elif org_rigidbody.shape_type == 1:
#                         # 箱の場合、高さの半分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
#                     elif org_rigidbody.shape_type == 2:
#                         # カプセルの場合、高さの半分 + 半径
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

#                     mat.translate(edge_pos)
                    
#                     # 元剛体の先端位置
#                     org_rigidbody_pos = mat * MVector3D()

#                     mat2 = MMatrix4x4()
#                     mat2.setToIdentity()
#                     # 元剛体の先端位置
#                     mat2.translate(org_rigidbody_pos)
#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
#                         mat2.rotate(org_rigidbody_qq)
#                     else:
#                         # 上を向いてたらX方向に反転
#                         mat2.rotate(org_rigidbody_qq)
#                         mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     # バランサー剛体の位置
#                     shape_position = mat2 * (edge_pos + rigidbody_volume * 2)

#                     # バランサー剛体のサイズ
#                     shape_size = org_rigidbody.shape_size + (rigidbody_volume * 4)

#                     # バランサー剛体用のボーン
#                     balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
#                     created_bones[balancer_bone.name] = balancer_bone

#                     rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
#                                           2, shape_size, shape_position, shape_rotation_radians, \
#                                           rigidbody_mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
#                                           org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
#                     created_rigidbodies[rigidbody.name] = rigidbody
#                     # 子剛体のサイズを保持
#                     rigidbody_volume += edge_pos

#             for rigidbody_name in sorted(created_rigidbodies.keys()):
#                 # ボーンを登録
#                 bone = created_bones[rigidbody_name]
#                 bone.index = len(model.bones)
#                 model.bones[bone.name] = bone

#                 # 剛体を登録
#                 rigidbody = created_rigidbodies[rigidbody_name]
#                 rigidbody.bone_index = bone.index
#                 rigidbody.index = len(model.rigidbodies)

#                 if rigidbody.name in model.rigidbodies:
#                     logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                     rigidbody.name += randomname(3)

#                 # 登録したボーン名と剛体の対比表を保持
#                 registed_rigidbodies[rigidbody_name] = rigidbody.name

#                 model.rigidbodies[rigidbody.name] = rigidbody
   
#         logger.info("-- 剛体: %s個目:終了", prev_rigidbody_cnt)

#         return parent_bone_rigidbody, registed_rigidbodies

#     def create_rigidbody_by_bone_blocks(self, model: PmxModel, param_option: dict, bone_blocks: dict):
#         # bone_grid_cols = param_option["bone_grid_cols"]
#         bone_grid_rows = param_option["bone_grid_rows"]
#         # bone_grid = param_option["bone_grid"]

#         # 剛体生成
#         registed_rigidbodies = {}
#         created_rigidbodies = {}
#         # 剛体の質量
#         created_rigidbody_masses = {}
#         created_rigidbody_linear_dampinges = {}
#         created_rigidbody_angular_dampinges = {}
#         prev_rigidbody_cnt = 0

#         # 剛体情報
#         param_rigidbody = param_option['rigidbody']
#         # 剛体係数
#         coefficient = param_option['rigidbody_coefficient']
#         # 剛体形状
#         rigidbody_shape_type = param_option["rigidbody_shape_type"]
#         # 物理タイプ
#         physics_type = param_option["physics_type"]

#         rigidbody_limit_thicks = np.linspace(0.1, 0.3, bone_grid_rows)

#         # 親ボーンに紐付く剛体がある場合、それを利用
#         parent_bone = model.bones[param_option['parent_bone_name']]
#         parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
#         if not parent_bone_rigidbody:
#             # 親ボーンに紐付く剛体がない場合、自前で作成
#             parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                               parent_bone_rigidbody.shape_type, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
#             parent_bone_rigidbody.index = len(model.rigidbodies)

#             if parent_bone_rigidbody.name in model.rigidbodies:
#                 logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
#                 parent_bone_rigidbody.name += randomname(3)

#             model.rigidbodies[parent_bone.name] = parent_bone_rigidbody
        
#         # 略称
#         abb_name = param_option['abb_name']
#         root_rigidbody_name = f'{abb_name}中心'
        
#         # 中心剛体を接触なしボーン追従剛体で生成
#         root_rigidbody = None if root_rigidbody_name not in model.rigidbodies else model.rigidbodies[root_rigidbody_name]
#         if not root_rigidbody:
#             root_rigidbody = RigidBody(root_rigidbody_name, root_rigidbody_name, parent_bone.index, param_rigidbody.collision_group, 0, \
#                                        parent_bone_rigidbody.shape_type, parent_bone_rigidbody.shape_size * 0.5, parent_bone_rigidbody.shape_position, \
#                                        parent_bone_rigidbody.shape_rotation, 1, 0.5, 0.5, 0, 0, 0)
#             root_rigidbody.index = len(model.rigidbodies)
#             model.rigidbodies[root_rigidbody.name] = root_rigidbody

#         # 登録したボーン名と剛体の対比表を保持
#         registed_rigidbodies[model.bone_indexes[parent_bone.index]] = root_rigidbody.name

#         target_rigidbodies = {}
#         for bone_block in bone_blocks.values():
#             prev_above_bone_name = bone_block['prev_above']
#             prev_above_bone_position = bone_block['prev_above_pos']
#             # prev_below_bone_name = bone_block['prev_below']
#             prev_below_bone_position = bone_block['prev_below_pos']
#             # next_above_bone_name = bone_block['next_above']
#             next_above_bone_position = bone_block['next_above_pos']
#             # next_below_bone_name = bone_block['next_below']
#             next_below_bone_position = bone_block['next_below_pos']
#             # prev_prev_above_bone_name = bone_block['prev_prev_above']
#             prev_prev_above_bone_position = bone_block['prev_prev_above_pos']
#             xi = bone_block['xi']
#             yi = bone_block['yi']
#             is_above_connected = bone_block['is_above_connected']

#             if prev_above_bone_name in created_rigidbodies:
#                 continue

#             prev_above_bone_index = -1
#             if prev_above_bone_name in model.bones:
#                 prev_above_bone_index = model.bones[prev_above_bone_name].index
#             else:
#                 continue
            
#             if xi not in target_rigidbodies:
#                 target_rigidbodies[xi] = []

#             target_rigidbodies[xi].append(prev_above_bone_name)

#             # 剛体の傾き
#             shape_axis = (prev_below_bone_position - prev_above_bone_position).round(5).normalized()
#             shape_axis_up = (next_above_bone_position - prev_prev_above_bone_position).round(5).normalized()
#             shape_axis_cross = MVector3D.crossProduct(shape_axis, shape_axis_up).round(5).normalized()

#             # if shape_axis_up.round(2) == MVector3D(1, 0, 0):
#             #     shape_rotation_qq = MQuaternion.fromEulerAngles(0, 180, 0)
#             # else:
#             #     shape_rotation_qq = MQuaternion.rotationTo(MVector3D(-1, 0, 0), shape_axis_up)

#             shape_rotation_qq = MQuaternion.fromDirection(shape_axis, shape_axis_cross)
#             if round(prev_below_bone_position.y(), 2) != round(prev_above_bone_position.y(), 2):
#                 shape_rotation_qq *= MQuaternion.fromEulerAngles(0, 0, -90)
#                 shape_rotation_qq *= MQuaternion.fromEulerAngles(-90, 0, 0)
#                 if is_above_connected:
#                     shape_rotation_qq *= MQuaternion.fromEulerAngles(0, -90, 0)

#             shape_rotation_euler = shape_rotation_qq.toEulerAngles()

#             if round(prev_below_bone_position.y(), 2) == round(prev_above_bone_position.y(), 2):
#                 shape_rotation_euler.setX(90)
                
#             shape_rotation_radians = MVector3D(math.radians(shape_rotation_euler.x()), math.radians(shape_rotation_euler.y()), math.radians(shape_rotation_euler.z()))

#             # 剛体の大きさ
#             if rigidbody_shape_type == 0:
#                 x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position), \
#                                  prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                 ball_size = max(0.25, x_size * 0.5)
#                 shape_size = MVector3D(ball_size, ball_size, ball_size)
#             elif rigidbody_shape_type == 2:
#                 x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
#                 y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                 if physics_type == logger.transtext('袖'):
#                     shape_size = MVector3D(x_size * 0.4, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
#                 else:
#                     shape_size = MVector3D(x_size * 0.5, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
#             else:
#                 x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
#                 y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                 shape_size = MVector3D(max(0.25, x_size * 0.55), max(0.25, y_size * 0.55), rigidbody_limit_thicks[yi])

#             # 剛体の位置
#             rigidbody_vertical_vec = ((prev_below_bone_position - prev_above_bone_position) / 2)
#             if round(prev_below_bone_position.y(), 3) != round(prev_above_bone_position.y(), 3):
#                 mat = MMatrix4x4()
#                 mat.setToIdentity()
#                 mat.translate(prev_above_bone_position)
#                 mat.rotate(shape_rotation_qq)
#                 # ローカルY軸方向にボーンの長さの半分を上げる
#                 mat.translate(MVector3D(0, -prev_below_bone_position.distanceToPoint(prev_above_bone_position) / 2, 0))
#                 shape_position = mat * MVector3D()
#             else:
#                 shape_position = prev_above_bone_position + rigidbody_vertical_vec + MVector3D(0, rigidbody_limit_thicks[yi] / 2, 0)

#             # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
#             mode = 2 if yi == 0 else 1
#             shape_type = param_rigidbody.shape_type
#             if prev_above_bone_name not in model.bones:
#                 # 登録ボーンの対象外である場合、余っているので球にしておく
#                 ball_size = np.max([0.25, x_size * 0.5, y_size * 0.5])
#                 shape_size = MVector3D(ball_size, ball_size, ball_size)
#                 shape_type = 0
#             mass = param_rigidbody.param.mass * shape_size.x() * shape_size.y() * shape_size.z()
#             linear_damping = param_rigidbody.param.linear_damping * shape_size.x() * shape_size.y() * shape_size.z()
#             angular_damping = param_rigidbody.param.angular_damping * shape_size.x() * shape_size.y() * shape_size.z()
#             rigidbody = RigidBody(prev_above_bone_name, prev_above_bone_name, prev_above_bone_index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                   shape_type, shape_size, shape_position, shape_rotation_radians, \
#                                   mass, linear_damping, angular_damping, param_rigidbody.param.restitution, param_rigidbody.param.friction, mode)
#             # 別途保持しておく
#             created_rigidbodies[rigidbody.name] = rigidbody
#             created_rigidbody_masses[rigidbody.name] = mass
#             created_rigidbody_linear_dampinges[rigidbody.name] = linear_damping
#             created_rigidbody_angular_dampinges[rigidbody.name] = angular_damping

#             if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 200 > prev_rigidbody_cnt:
#                 logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))
#                 prev_rigidbody_cnt = len(created_rigidbodies) // 200

#         min_mass = 0
#         min_linear_damping = 0
#         min_angular_damping = 0

#         max_mass = 0
#         max_linear_damping = 0
#         max_angular_damping = 0
        
#         if len(created_rigidbody_masses.values()) > 0:
#             min_mass = np.min(list(created_rigidbody_masses.values()))
#             min_linear_damping = np.min(list(created_rigidbody_linear_dampinges.values()))
#             min_angular_damping = np.min(list(created_rigidbody_angular_dampinges.values()))

#             max_mass = np.max(list(created_rigidbody_masses.values()))
#             max_linear_damping = np.max(list(created_rigidbody_linear_dampinges.values()))
#             max_angular_damping = np.max(list(created_rigidbody_angular_dampinges.values()))

#         for rigidbody_name in sorted(created_rigidbodies.keys()):
#             # 剛体を登録
#             rigidbody = created_rigidbodies[rigidbody_name]
#             rigidbody.index = len(model.rigidbodies)

#             # 質量と減衰は面積に応じた値に変換
#             if min_mass != max_mass:
#                 rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
#             if min_linear_damping != max_linear_damping:
#                 rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
#                     min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
#             if min_angular_damping != max_angular_damping:
#                 rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
#                     min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa

#             if rigidbody.name in model.rigidbodies:
#                 logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                 rigidbody.name += randomname(3)

#             # 登録したボーン名と剛体の対比表を保持
#             registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name
            
#             model.rigidbodies[rigidbody.name] = rigidbody

#         logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

#         # バランサー剛体が必要な場合
#         if param_option["rigidbody_balancer"]:
#             # すべて非衝突対象
#             balancer_no_collision_group = 0
#             # 剛体生成
#             created_rigidbodies = {}
#             # ボーン生成
#             created_bones = {}

#             for rigidbody_params in target_rigidbodies.values():
#                 rigidbody_mass = 0
#                 rigidbody_volume = MVector3D()
#                 for org_rigidbody_name in reversed(rigidbody_params):
#                     org_rigidbody = model.rigidbodies[org_rigidbody_name]
#                     org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
#                     org_tail_position = org_bone.tail_position
#                     if org_bone.tail_index >= 0:
#                         org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
#                     org_axis = (org_tail_position - org_bone.position).normalized()

#                     if rigidbody_mass > 0:
#                         # 中間は子の1.5倍
#                         org_rigidbody.param.mass = rigidbody_mass * 1.5
#                     org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
#                                                                    math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

#                     # 名前にバランサー追加
#                     rigidbody_name = f'B-{org_rigidbody_name}'
#                     # 質量は子の1.5倍
#                     rigidbody_mass = org_rigidbody.param.mass

#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
#                         rigidbody_qq *= org_rigidbody_qq
#                     else:
#                         # 上を向いてたらX方向に反転
#                         rigidbody_qq = org_rigidbody_qq
#                         rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

#                     shape_euler = rigidbody_qq.toEulerAngles()
#                     shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

#                     # 剛体の位置は剛体の上端から反対向き
#                     mat = MMatrix4x4()
#                     mat.setToIdentity()
#                     mat.translate(org_rigidbody.shape_position)
#                     mat.rotate(org_rigidbody_qq)
#                     # X方向に反転
#                     mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     edge_pos = MVector3D()
#                     if org_rigidbody.shape_type == 0:
#                         # 球の場合、半径分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
#                     elif org_rigidbody.shape_type == 1:
#                         # 箱の場合、高さの半分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
#                     elif org_rigidbody.shape_type == 2:
#                         # カプセルの場合、高さの半分 + 半径
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

#                     mat.translate(-edge_pos)
                    
#                     # 元剛体の先端位置
#                     org_rigidbody_pos = mat * MVector3D()

#                     mat2 = MMatrix4x4()
#                     mat2.setToIdentity()
#                     # 元剛体の先端位置
#                     mat2.translate(org_rigidbody_pos)
#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
#                         mat2.rotate(org_rigidbody_qq)
#                     else:
#                         # 上を向いてたらX方向に反転
#                         mat2.rotate(org_rigidbody_qq)
#                         mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     # バランサー剛体の位置
#                     shape_position = mat2 * (-edge_pos - rigidbody_volume * 4)

#                     # バランサー剛体のサイズ
#                     shape_size = org_rigidbody.shape_size + (rigidbody_volume * 8)
#                     if org_rigidbody.shape_type != 2:
#                         shape_size.setX(0.3)

#                     # バランサー剛体用のボーン
#                     balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
#                     created_bones[balancer_bone.name] = balancer_bone

#                     rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
#                                           2, shape_size, shape_position, shape_rotation_radians, \
#                                           rigidbody_mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
#                                           org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
#                     created_rigidbodies[rigidbody.name] = rigidbody
#                     # 子剛体のサイズを保持
#                     rigidbody_volume += edge_pos

#             for rigidbody_name in sorted(created_rigidbodies.keys()):
#                 # ボーンを登録
#                 bone = created_bones[rigidbody_name]
#                 bone.index = len(model.bones)
#                 model.bones[bone.name] = bone

#                 # 剛体を登録
#                 rigidbody = created_rigidbodies[rigidbody_name]
#                 rigidbody.bone_index = bone.index
#                 rigidbody.index = len(model.rigidbodies)

#                 if rigidbody.name in model.rigidbodies:
#                     logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                     rigidbody.name += randomname(3)

#                 # 登録したボーン名と剛体の対比表を保持
#                 registed_rigidbodies[rigidbody_name] = rigidbody.name
                
#                 model.rigidbodies[rigidbody.name] = rigidbody

#         logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

#         return root_rigidbody, registed_rigidbodies

#     def create_rigidbody(self, model: PmxModel, param_option: dict, vertex_connected: dict, tmp_all_bones: dict, registed_bone_indexs: dict, root_bone: Bone):
#         # 剛体生成
#         registed_rigidbodies = {}
#         created_rigidbodies = {}
#         # 剛体の質量
#         created_rigidbody_masses = {}
#         created_rigidbody_linear_dampinges = {}
#         created_rigidbody_angular_dampinges = {}
#         prev_rigidbody_cnt = 0

#         # 略称
#         abb_name = param_option['abb_name']
#         # 剛体情報
#         param_rigidbody = param_option['rigidbody']
#         # 剛体係数
#         coefficient = param_option['rigidbody_coefficient']
#         # 剛体形状
#         rigidbody_shape_type = param_option["rigidbody_shape_type"]
#         # 物理タイプ
#         physics_type = param_option["physics_type"]

#         # 親ボーンに紐付く剛体がある場合、それを利用
#         parent_bone = model.bones[param_option['parent_bone_name']]
#         parent_bone_rigidbody = self.get_rigidbody(model, parent_bone.name)
        
#         if not parent_bone_rigidbody:
#             # 親ボーンに紐付く剛体がない場合、自前で作成
#             parent_bone_rigidbody = RigidBody(parent_bone.name, parent_bone.english_name, parent_bone.index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                               0, MVector3D(1, 1, 1), parent_bone.position, MVector3D(), 1, 0.5, 0.5, 0, 0, 0)
#             parent_bone_rigidbody.index = len(model.rigidbodies)

#             if parent_bone_rigidbody.name in model.rigidbodies:
#                 logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", parent_bone_rigidbody.name)
#                 parent_bone_rigidbody.name += randomname(3)

#             # 登録したボーン名と剛体の対比表を保持
#             registed_rigidbodies[model.bone_indexes[parent_bone_rigidbody.bone_index]] = parent_bone_rigidbody.name

#             model.rigidbodies[parent_bone.name] = parent_bone_rigidbody
        
#         root_rigidbody = self.get_rigidbody(model, root_bone.name)
#         if not root_rigidbody:
#             # 中心剛体を接触なしボーン追従剛体で生成
#             root_rigidbody = RigidBody(root_bone.name, root_bone.english_name, root_bone.index, param_rigidbody.collision_group, 0, \
#                                        parent_bone_rigidbody.shape_type, parent_bone_rigidbody.shape_size, parent_bone_rigidbody.shape_position, \
#                                        parent_bone_rigidbody.shape_rotation, 1, 0.5, 0.5, 0, 0, 0)
#             root_rigidbody.index = len(model.rigidbodies)
#             model.rigidbodies[root_rigidbody.name] = root_rigidbody

#         # 登録したボーン名と剛体の対比表を保持
#         registed_rigidbodies[model.bone_indexes[root_bone.index]] = root_rigidbody.name

#         v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
#         rigidbody_limit_thicks = np.linspace(0.3, 0.1, len(v_yidxs))

#         target_rigidbodies = {}
#         for yi, (above_v_yidx, below_v_yidx) in enumerate(zip(v_yidxs[1:], v_yidxs[:-1])):
#             above_v_xidxs = list(registed_bone_indexs[above_v_yidx].keys())
#             logger.debug(f"yi: {yi}, above_v_xidxs: {above_v_xidxs}")

#             if above_v_yidx < len(vertex_connected) and vertex_connected[above_v_yidx]:
#                 # 繋がってる場合、最後に最初のボーンを追加する
#                 above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[0]]
#             elif len(registed_bone_indexs[above_v_yidx]) > 2:
#                 # 繋がってない場合、最後に最後のひとつ前のボーンを追加する
#                 above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[-2]]
#             logger.debug(f"yi: {yi}, above_v_xidxs: {above_v_xidxs}")

#             target_rigidbodies[yi] = []

#             for xi, (prev_above_vxidx, next_above_vxidx) in enumerate(zip(above_v_xidxs[:-1], above_v_xidxs[1:])):
#                 prev_above_v_xidx = registed_bone_indexs[above_v_yidx][prev_above_vxidx]
#                 prev_above_v_xno = prev_above_v_xidx + 1
#                 next_above_v_xidx = registed_bone_indexs[above_v_yidx][next_above_vxidx]
#                 next_above_v_xno = next_above_v_xidx + 1
#                 above_v_yno = above_v_yidx + 1

#                 prev_above_bone_name = self.get_bone_name(abb_name, above_v_yno, prev_above_v_xno)
#                 prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
#                 next_above_bone_name = self.get_bone_name(abb_name, above_v_yno, next_above_v_xno)
#                 next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

#                 prev_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, prev_above_v_xidx + 1)
#                 if prev_below_bone_name not in tmp_all_bones:
#                     prev_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_v_yidx].values())) - registed_bone_indexs[above_v_yidx][prev_above_vxidx])
#                     prev_below_v_xidx = list(registed_bone_indexs[below_v_yidx].values())[(0 if prev_above_vxidx == 0 else np.argmin(prev_below_v_xidx_diff))]
#                     prev_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, prev_below_v_xidx + 1)
#                 prev_below_bone_position = tmp_all_bones[prev_below_bone_name]["bone"].position
                
#                 next_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, next_above_v_xidx + 1)
#                 if next_below_bone_name not in tmp_all_bones:
#                     next_below_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[below_v_yidx].values())) - registed_bone_indexs[above_v_yidx][next_above_vxidx])
#                     next_below_v_xidx = list(registed_bone_indexs[below_v_yidx].values())[(0 if next_above_vxidx == 0 else np.argmin(next_below_v_xidx_diff))]
#                     next_below_bone_name = self.get_bone_name(abb_name, below_v_yidx + 1, next_below_v_xidx + 1)

#                 next_below_bone_position = tmp_all_bones[next_below_bone_name]["bone"].position

#                 # prev_above_bone_name = tmp_all_bones[prev_below_bone_name]["parent"]
#                 # prev_above_bone_position = tmp_all_bones[prev_above_bone_name]["bone"].position
#                 # prev_above_v_yidx, _ = self.disassemble_bone_name(prev_above_bone_name)
    
#                 # next_above_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi + 1]].values())) - next_below_v_xidx)
#                 # next_above_v_xidx = list(registed_bone_indexs[v_yidxs[yi + 1]].values())[(0 if next_below_v_xidx == 0 else np.argmin(next_above_v_xidx_diff))]
#                 # next_above_bone_name = self.get_bone_name(abb_name, prev_above_v_yidx + 1, next_above_v_xidx + 1)
#                 # next_above_bone_position = tmp_all_bones[next_above_bone_name]["bone"].position

#                 prev_prev_above_bone_position = None
#                 if 0 == xi:
#                     # 先頭の場合、繋がっていたら最後のを加える
#                     if vertex_connected[above_v_yidx]:
#                         prev_prev_above_v_xidx = list(registed_bone_indexs[above_v_yidx].keys())[-1]
#                         prev_prev_above_bone_name = self.get_bone_name(abb_name, above_v_yidx + 1, prev_prev_above_v_xidx + 1)
#                         if prev_prev_above_bone_name in tmp_all_bones:
#                             prev_prev_above_bone_position = tmp_all_bones[prev_prev_above_bone_name]["bone"].position
#                 else:
#                     prev_prev_above_v_xidx = registed_bone_indexs[above_v_yidx][above_v_xidxs[xi - 1]]
#                     prev_prev_above_bone_name = self.get_bone_name(abb_name, above_v_yidx + 1, prev_prev_above_v_xidx + 1)
#                     if prev_prev_above_bone_name in tmp_all_bones:
#                         prev_prev_above_bone_position = tmp_all_bones[prev_prev_above_bone_name]["bone"].position
                
#                 if prev_above_bone_name in created_rigidbodies or (prev_above_bone_name in model.bones and not model.bones[prev_above_bone_name].getVisibleFlag()):
#                     continue

#                 prev_above_bone_index = -1
#                 if prev_above_bone_name in model.bones:
#                     prev_above_bone_index = model.bones[prev_above_bone_name].index

#                 target_rigidbodies[yi].append(prev_above_bone_name)

#                 # 剛体の傾き
#                 shape_axis = (prev_below_bone_position - prev_above_bone_position).round(5).normalized()
#                 if prev_prev_above_bone_position:
#                     shape_axis_up = (next_above_bone_position - prev_prev_above_bone_position).round(5).normalized()
#                 else:
#                     shape_axis_up = (next_above_bone_position - prev_above_bone_position).round(5).normalized()
#                 shape_axis_cross = MVector3D.crossProduct(shape_axis, shape_axis_up).round(5).normalized()

#                 shape_rotation_qq = MQuaternion.fromDirection(shape_axis, shape_axis_cross)
#                 if round(prev_below_bone_position.y(), 2) != round(prev_above_bone_position.y(), 2):
#                     shape_rotation_qq *= MQuaternion.fromEulerAngles(0, 0, -90)
#                     shape_rotation_qq *= MQuaternion.fromEulerAngles(-90, 0, 0)
#                     shape_rotation_qq *= MQuaternion.fromEulerAngles(0, -90, 0)

#                 shape_rotation_euler = shape_rotation_qq.toEulerAngles()

#                 if round(prev_below_bone_position.y(), 2) == round(prev_above_bone_position.y(), 2):
#                     shape_rotation_euler.setX(90)
                    
#                 shape_rotation_radians = MVector3D(math.radians(shape_rotation_euler.x()), math.radians(shape_rotation_euler.y()), math.radians(shape_rotation_euler.z()))

#                 # 剛体の大きさ
#                 if rigidbody_shape_type == 0:
#                     x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position), \
#                                      prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                     ball_size = max(0.25, x_size * 0.5)
#                     shape_size = MVector3D(ball_size, ball_size, ball_size)
#                 elif rigidbody_shape_type == 2:
#                     x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
#                     y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                     if physics_type == logger.transtext('袖'):
#                         shape_size = MVector3D(x_size * 0.4, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
#                     else:
#                         shape_size = MVector3D(x_size * 0.5, max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])
#                 else:
#                     x_size = np.max([prev_below_bone_position.distanceToPoint(next_below_bone_position), prev_above_bone_position.distanceToPoint(next_above_bone_position)])
#                     y_size = np.max([prev_below_bone_position.distanceToPoint(prev_above_bone_position), next_below_bone_position.distanceToPoint(next_above_bone_position)])
#                     shape_size = MVector3D(max(0.25, x_size * 0.5), max(0.25, y_size * 0.5), rigidbody_limit_thicks[yi])

#                 # 剛体の位置
#                 rigidbody_vertical_vec = ((prev_below_bone_position - prev_above_bone_position) / 2)
#                 if round(prev_below_bone_position.y(), 3) != round(prev_above_bone_position.y(), 3):
#                     mat = MMatrix4x4()
#                     mat.setToIdentity()
#                     mat.translate(prev_above_bone_position)
#                     mat.rotate(shape_rotation_qq)
#                     # ローカルY軸方向にボーンの長さの半分を上げる
#                     mat.translate(MVector3D(0, -prev_below_bone_position.distanceToPoint(prev_above_bone_position) / 2, 0))
#                     shape_position = mat * MVector3D()
#                 else:
#                     shape_position = prev_above_bone_position + rigidbody_vertical_vec + MVector3D(0, rigidbody_limit_thicks[yi] / 2, 0)

#                 # 根元は物理演算 + Bone位置合わせ、それ以降は物理剛体
#                 mode = 2 if yi == len(v_yidxs) - 2 else 1
#                 shape_type = param_rigidbody.shape_type
#                 if prev_above_bone_name not in model.bones:
#                     # 登録ボーンの対象外である場合、余っているので球にしておく
#                     ball_size = np.max([0.25, x_size * 0.5, y_size * 0.5])
#                     shape_size = MVector3D(ball_size, ball_size, ball_size)
#                     shape_type = 0
#                 mass = param_rigidbody.param.mass * shape_size.x() * shape_size.y() * shape_size.z()
#                 linear_damping = param_rigidbody.param.linear_damping * shape_size.x() * shape_size.y() * shape_size.z()
#                 angular_damping = param_rigidbody.param.angular_damping * shape_size.x() * shape_size.y() * shape_size.z()
#                 rigidbody = RigidBody(prev_above_bone_name, prev_above_bone_name, prev_above_bone_index, param_rigidbody.collision_group, param_rigidbody.no_collision_group, \
#                                       shape_type, shape_size, shape_position, shape_rotation_radians, \
#                                       mass, linear_damping, angular_damping, param_rigidbody.param.restitution, param_rigidbody.param.friction, mode)
#                 # 別途保持しておく
#                 created_rigidbodies[rigidbody.name] = rigidbody
#                 created_rigidbody_masses[rigidbody.name] = mass
#                 created_rigidbody_linear_dampinges[rigidbody.name] = linear_damping
#                 created_rigidbody_angular_dampinges[rigidbody.name] = angular_damping

#                 if len(created_rigidbodies) > 0 and len(created_rigidbodies) // 200 > prev_rigidbody_cnt:
#                     logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))
#                     prev_rigidbody_cnt = len(created_rigidbodies) // 200
        
#         min_mass = 0
#         min_linear_damping = 0
#         min_angular_damping = 0

#         max_mass = 0
#         max_linear_damping = 0
#         max_angular_damping = 0
        
#         if len(created_rigidbody_masses.values()) > 0:
#             min_mass = np.min(list(created_rigidbody_masses.values()))
#             min_linear_damping = np.min(list(created_rigidbody_linear_dampinges.values()))
#             min_angular_damping = np.min(list(created_rigidbody_angular_dampinges.values()))

#             max_mass = np.max(list(created_rigidbody_masses.values()))
#             max_linear_damping = np.max(list(created_rigidbody_linear_dampinges.values()))
#             max_angular_damping = np.max(list(created_rigidbody_angular_dampinges.values()))

#         for rigidbody_name in sorted(created_rigidbodies.keys()):
#             # 剛体を登録
#             rigidbody = created_rigidbodies[rigidbody_name]
#             rigidbody.index = len(model.rigidbodies)

#             # 質量と減衰は面積に応じた値に変換
#             if min_mass != max_mass:
#                 rigidbody.param.mass = calc_ratio(rigidbody.param.mass, max_mass, min_mass, param_rigidbody.param.mass, param_rigidbody.param.mass * coefficient)
#             if min_linear_damping != max_linear_damping:
#                 rigidbody.param.linear_damping = calc_ratio(rigidbody.param.linear_damping, max_linear_damping, min_linear_damping, param_rigidbody.param.linear_damping, \
#                     min(0.9999999, param_rigidbody.param.linear_damping * coefficient))     # noqa
#             if min_angular_damping != max_angular_damping:
#                 rigidbody.param.angular_damping = calc_ratio(rigidbody.param.angular_damping, max_angular_damping, min_angular_damping, param_rigidbody.param.angular_damping, \
#                     min(0.9999999, param_rigidbody.param.angular_damping * coefficient))    # noqa

#             if rigidbody.name in model.rigidbodies:
#                 logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                 rigidbody.name += randomname(3)

#             # 登録したボーン名と剛体の対比表を保持
#             registed_rigidbodies[model.bone_indexes[rigidbody.bone_index]] = rigidbody.name
            
#             model.rigidbodies[rigidbody.name] = rigidbody
#             logger.debug(f"rigidbody: {rigidbody}")

#         logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

#         # バランサー剛体が必要な場合
#         if param_option["rigidbody_balancer"]:
#             # すべて非衝突対象
#             balancer_no_collision_group = 0
#             # 剛体生成
#             created_rigidbodies = {}
#             # ボーン生成
#             created_bones = {}

#             rigidbody_volume = MVector3D()
#             rigidbody_mass = 0
#             for yi in sorted(target_rigidbodies.keys()):
#                 rigidbody_params = target_rigidbodies[yi]
#                 for org_rigidbody_name in rigidbody_params:
#                     org_rigidbody = model.rigidbodies[org_rigidbody_name]
#                     org_bone = model.bones[model.bone_indexes[org_rigidbody.bone_index]]
#                     org_tail_position = org_bone.tail_position
#                     if org_bone.tail_index >= 0:
#                         org_tail_position = model.bones[model.bone_indexes[org_bone.tail_index]].position
#                     org_axis = (org_tail_position - org_bone.position).normalized()

#                     if rigidbody_mass > 0:
#                         # 中間は子の1.5倍
#                         org_rigidbody.param.mass = rigidbody_mass * 1.5
#                     org_rigidbody_qq = MQuaternion.fromEulerAngles(math.degrees(org_rigidbody.shape_rotation.x()), \
#                                                                    math.degrees(org_rigidbody.shape_rotation.y()), math.degrees(org_rigidbody.shape_rotation.z()))

#                     # 名前にバランサー追加
#                     rigidbody_name = f'B-{org_rigidbody_name}'

#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         rigidbody_qq = MQuaternion.fromEulerAngles(0, 180, 0)
#                         rigidbody_qq *= org_rigidbody_qq
#                     else:
#                         # 上を向いてたらX方向に反転
#                         rigidbody_qq = org_rigidbody_qq
#                         rigidbody_qq *= MQuaternion.fromEulerAngles(180, 0, 0)

#                     shape_euler = rigidbody_qq.toEulerAngles()
#                     shape_rotation_radians = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

#                     # 剛体の位置は剛体の上端から反対向き
#                     mat = MMatrix4x4()
#                     mat.setToIdentity()
#                     mat.translate(org_rigidbody.shape_position)
#                     mat.rotate(org_rigidbody_qq)
#                     # X方向に反転
#                     mat.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     edge_pos = MVector3D()
#                     if org_rigidbody.shape_type == 0:
#                         # 球の場合、半径分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.x(), 0)
#                     elif org_rigidbody.shape_type == 1:
#                         # 箱の場合、高さの半分移動
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2, 0)
#                     elif org_rigidbody.shape_type == 2:
#                         # カプセルの場合、高さの半分 + 半径
#                         edge_pos = MVector3D(0, org_rigidbody.shape_size.y() / 2 + org_rigidbody.shape_size.x(), 0)

#                     mat.translate(-edge_pos)
                    
#                     # 元剛体の先端位置
#                     org_rigidbody_pos = mat * MVector3D()

#                     mat2 = MMatrix4x4()
#                     mat2.setToIdentity()
#                     # 元剛体の先端位置
#                     mat2.translate(org_rigidbody_pos)
#                     if org_axis.y() < 0:
#                         # 下を向いてたらY方向に反転
#                         mat2.rotate(MQuaternion.fromEulerAngles(0, 180, 0))
#                         mat2.rotate(org_rigidbody_qq)
#                     else:
#                         # 上を向いてたらX方向に反転
#                         mat2.rotate(org_rigidbody_qq)
#                         mat2.rotate(MQuaternion.fromEulerAngles(180, 0, 0))

#                     # バランサー剛体の位置
#                     shape_position = mat2 * (-edge_pos - rigidbody_volume * 4)

#                     # バランサー剛体のサイズ
#                     shape_size = org_rigidbody.shape_size + (rigidbody_volume * 8)
#                     if org_rigidbody.shape_type != 2:
#                         shape_size.setX(0.3)

#                     # バランサー剛体用のボーン
#                     balancer_bone = Bone(rigidbody_name, rigidbody_name, shape_position, org_rigidbody.bone_index, 0, 0x0002)
#                     created_bones[balancer_bone.name] = balancer_bone

#                     rigidbody = RigidBody(rigidbody_name, rigidbody_name, -1, org_rigidbody.collision_group, balancer_no_collision_group, \
#                                           2, shape_size, shape_position, shape_rotation_radians, \
#                                           org_rigidbody.param.mass, org_rigidbody.param.linear_damping, org_rigidbody.param.angular_damping, \
#                                           org_rigidbody.param.restitution, org_rigidbody.param.friction, 1)
#                     created_rigidbodies[rigidbody.name] = rigidbody
#                 # 子剛体のサイズを保持
#                 rigidbody_volume += edge_pos
#                 # 質量は子の1.5倍
#                 rigidbody_mass = org_rigidbody.param.mass

#             for rigidbody_name in sorted(created_rigidbodies.keys()):
#                 # ボーンを登録
#                 bone = created_bones[rigidbody_name]
#                 bone.index = len(model.bones)
#                 model.bones[bone.name] = bone

#                 # 剛体を登録
#                 rigidbody = created_rigidbodies[rigidbody_name]
#                 rigidbody.bone_index = bone.index
#                 rigidbody.index = len(model.rigidbodies)

#                 if rigidbody.name in model.rigidbodies:
#                     logger.warning("同じ剛体名が既に登録されているため、末尾に乱数を追加します。 既存剛体名: %s", rigidbody.name)
#                     rigidbody.name += randomname(3)

#                 # 登録したボーン名と剛体の対比表を保持
#                 registed_rigidbodies[rigidbody_name] = rigidbody.name
                
#                 model.rigidbodies[rigidbody.name] = rigidbody

#         logger.info("-- 剛体: %s個目:終了", len(created_rigidbodies))

#         return root_rigidbody, registed_rigidbodies

#     def create_weight(self, model: PmxModel, param_option: dict, vertex_map: np.ndarray, vertex_connected: dict, duplicate_vertices: dict, \
#                       registed_bone_indexs: dict, bone_horizonal_distances: dict, bone_vertical_distances: dict, vertex_remaining_set: set, target_vertices: list):
#         # ウェイト分布
#         prev_weight_cnt = 0
#         weight_cnt = 0

#         # 略称
#         abb_name = param_option['abb_name']

#         v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
#         for above_v_yidx, below_v_yidx in zip(v_yidxs[1:], v_yidxs[:-1]):
#             above_v_xidxs = list(registed_bone_indexs[above_v_yidx].keys())
#             below_v_xidxs = list(registed_bone_indexs[below_v_yidx].keys())
#             # 繋がってる場合、最後に最初のボーンを追加する
#             if above_v_yidx < len(vertex_connected) and vertex_connected[above_v_yidx]:
#                 above_v_xidxs += [list(registed_bone_indexs[above_v_yidx].keys())[0]]
#             if below_v_yidx < len(vertex_connected) and vertex_connected[below_v_yidx]:
#                 below_v_xidxs += [list(registed_bone_indexs[below_v_yidx].keys())[0]]

#             for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
#                 prev_below_v_xno = registed_bone_indexs[below_v_yidx][prev_below_v_xidx] + 1
#                 next_below_v_xno = registed_bone_indexs[below_v_yidx][next_below_v_xidx] + 1
#                 below_v_yno = below_v_yidx + 1

#                 prev_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)]
#                 next_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)]
#                 prev_above_bone = model.bones[model.bone_indexes[prev_below_bone.parent_index]]
#                 next_above_bone = model.bones[model.bone_indexes[next_below_bone.parent_index]]

#                 _, prev_above_v_xidx = self.disassemble_bone_name(prev_above_bone.name, registed_bone_indexs[above_v_yidx])
#                 _, next_above_v_xidx = self.disassemble_bone_name(next_above_bone.name, registed_bone_indexs[above_v_yidx])

#                 if xi > 0 and (next_below_v_xidx == registed_bone_indexs[below_v_yidx][list(registed_bone_indexs[below_v_yidx].keys())[0]] \
#                                or next_above_v_xidx == registed_bone_indexs[above_v_yidx][list(registed_bone_indexs[above_v_yidx].keys())[0]]):
#                     # nextが最初のボーンである場合、最後まで
#                     v_map = vertex_map[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
#                     b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
#                     b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
#                 else:
#                     v_map = vertex_map[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
#                     b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
#                     b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                
#                 for vi, v_vertices in enumerate(v_map):
#                     for vhi, vertex_idx in enumerate(v_vertices):
#                         if vertex_idx < 0 or vertex_idx not in target_vertices:
#                             continue

#                         horizonal_distance = np.sum(b_h_distances[vi, :])
#                         v_horizonal_distance = np.sum(b_h_distances[vi, :(vhi + 1)]) - b_h_distances[vi, 0]
#                         vertical_distance = np.sum(b_v_distances[:, vhi])
#                         v_vertical_distance = np.sum(b_v_distances[:(vi + 1), vhi]) - b_v_distances[0, vhi]

#                         prev_above_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
#                         prev_below_weight = max(0, ((horizonal_distance - v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))
#                         next_above_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((vertical_distance - v_vertical_distance) / vertical_distance))
#                         next_below_weight = max(0, ((v_horizonal_distance) / horizonal_distance) * ((v_vertical_distance) / vertical_distance))

#                         if below_v_yidx == v_yidxs[0]:
#                             # 最下段は末端ボーンにウェイトを振らない
#                             # 処理対象全ボーン名
#                             weight_bones = [prev_above_bone, next_above_bone]
#                             # ウェイト
#                             total_weights = [prev_above_weight + prev_below_weight, next_above_weight + next_below_weight]
#                         else:
#                             # 全処理対象ボーン名
#                             weight_bones = [prev_above_bone, next_above_bone, prev_below_bone, next_below_bone]
#                             # ウェイト
#                             total_weights = [prev_above_weight, next_above_weight, prev_below_weight, next_below_weight]

#                         bone_weights = {}
#                         for b, w in zip(weight_bones, total_weights):
#                             if b and b.getVisibleFlag():
#                                 if b not in bone_weights:
#                                     bone_weights[b.name] = 0
#                                 bone_weights[b.name] += w
                        
#                         if len(bone_weights) > 2:
#                             for _ in range(len(bone_weights), 5):
#                                 bone_weights[param_option['parent_bone_name']] = 0

#                         # 対象となるウェイト値
#                         weight_names = list(bone_weights.keys())
#                         total_weights = np.array(list(bone_weights.values()))

#                         if len(np.nonzero(total_weights)[0]) > 0:
#                             weights = total_weights / total_weights.sum(axis=0, keepdims=1)
#                             weight_idxs = np.argsort(weights)
#                             v = model.vertex_dict[vertex_idx]
#                             vertex_remaining_set -= set(duplicate_vertices[v.position.to_log()])

#                             for vvidx in duplicate_vertices[v.position.to_log()]:
#                                 vv = model.vertex_dict[vvidx]

#                                 logger.debug(f'vertex_idx: {vvidx}, weight_names: [{weight_names}], total_weights: [{total_weights}]')

#                                 if vv.deform.index0 == model.bones[param_option['parent_bone_name']].index:
#                                     # 重複頂点にも同じウェイトを割り当てる
#                                     if np.count_nonzero(weights) == 1:
#                                         vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
#                                     elif np.count_nonzero(weights) == 2:
#                                         vv.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
#                                     else:
#                                         vv.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
#                                                           model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
#                                                           weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])

#                                 weight_cnt += 1
#                                 if weight_cnt > 0 and weight_cnt // 1000 > prev_weight_cnt:
#                                     logger.info("-- 頂点ウェイト: %s個目:終了", weight_cnt)
#                                     prev_weight_cnt = weight_cnt // 1000

#         logger.info("-- 頂点ウェイト: %s個目:終了", weight_cnt)
        
#         return vertex_remaining_set

#     def create_remaining_weight(self, model: PmxModel, param_option: dict, vertex_maps: dict, \
#                                 vertex_remaining_set: set, boned_base_map_idxs: list, target_vertices: list):
#         # ウェイト分布
#         prev_weight_cnt = 0
#         weight_cnt = 0

#         vertex_distances = {}
#         for boned_map_idx in boned_base_map_idxs:
#             # 登録済み頂点との距離を測る（一番近いのと似たウェイト構成になるはず）
#             boned_vertex_map = vertex_maps[boned_map_idx]
#             for yi in range(boned_vertex_map.shape[0] - 1):
#                 for xi in range(boned_vertex_map.shape[1] - 1):
#                     if boned_vertex_map[yi, xi] >= 0:
#                         vi = boned_vertex_map[yi, xi]
#                         vertex_distances[vi] = model.vertex_dict[vi].position.data()

#         # 基準頂点マップ以外の頂点が残っていたら、それも割り当てる
#         for vertex_idx in list(vertex_remaining_set):
#             v = model.vertex_dict[vertex_idx]
#             if vertex_idx < 0 or vertex_idx not in target_vertices:
#                 continue
            
#             # 各頂点の位置との差分から距離を測る
#             rv_distances = np.linalg.norm((np.array(list(vertex_distances.values())) - v.position.data()), ord=2, axis=1)

#             # 近い頂点のうち、親ボーンにウェイトが乗ってないのを選択
#             for nearest_vi in np.argsort(rv_distances):
#                 nearest_vidx = list(vertex_distances.keys())[nearest_vi]
#                 nearest_v = model.vertex_dict[nearest_vidx]
#                 nearest_deform = nearest_v.deform
#                 if type(nearest_deform) is Bdef1 and nearest_deform.index0 == model.bones[param_option['parent_bone_name']].index:
#                     # 直近が親ボーンの場合、一旦スルー
#                     continue
#                 else:
#                     break

#             if type(nearest_deform) is Bdef1:
#                 logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{model.bone_indexes[nearest_deform.index0]}], total_weights: [1]')

#                 v.deform = Bdef1(nearest_deform.index0)
#             elif type(nearest_deform) is Bdef2:
#                 weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
#                 weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]

#                 bone1_distance = v.position.distanceToPoint(weight_bone1.position)
#                 bone2_distance = v.position.distanceToPoint(weight_bone2.position) if nearest_deform.weight0 < 1 else 0

#                 weight_names = np.array([weight_bone1.name, weight_bone2.name])
#                 if bone1_distance + bone2_distance != 0:
#                     total_weights = np.array([bone1_distance / (bone1_distance + bone2_distance), bone2_distance / (bone1_distance + bone2_distance)])
#                 else:
#                     total_weights = np.array([1, 0])
#                     logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", v.index)
#                 weights = total_weights / total_weights.sum(axis=0, keepdims=1)
#                 weight_idxs = np.argsort(weights)

#                 logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{weight_names}], total_weights: [{total_weights}]')
                
#                 if np.count_nonzero(weights) == 1:
#                     v.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
#                 elif np.count_nonzero(weights) == 2:
#                     v.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
#                 else:
#                     v.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
#                                      model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
#                                      weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])

#             elif type(nearest_deform) is Bdef4:
#                 weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
#                 weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]
#                 weight_bone3 = model.bones[model.bone_indexes[nearest_deform.index2]]
#                 weight_bone4 = model.bones[model.bone_indexes[nearest_deform.index3]]

#                 bone1_distance = v.position.distanceToPoint(weight_bone1.position) if nearest_deform.weight0 > 0 else 0
#                 bone2_distance = v.position.distanceToPoint(weight_bone2.position) if nearest_deform.weight1 > 0 else 0
#                 bone3_distance = v.position.distanceToPoint(weight_bone3.position) if nearest_deform.weight2 > 0 else 0
#                 bone4_distance = v.position.distanceToPoint(weight_bone4.position) if nearest_deform.weight3 > 0 else 0
#                 all_distance = bone1_distance + bone2_distance + bone3_distance + bone4_distance

#                 weight_names = np.array([weight_bone1.name, weight_bone2.name, weight_bone3.name, weight_bone4.name])
#                 if all_distance != 0:
#                     total_weights = np.array([bone1_distance / all_distance, bone2_distance / all_distance, bone3_distance / all_distance, bone4_distance / all_distance])
#                 else:
#                     total_weights = np.array([1, bone2_distance, bone3_distance, bone4_distance])
#                     logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", v.index)
#                 weights = total_weights / total_weights.sum(axis=0, keepdims=1)
#                 weight_idxs = np.argsort(weights)

#                 logger.debug(f'remaining vertex_idx: {v.index}, weight_names: [{weight_names}], total_weights: [{total_weights}]')

#                 if np.count_nonzero(weights) == 1:
#                     v.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
#                 elif np.count_nonzero(weights) == 2:
#                     v.deform = Bdef2(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, weights[weight_idxs[-1]])
#                 else:
#                     v.deform = Bdef4(model.bones[weight_names[weight_idxs[-1]]].index, model.bones[weight_names[weight_idxs[-2]]].index, \
#                                      model.bones[weight_names[weight_idxs[-3]]].index, model.bones[weight_names[weight_idxs[-4]]].index, \
#                                      weights[weight_idxs[-1]], weights[weight_idxs[-2]], weights[weight_idxs[-3]], weights[weight_idxs[-4]])
            
#             weight_cnt += 1
#             if weight_cnt > 0 and weight_cnt // 100 > prev_weight_cnt:
#                 logger.info("-- 残頂点ウェイト: %s個目:終了", weight_cnt)
#                 prev_weight_cnt = weight_cnt // 100

#         logger.info("-- 残頂点ウェイト: %s個目:終了", weight_cnt)

#     def create_back_weight(self, model: PmxModel, param_option: dict):
#         # ウェイト分布
#         prev_weight_cnt = 0
#         weight_cnt = 0

#         front_vertex_keys = []
#         front_vertex_positions = []
#         for front_vertex_idx in list(model.material_vertices[param_option['material_name']]):
#             front_vertex_keys.append(front_vertex_idx)
#             front_vertex_positions.append(model.vertex_dict[front_vertex_idx].position.data())

#         for vertex_idx in list(model.material_vertices[param_option['back_material_name']]):
#             bv = model.vertex_dict[vertex_idx]

#             # 各頂点の位置との差分から距離を測る
#             bv_distances = np.linalg.norm((np.array(front_vertex_positions) - bv.position.data()), ord=2, axis=1)

#             # 直近頂点INDEXのウェイトを転写
#             copy_front_vertex_idx = front_vertex_keys[np.argmin(bv_distances)]
#             bv.deform = copy.deepcopy(model.vertex_dict[copy_front_vertex_idx].deform)

#             weight_cnt += 1
#             if weight_cnt > 0 and weight_cnt // 200 > prev_weight_cnt:
#                 logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)
#                 prev_weight_cnt = weight_cnt // 200

#         logger.info("-- 裏頂点ウェイト: %s個目:終了", weight_cnt)

#     def create_root_bone(self, model: PmxModel, param_option: dict):
#         # 略称
#         abb_name = param_option['abb_name']

#         root_bone = Bone(f'{abb_name}中心', f'{abb_name}中心', model.bones[param_option['parent_bone_name']].position, \
#                          model.bones[param_option['parent_bone_name']].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
#         root_bone.index = len(list(model.bones.keys()))

#         # ボーン
#         model.bones[root_bone.name] = root_bone
#         model.bone_indexes[root_bone.index] = root_bone.name

#         return root_bone

#     def create_bone(self, model: PmxModel, param_option: dict, vertex_map_orders: list, vertex_maps: dict, vertex_connecteds: dict):
#         # 中心ボーン生成

#         # 略称
#         abb_name = param_option['abb_name']
#         # 材質名
#         material_name = param_option['material_name']
#         # 表示枠名
#         display_name = f"{abb_name}:{material_name}"

#         # 表示枠定義
#         model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)

#         root_bone = self.create_root_bone(model, param_option)
        
#         # 表示枠
#         model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

#         tmp_all_bones = {}
#         all_yidxs = {}
#         all_bone_indexes = {}
#         all_registed_bone_indexs = {}

#         all_bone_horizonal_distances = {}
#         all_bone_vertical_distances = {}

#         for base_map_idx, vertex_map in enumerate(vertex_maps):
#             bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1] + 1))
#             bone_vertical_distances = np.zeros(vertex_map.shape)

#             # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
#             for v_yidx in range(vertex_map.shape[0]):
#                 for v_xidx in range(vertex_map.shape[1]):
#                     if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx, v_xidx - 1] >= 0:
#                         now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
#                         prev_v_vec = now_v_vec if v_xidx == 0 else model.vertex_dict[vertex_map[v_yidx, v_xidx - 1]].position
#                         bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)
#                     if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx - 1, v_xidx] >= 0:
#                         now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
#                         prev_v_vec = now_v_vec if v_yidx == 0 else model.vertex_dict[vertex_map[v_yidx - 1, v_xidx]].position
#                         bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(prev_v_vec)
#                 if vertex_map[v_yidx, v_xidx] >= 0 and vertex_map[v_yidx, 0] >= 0:
#                     # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
#                     now_v_vec = model.vertex_dict[vertex_map[v_yidx, v_xidx]].position
#                     prev_v_vec = model.vertex_dict[vertex_map[v_yidx, 0]].position
#                     bone_horizonal_distances[v_yidx, v_xidx + 1] = now_v_vec.distanceToPoint(prev_v_vec)

#             all_bone_horizonal_distances[base_map_idx] = bone_horizonal_distances
#             all_bone_vertical_distances[base_map_idx] = bone_vertical_distances

#         for base_map_idx in vertex_map_orders:
#             vertex_map = vertex_maps[base_map_idx]
#             vertex_connected = vertex_connecteds[base_map_idx]

#             bone_vertical_distances = all_bone_vertical_distances[base_map_idx]
#             full_xs = np.arange(0, vertex_map.shape[1])[np.count_nonzero(bone_vertical_distances, axis=0) == max(np.count_nonzero(bone_vertical_distances, axis=0))]
#             median_x = int(np.median(full_xs))
#             median_y_distance = np.mean(bone_vertical_distances[:, median_x][np.nonzero(bone_vertical_distances[:, median_x])])

#             prev_yi = 0
#             v_yidxs = []
#             for yi, bh in enumerate(bone_vertical_distances[1:, median_x]):
#                 if yi == 0 or np.sum(bone_vertical_distances[prev_yi:(yi + 1), median_x]) >= median_y_distance * param_option["vertical_bone_density"] * 0.8:
#                     v_yidxs.append(yi)
#                     prev_yi = yi + 1
#             if v_yidxs[-1] < vertex_map.shape[0] - 1:
#                 # 最下段は必ず登録
#                 v_yidxs = v_yidxs + [vertex_map.shape[0] - 1]
#             all_yidxs[base_map_idx] = v_yidxs

#             # 中央あたりの横幅中央値ベースで横の割りを決める
#             bone_horizonal_distances = all_bone_horizonal_distances[base_map_idx]
#             full_ys = [y for i, y in enumerate(v_yidxs) if np.count_nonzero(bone_horizonal_distances[i, :]) == max(np.count_nonzero(bone_horizonal_distances, axis=1))]
#             if not full_ys:
#                 full_ys = v_yidxs
#             median_y = int(np.median(full_ys))
#             median_x_distance = np.median(bone_horizonal_distances[median_y, :][np.nonzero(bone_horizonal_distances[median_y, :])])

#             prev_xi = 0
#             base_v_xidxs = []
#             if param_option["density_type"] == logger.transtext('距離'):
#                 # 距離ベースの場合、中間距離で割りを決める
#                 for xi, bh in enumerate(bone_horizonal_distances[median_y, 1:]):
#                     if xi == 0 or np.sum(bone_horizonal_distances[median_y, prev_xi:(xi + 1)]) >= median_x_distance * param_option["horizonal_bone_density"] * 0.8:
#                         base_v_xidxs.append(xi)
#                         prev_xi = xi + 1
#             else:
#                 base_v_xidxs = list(range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]))

#             if base_v_xidxs[-1] < vertex_map.shape[1] - param_option["horizonal_bone_density"]:
#                 # 右端は必ず登録
#                 base_v_xidxs = base_v_xidxs + [vertex_map.shape[1] - param_option["horizonal_bone_density"]]

#             all_bone_indexes[base_map_idx] = {}
#             for yi in range(vertex_map.shape[0]):
#                 all_bone_indexes[base_map_idx][yi] = {}
#                 v_xidxs = copy.deepcopy(base_v_xidxs)
#                 if not vertex_connected[yi] and v_xidxs[-1] < vertex_map.shape[1] - 1:
#                     # 繋がってなくて、かつ端っこが登録されていない場合、登録
#                     v_xidxs = v_xidxs + [vertex_map.shape[1] - 1]
#                 max_xi = 0
#                 for midx, myidxs in all_bone_indexes.items():
#                     if midx != base_map_idx and yi in all_bone_indexes[midx]:
#                         max_xi = max(list(all_bone_indexes[midx][yi].keys())) + 1
#                 for xi in v_xidxs:
#                     all_bone_indexes[base_map_idx][yi][xi] = xi + max_xi
            
#         for base_map_idx in vertex_map_orders:
#             v_yidxs = all_yidxs[base_map_idx]
#             vertex_map = vertex_maps[base_map_idx]
#             vertex_connected = vertex_connecteds[base_map_idx]
#             registed_bone_indexs = {}

#             for yi, v_yidx in enumerate(v_yidxs):
#                 for v_xidx, total_v_xidx in all_bone_indexes[base_map_idx][yi].items():
#                     if v_yidx >= vertex_map.shape[0] or v_xidx >= vertex_map.shape[1] or vertex_map[v_yidx, v_xidx] < 0:
#                         # 存在しない頂点はスルー
#                         continue
                    
#                     v = model.vertex_dict[vertex_map[v_yidx, v_xidx]]
#                     v_xno = total_v_xidx + 1
#                     v_yno = v_yidx + 1

#                     # ボーン仮登録
#                     bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
#                     bone = Bone(bone_name, bone_name, v.position, root_bone.index, 0, 0x0000 | 0x0002)
#                     bone.local_z_vector = v.normal.copy()
#                     tmp_all_bones[bone.name] = {"bone": bone, "parent": root_bone.name, "regist": False}
#                     logger.debug(f"tmp_all_bones: {bone.name}, pos: {bone.position.to_log()}")

#             # 最下段の横幅最小値(段数単位)
#             edge_size = np.min(all_bone_horizonal_distances[base_map_idx][-1, 1:]) * param_option["horizonal_bone_density"]

#             for yi, v_yidx in enumerate(v_yidxs):
#                 prev_xidx = 0
#                 if v_yidx not in registed_bone_indexs:
#                     registed_bone_indexs[v_yidx] = {}

#                 for v_xidx, total_v_xidx in all_bone_indexes[base_map_idx][yi].items():
#                     if v_xidx == 0 or (not vertex_connected[yi] and v_xidx == list(all_bone_indexes[base_map_idx][yi].keys())[-1]) or \
#                         not param_option['bone_thinning_out'] or (param_option['bone_thinning_out'] and \
#                         np.sum(all_bone_horizonal_distances[base_map_idx][v_yidx, (prev_xidx + 1):(v_xidx + 1)]) >= edge_size * 0.9):  # noqa
#                         # 前ボーンとの間隔が最下段の横幅平均値より開いている場合、登録対象
#                         v_xno = total_v_xidx + 1
#                         v_yno = v_yidx + 1

#                         # ボーン名
#                         bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
#                         if bone_name not in tmp_all_bones:
#                             continue

#                         # ボーン本登録
#                         bone = tmp_all_bones[bone_name]["bone"]
#                         bone.index = len(list(model.bones.keys()))

#                         if yi > 0:
#                             parent_v_xidx_diff = np.abs(np.array(list(registed_bone_indexs[v_yidxs[yi - 1]].values())) - total_v_xidx)

#                             # 2段目以降は最も近い親段でv_xidxを探す
#                             parent_v_xidx = list(registed_bone_indexs[v_yidxs[yi - 1]].values())[(0 if vertex_connected[yi] and (v_xidxs[-1] + 1) - v_xidx < np.min(parent_v_xidx_diff) else np.argmin(parent_v_xidx_diff))]   # noqa

#                             parent_bone = model.bones[self.get_bone_name(abb_name, v_yidxs[yi - 1] + 1, parent_v_xidx + 1)]
#                             bone.parent_index = parent_bone.index
#                             bone.local_x_vector = (bone.position - parent_bone.position).normalized()
#                             bone.local_z_vector *= MVector3D(-1, 1, -1)
#                             bone.flag |= 0x0800

#                             tmp_all_bones[bone.name]["parent"] = parent_bone.name

#                             # 親ボーンの表示先も同時設定
#                             parent_bone.tail_index = bone.index
#                             parent_bone.local_x_vector = (bone.position - parent_bone.position).normalized()
#                             parent_bone.flag |= 0x0001

#                             # 表示枠
#                             parent_bone.flag |= 0x0008 | 0x0010
#                             model.display_slots[display_name].references.append((0, parent_bone.index))

#                         model.bones[bone.name] = bone
#                         model.bone_indexes[bone.index] = bone.name
#                         tmp_all_bones[bone.name]["regist"] = True

#                         registed_bone_indexs[v_yidx][v_xidx] = total_v_xidx

#                         # 前ボーンとして設定
#                         prev_xidx = v_xidx
            
#             logger.debug(f"registed_bone_indexs: {registed_bone_indexs}")

#             all_registed_bone_indexs[base_map_idx] = registed_bone_indexs

#         return root_bone, tmp_all_bones, all_registed_bone_indexs, all_bone_horizonal_distances, all_bone_vertical_distances

#     def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
#         return f'{abb_name}-{(v_yno):03d}-{(v_xno):03d}'

#     def disassemble_bone_name(self, bone_name: str, v_xidxs=None):
#         total_v_xidx = int(bone_name[-3:]) - 1
#         now_vidxs = [k for k, v in v_xidxs.items() if v == total_v_xidx] if v_xidxs else [total_v_xidx]
#         v_xidx = now_vidxs[0] if now_vidxs else total_v_xidx
#         v_yidx = int(bone_name[-7:-4]) - 1

#         return v_yidx, v_xidx
    
#     def get_saved_bone_names(self, model: PmxModel):
#         saved_bone_names = []
#         # 準標準ボーンまでは削除対象外
#         saved_bone_names.extend(SEMI_STANDARD_BONE_NAMES)

#         for pidx, param_option in enumerate(self.options.param_options):
#             if param_option['exist_physics_clear'] == logger.transtext('そのまま'):
#                 continue

#             edge_material_name = param_option['edge_material_name']
#             back_material_name = param_option['back_material_name']
#             weighted_bone_indexes = {}

#             # 頂点CSVが指定されている場合、対象頂点リスト生成
#             if param_option['vertices_csv']:
#                 target_vertices = []
#                 with open(param_option['vertices_csv'], encoding='cp932', mode='r') as f:
#                     reader = csv.reader(f)
#                     next(reader)            # ヘッダーを読み飛ばす
#                     for row in reader:
#                         if len(row) > 1 and int(row[1]) in model.material_vertices[param_option['material_name']]:
#                             target_vertices.append(int(row[1]))
#             else:
#                 target_vertices = list(model.material_vertices[param_option['material_name']])
            
#             if edge_material_name:
#                 target_vertices = list(set(target_vertices) | set(model.material_vertices[edge_material_name]))
            
#             if back_material_name:
#                 target_vertices = list(set(target_vertices) | set(model.material_vertices[back_material_name]))

#             if param_option['exist_physics_clear'] == logger.transtext('再利用'):
#                 # 再利用の場合、指定されている全ボーンを対象とする
#                 bone_grid = param_option["bone_grid"]
#                 bone_grid_cols = param_option["bone_grid_cols"]
#                 bone_grid_rows = param_option["bone_grid_rows"]

#                 for r in range(bone_grid_rows):
#                     for c in range(bone_grid_cols):
#                         if bone_grid[r][c]:
#                             weighted_bone_indexes[bone_grid[r][c]] = model.bones[bone_grid[r][c]].index
#             else:
#                 for vertex_idx in target_vertices:
#                     vertex = model.vertex_dict[vertex_idx]
#                     if type(vertex.deform) is Bdef1:
#                         if vertex.deform.index0 not in list(weighted_bone_indexes.values()):
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                     elif type(vertex.deform) is Bdef2:
#                         if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                         if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
#                     elif type(vertex.deform) is Bdef4:
#                         if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                         if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight1 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
#                         if vertex.deform.index2 not in list(weighted_bone_indexes.values()) and vertex.deform.weight2 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index2]] = vertex.deform.index2
#                         if vertex.deform.index3 not in list(weighted_bone_indexes.values()) and vertex.deform.weight3 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index3]] = vertex.deform.index3
#                     elif type(vertex.deform) is Sdef:
#                         if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                         if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
#                             weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
            
#             if param_option['exist_physics_clear'] == logger.transtext('再利用'):
#                 # 再利用する場合、ボーンは全部残す
#                 saved_bone_names.extend(list(model.bones.keys()))
#             else:
#                 # 他の材質で該当ボーンにウェイト割り当てられている場合、ボーンの削除だけは避ける
#                 for bone_idx, vertices in model.vertices.items():
#                     is_not_delete = False
#                     if bone_idx in list(weighted_bone_indexes.values()) and len(vertices) > 0:
#                         is_not_delete = False
#                         for vertex in vertices:
#                             if vertex.index not in target_vertices:
#                                 is_not_delete = True
#                                 for material_name, material_vertices in model.material_vertices.items():
#                                     if vertex.index in material_vertices:
#                                         break
#                                 logger.info("削除対象外ボーン: %s(%s), 対象外頂点: %s, 所属材質: %s", \
#                                             model.bone_indexes[bone_idx], bone_idx, vertex.index, material_name)
#                                 break
#                     if is_not_delete:
#                         saved_bone_names.append(model.bone_indexes[bone_idx])

#             # 非表示子ボーンも削除する
#             for bone in model.bones.values():
#                 if not bone.getVisibleFlag() and bone.parent_index in model.bone_indexes and model.bone_indexes[bone.parent_index] in weighted_bone_indexes \
#                         and model.bone_indexes[bone.parent_index] not in saved_bone_names:
#                     weighted_bone_indexes[bone.name] = bone.index
            
#             logger.debug('weighted_bone_indexes: %s', ", ".join(list(weighted_bone_indexes.keys())))
#             logger.debug('saved_bone_names: %s', ", ".join(saved_bone_names))

#         return saved_bone_names

#     def clear_exist_physics(self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list, saved_bone_names: list):
#         logger.info("%s: 削除対象抽出", material_name)

#         weighted_bone_indexes = {}
#         if param_option['exist_physics_clear'] == logger.transtext('再利用'):
#             # 再利用の場合、指定されている全ボーンを対象とする
#             bone_grid = param_option["bone_grid"]
#             bone_grid_cols = param_option["bone_grid_cols"]
#             bone_grid_rows = param_option["bone_grid_rows"]

#             for r in range(bone_grid_rows):
#                 for c in range(bone_grid_cols):
#                     if bone_grid[r][c]:
#                         weighted_bone_indexes[bone_grid[r][c]] = model.bones[bone_grid[r][c]].index
#         else:
#             for vertex_idx in target_vertices:
#                 vertex = model.vertex_dict[vertex_idx]
#                 if type(vertex.deform) is Bdef1:
#                     if vertex.deform.index0 not in list(weighted_bone_indexes.values()):
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                 elif type(vertex.deform) is Bdef2:
#                     if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                     if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
#                 elif type(vertex.deform) is Bdef4:
#                     if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                     if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight1 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
#                     if vertex.deform.index2 not in list(weighted_bone_indexes.values()) and vertex.deform.weight2 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index2]] = vertex.deform.index2
#                     if vertex.deform.index3 not in list(weighted_bone_indexes.values()) and vertex.deform.weight3 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index3]] = vertex.deform.index3
#                 elif type(vertex.deform) is Sdef:
#                     if vertex.deform.index0 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 > 0:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
#                     if vertex.deform.index1 not in list(weighted_bone_indexes.values()) and vertex.deform.weight0 < 1:
#                         weighted_bone_indexes[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
        
#         # 非表示子ボーンも削除する
#         for bone in model.bones.values():
#             if not bone.getVisibleFlag() and bone.parent_index in model.bone_indexes and model.bone_indexes[bone.parent_index] in weighted_bone_indexes \
#                     and model.bone_indexes[bone.parent_index] not in SEMI_STANDARD_BONE_NAMES:
#                 weighted_bone_indexes[bone.name] = bone.index
        
#         for bone in model.bones.values():
#             is_target = True
#             if bone.name in saved_bone_names and bone.name in weighted_bone_indexes:
#                 # 保存済みボーン名に入ってても対象外
#                 logger.warning("他の材質のウェイトボーンとして設定されているため、ボーン「%s」を削除対象外とします。", bone.name)
#                 is_target = False

#             if is_target:
#                 for vertex in model.vertices.get(bone.name, []):
#                     for vertex_weight_bone_index in vertex.get_idx_list():
#                         if vertex_weight_bone_index not in weighted_bone_indexes.values():
#                             # 他のボーンのウェイトが乗ってたら対象外
#                             logger.warning("削除対象外ボーンにウェイトが乗っているため、ボーン「%s」を削除対象外とします。\n調査対象インデックス：%s", bone.name, vertex.index)
#                             is_target = False
#                             break

#             if not is_target and bone.name in weighted_bone_indexes:
#                 logger.debug("他ウェイト対象外: %s", bone.name)
#                 del weighted_bone_indexes[bone.name]

#         for bone_name, bone_index in weighted_bone_indexes.items():
#             bone = model.bones[bone_name]
#             for morph in model.org_morphs.values():
#                 if morph.morph_type == 2:
#                     for offset in morph.offsets:
#                         if type(offset) is BoneMorphData:
#                             if offset.bone_index == bone_index:
#                                 logger.error("削除対象ボーンがボーンモーフとして登録されているため、削除出来ません。\n" \
#                                              + "事前にボーンモーフから外すか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s), モーフ名: %s", \
#                                              bone_name, morph.name, decoration=MLogger.DECORATION_BOX)
#                                 return None
#             for bidx, bone in enumerate(model.bones.values()):
#                 if bone.parent_index == bone_index and bone.index not in weighted_bone_indexes.values():
#                     logger.error("削除対象ボーンが削除対象外ボーンの親ボーンとして登録されているため、削除出来ません。\n" \
#                                  + "事前に親子関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外子ボーン: %s(%s)", \
#                                  bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
#                     return None

#                 if (bone.getExternalRotationFlag() or bone.getExternalTranslationFlag()) \
#                    and bone.effect_index == bone_index and bone.index not in weighted_bone_indexes.values():
#                     logger.error("削除対象ボーンが削除対象外ボーンの付与親ボーンとして登録されているため、削除出来ません。\n" \
#                                  + "事前に付与関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外付与子ボーン: %s(%s)", \
#                                  bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
#                     return None
                    
#                 if bone.getIkFlag():
#                     if bone.ik.target_index == bone_index and bone.index not in weighted_bone_indexes.values():
#                         logger.error("削除対象ボーンが削除対象外ボーンのリンクターゲットボーンとして登録されているため、削除出来ません。\n" \
#                                      + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)", \
#                                      bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
#                         return None

#                     for link in bone.ik.link:
#                         if link.bone_index == bone_index and bone.index not in weighted_bone_indexes.values():
#                             logger.error("削除対象ボーンが削除対象外ボーンのリンクボーンとして登録されているため、削除出来ません。\n" \
#                                          + "事前にIK関係を解除するか、再利用で物理を生成してください。\n削除対象ボーン：%s(%s)\n削除対象外IKボーン: %s(%s)", \
#                                          bone_name, bone_index, bone.name, bone.index, decoration=MLogger.DECORATION_BOX)
#                             return None
                           
#         weighted_rigidbody_indexes = {}
#         for rigidbody in model.rigidbodies.values():
#             if rigidbody.index not in list(weighted_rigidbody_indexes.values()) and rigidbody.bone_index in list(weighted_bone_indexes.values()) \
#                and model.bone_indexes[rigidbody.bone_index] not in SEMI_STANDARD_BONE_NAMES:
#                 weighted_rigidbody_indexes[rigidbody.name] = rigidbody.index

#         weighted_joint_indexes = {}
#         for joint in model.joints.values():
#             if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_a in list(weighted_rigidbody_indexes.values()):
#                 weighted_joint_indexes[joint.name] = joint.name
#             if joint.name not in list(weighted_joint_indexes.values()) and joint.rigidbody_index_b in list(weighted_rigidbody_indexes.values()):
#                 weighted_joint_indexes[joint.name] = joint.name

#         logger.info("%s: 削除実行", material_name)

#         logger.info('削除対象ボーンリスト: %s', ", ".join(list(weighted_bone_indexes.keys())))
#         logger.info('削除対象剛体リスト: %s', ", ".join(list(weighted_rigidbody_indexes.keys())))
#         logger.info('削除対象ジョイントリスト: %s', ", ".join((weighted_joint_indexes.keys())))

#         # 削除
#         for joint_name in weighted_joint_indexes.keys():
#             del model.joints[joint_name]

#         for rigidbody_name in weighted_rigidbody_indexes.keys():
#             del model.rigidbodies[rigidbody_name]

#         for bone_name in weighted_bone_indexes.keys():
#             if bone_name not in saved_bone_names:
#                 del model.bones[bone_name]

#         logger.info("%s: INDEX振り直し", material_name)

#         reset_rigidbodies = {}
#         for ridx, (rigidbody_name, rigidbody) in enumerate(model.rigidbodies.items()):
#             reset_rigidbodies[rigidbody.index] = {'name': rigidbody_name, 'index': ridx}
#             model.rigidbodies[rigidbody_name].index = ridx

#         reset_bones = {}
#         for bidx, (bone_name, bone) in enumerate(model.bones.items()):
#             reset_bones[bone.index] = {'name': bone_name, 'index': bidx}
#             model.bones[bone_name].index = bidx
#             model.bone_indexes[bidx] = bone_name

#         logger.info("%s: INDEX再割り当て", material_name)

#         for jidx, (joint_name, joint) in enumerate(model.joints.items()):
#             if joint.rigidbody_index_a in reset_rigidbodies:
#                 joint.rigidbody_index_a = reset_rigidbodies[joint.rigidbody_index_a]['index']
#             if joint.rigidbody_index_b in reset_rigidbodies:
#                 joint.rigidbody_index_b = reset_rigidbodies[joint.rigidbody_index_b]['index']
#         for rigidbody in model.rigidbodies.values():
#             if rigidbody.bone_index in reset_bones:
#                 rigidbody.bone_index = reset_bones[rigidbody.bone_index]['index']
#             else:
#                 rigidbody.bone_index = -1

#         for display_slot in model.display_slots.values():
#             new_references = []
#             for display_type, bone_idx in display_slot.references:
#                 if display_type == 0:
#                     if bone_idx in reset_bones:
#                         new_references.append((display_type, reset_bones[bone_idx]['index']))
#                 else:
#                     new_references.append((display_type, bone_idx))
#             display_slot.references = new_references

#         for morph in model.org_morphs.values():
#             if morph.morph_type == 2:
#                 new_offsets = []
#                 for offset in morph.offsets:
#                     if type(offset) is BoneMorphData:
#                         if offset.bone_index in reset_bones:
#                             offset.bone_index = reset_bones[offset.bone_index]['index']
#                             new_offsets.append(offset)
#                         else:
#                             offset.bone_index = -1
#                             new_offsets.append(offset)
#                     else:
#                         new_offsets.append(offset)
#                 morph.offsets = new_offsets

#         for bidx, bone in enumerate(model.bones.values()):
#             if bone.parent_index in reset_bones:
#                 bone.parent_index = reset_bones[bone.parent_index]['index']
#             else:
#                 bone.parent_index = -1

#             if bone.getConnectionFlag():
#                 if bone.tail_index in reset_bones:
#                     bone.tail_index = reset_bones[bone.tail_index]['index']
#                 else:
#                     bone.tail_index = -1

#             if bone.getExternalRotationFlag() or bone.getExternalTranslationFlag():
#                 if bone.effect_index in reset_bones:
#                     bone.effect_index = reset_bones[bone.effect_index]['index']
#                 else:
#                     bone.effect_index = -1

#             if bone.getIkFlag():
#                 if bone.ik.target_index in reset_bones:
#                     bone.ik.target_index = reset_bones[bone.ik.target_index]['index']
#                     for link in bone.ik.link:
#                         link.bone_index = reset_bones[link.bone_index]['index']
#                 else:
#                     bone.ik.target_index = -1
#                     for link in bone.ik.link:
#                         link.bone_index = -1

#         for vidx, vertex in enumerate(model.vertex_dict.values()):
#             if type(vertex.deform) is Bdef1:
#                 vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
#             elif type(vertex.deform) is Bdef2:
#                 vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
#                 vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1
#             elif type(vertex.deform) is Bdef4:
#                 vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
#                 vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1
#                 vertex.deform.index2 = reset_bones[vertex.deform.index2]['index'] if vertex.deform.index2 in reset_bones else -1
#                 vertex.deform.index3 = reset_bones[vertex.deform.index3]['index'] if vertex.deform.index3 in reset_bones else -1
#             elif type(vertex.deform) is Sdef:
#                 vertex.deform.index0 = reset_bones[vertex.deform.index0]['index'] if vertex.deform.index0 in reset_bones else -1
#                 vertex.deform.index1 = reset_bones[vertex.deform.index1]['index'] if vertex.deform.index1 in reset_bones else -1

#         return model

#     # 頂点を展開した図を作成
#     def create_vertex_map(self, model: PmxModel, param_option: dict, material_name: str, target_vertices: list):
#         logger.info("%s: 面の抽出", material_name)

#         logger.info("%s: 面の抽出準備①", material_name)

#         # 位置ベースで重複頂点の抽出
#         duplicate_vertices = {}
#         ybase_vertices = {}
#         for vertex_idx in model.material_vertices[material_name]:
#             if vertex_idx not in target_vertices:
#                 continue
#             # 重複頂点の抽出
#             vertex = model.vertex_dict[vertex_idx]
#             key = vertex.position.to_log()
#             if key not in duplicate_vertices:
#                 duplicate_vertices[key] = []
#             if vertex.index not in duplicate_vertices[key]:
#                 duplicate_vertices[key].append(vertex.index)

#             key = round(vertex.position.y(), 3)
#             if key not in ybase_vertices:
#                 ybase_vertices[key] = []
#             if vertex.index not in ybase_vertices[key]:
#                 ybase_vertices[key].append(vertex.index)

#             # 一旦ルートボーンにウェイトを一括置換
#             vertex.deform = Bdef1(model.bones[param_option['parent_bone_name']].index)

#         if len(ybase_vertices.keys()) == 0:
#             logger.warning("対象範囲となる頂点が取得できなかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
#             return None, None, None, None, None, None
                
#         ymin = np.min(np.array(list(ybase_vertices.keys())))
#         ymax = np.max(np.array(list(ybase_vertices.keys())))
#         ymedian = np.median(np.array(list(ybase_vertices.keys())))

#         logger.info("%s: 面の抽出準備②", material_name)

#         non_target_iidxs = []
#         # 面組み合わせの生成
#         indices_by_vidx = {}
#         indices_by_vpos = {}
#         index_combs_by_vpos = {}
#         duplicate_indices = {}
#         # below_iidx = None
#         # max_below_x = -9999
#         # max_below_size = -9999
#         for index_idx in model.material_indices[material_name]:
#             # 頂点の組み合わせから面INDEXを引く
#             indices_by_vidx[tuple(sorted(model.indices[index_idx]))] = index_idx
#             v0 = model.vertex_dict[model.indices[index_idx][0]]
#             v1 = model.vertex_dict[model.indices[index_idx][1]]
#             v2 = model.vertex_dict[model.indices[index_idx][2]]
#             if v0.index not in target_vertices or v1.index not in target_vertices or v2.index not in target_vertices:
#                 # 3つ揃ってない場合、スルー
#                 non_target_iidxs.append(index_idx)
#                 continue
            
#             # 重複辺（2点）の組み合わせ
#             index_combs = list(itertools.combinations(model.indices[index_idx], 2))
#             for (iv1, iv2) in index_combs:
#                 for ivv1, ivv2 in list(itertools.product(duplicate_vertices[model.vertex_dict[iv1].position.to_log()], duplicate_vertices[model.vertex_dict[iv2].position.to_log()])):
#                     # 小さいINDEX・大きい頂点INDEXのセットでキー生成
#                     key = (min(ivv1, ivv2), max(ivv1, ivv2))
#                     if key not in duplicate_indices:
#                         duplicate_indices[key] = []
#                     if index_idx not in duplicate_indices[key]:
#                         duplicate_indices[key].append(index_idx)
#             # 頂点別に組み合わせも登録
#             for iv in model.indices[index_idx]:
#                 vpkey = model.vertex_dict[iv].position.to_log()
#                 if vpkey in duplicate_vertices and vpkey not in index_combs_by_vpos:
#                     index_combs_by_vpos[vpkey] = []
#                 # 同一頂点位置を持つ面のリスト
#                 if vpkey in duplicate_vertices and vpkey not in indices_by_vpos:
#                     indices_by_vpos[vpkey] = []
#                 if index_idx not in indices_by_vpos[vpkey]:
#                     indices_by_vpos[vpkey].append(index_idx)
#             for (iv1, iv2) in index_combs:
#                 # 小さいINDEX・大きい頂点INDEXのセットでキー生成
#                 key = (min(iv1, iv2), max(iv1, iv2))
#                 if key not in index_combs_by_vpos[vpkey]:
#                     index_combs_by_vpos[vpkey].append(key)

#         logger.info("%s: 相対頂点マップの生成", material_name)

#         # 頂点マップ生成(最初の頂点が(0, 0))
#         vertex_axis_maps = []
#         vertex_coordinate_maps = []
#         registed_iidxs = copy.deepcopy(non_target_iidxs)
#         vertical_iidxs = []
#         prev_index_cnt = 0

#         while len(registed_iidxs) < len(model.material_indices[material_name]):
#             if not vertical_iidxs:
#                 # 切替時はとりあえず一面取り出して判定(二次元配列になる)
#                 # 出来るだけ真っ直ぐの辺がある面とする
#                 max_below_x = 0
#                 max_below_x_size = 0
#                 max_below_y = 0
#                 max_below_y_size = 0
#                 remaining_iidxs = list(set(model.material_indices[material_name]) - set(registed_iidxs))
#                 below_x_iidx = None
#                 below_y_iidx = None
#                 for index_idx in remaining_iidxs:
#                     v0 = model.vertex_dict[model.indices[index_idx][0]]
#                     v1 = model.vertex_dict[model.indices[index_idx][1]]
#                     v2 = model.vertex_dict[model.indices[index_idx][2]]
#                     if v0.index not in target_vertices or v1.index not in target_vertices or v2.index not in target_vertices:
#                         # 3つ揃ってない場合、スルー
#                         continue

#                     # 方向に応じて判定値を変える
#                     if param_option['direction'] == '上':
#                         v0v = -v0.position.y()
#                         v1v = -v1.position.y()
#                         base_vertical_axis = MVector3D(0, 1, 0)
#                         base_horizonal_axis = MVector3D(1, 0, 0)
#                     elif param_option['direction'] == '右':
#                         v0v = v0.position.x()
#                         v1v = v1.position.x()
#                         base_vertical_axis = MVector3D(-1, 0, 0)
#                         base_horizonal_axis = MVector3D(0, -1, 0)
#                     elif param_option['direction'] == '左':
#                         v0v = -v0.position.x()
#                         v1v = -v1.position.x()
#                         base_vertical_axis = MVector3D(1, 0, 0)
#                         base_horizonal_axis = MVector3D(0, -1, 0)
#                     else:
#                         # デフォルトは下
#                         v0v = v0.position.y()
#                         v1v = v1.position.y()
#                         base_vertical_axis = MVector3D(0, -1, 0)
#                         base_horizonal_axis = MVector3D(1, 0, 0)

#                     v21_axis = (v2.position - v1.position).normalized()

#                     v10_axis = (v1.position - v0.position).normalized()
#                     v10_axis_cross = MVector3D.crossProduct(v10_axis, v21_axis).normalized()
#                     v10_axis_qq = MQuaternion.fromDirection(base_vertical_axis, v10_axis_cross)

#                     v10_mat = MMatrix4x4()
#                     v10_mat.setToIdentity()
#                     v10_mat.translate(v0.position)
#                     v10_mat.rotate(v10_axis_qq)

#                     v1_local_position = v10_mat.inverted() * v1.position

#                     below_x = MVector3D.dotProduct(v1_local_position.normalized(), base_vertical_axis)
#                     below_y = MVector3D.dotProduct(v1_local_position.normalized(), base_horizonal_axis)

#                     below_size = v0.position.distanceToPoint(v1.position) * v1.position.distanceToPoint(v2.position) * v2.position.distanceToPoint(v0.position)

#                     if v0v > v1v and abs(below_x) > max_below_x and below_size > max_below_x_size * 0.6 and (set(registed_iidxs) - set(non_target_iidxs) or \
#                        (not set(registed_iidxs) - set(non_target_iidxs) and ymin + (ymedian - ymin) * 0.1 < v0.position.y() < ymax - (ymax - ymedian) * 0.1)):
#                         logger.debug(f'vertical iidx[{index_idx}], v1_local_position[{v1_local_position.to_log()}], below_x[{below_x}], ' \
#                                      + f'below_size[{below_size}], below_x_iidx[{below_x_iidx}], max_below_x[{max_below_x}], max_below_x_size[{max_below_x_size}]')
#                         below_x_iidx = index_idx
#                         max_below_x = abs(below_x)
#                         max_below_x_size = below_size

#                     if v0v > v1v and abs(below_y) > max_below_y and below_size > max_below_y_size * 0.6:
#                         logger.debug(f'horizonal iidx[{index_idx}], v1_local_position[{v1_local_position.to_log()}], below_y[{below_y}], ' \
#                                      + f'below_size[{below_size}], below_y_iidx[{below_y_iidx}], max_below_y[{max_below_y}], max_below_y_size[{max_below_y_size}]')
#                         below_y_iidx = index_idx
#                         max_below_y = abs(below_y)
#                         max_below_y_size = below_size
                
#                 below_iidx = below_x_iidx if below_x_iidx and max_below_x > 0.97 and max_below_x > max_below_y else below_y_iidx if below_y_iidx else remaining_iidxs[0]

#                 logger.debug(f'below_iidx: {below_iidx}, max_below_x: {max_below_x}, max_below_y: {max_below_y}')
#                 first_vertex_axis_map, first_vertex_coordinate_map = \
#                     self.create_vertex_map_by_index(model, param_option, duplicate_vertices, {}, {}, below_iidx)
#                 vertex_axis_maps.append(first_vertex_axis_map)
#                 vertex_coordinate_maps.append(first_vertex_coordinate_map)
#                 registed_iidxs.append(below_iidx)
#                 vertical_iidxs.append(below_iidx)
                
#                 # 斜めが埋まってる場合、残りの一点を埋める
#                 vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, diagonal_now_iidxs = \
#                     self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
#                                                            vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, vertical_iidxs)

#                 # これで四角形が求められた
#                 registed_iidxs = list(set(registed_iidxs) | set(diagonal_now_iidxs))
#                 vertical_iidxs = list(set(vertical_iidxs) | set(diagonal_now_iidxs))

#             total_vertical_iidxs = []

#             if vertical_iidxs:
#                 now_vertical_iidxs = vertical_iidxs
#                 total_vertical_iidxs.extend(now_vertical_iidxs)

#                 logger.debug(f'縦初回: {total_vertical_iidxs}')

#                 # 縦辺がいる場合（まずは下方向）
#                 n = 0
#                 while n < 200:
#                     n += 1
#                     vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, now_vertical_iidxs \
#                         = self.fill_vertical_indices(model, param_option, duplicate_indices, duplicate_vertices, \
#                                                      vertex_axis_maps[-1], vertex_coordinate_maps[-1], indices_by_vpos, indices_by_vidx, \
#                                                      registed_iidxs, now_vertical_iidxs, 1)
#                     total_vertical_iidxs.extend(now_vertical_iidxs)
#                     logger.debug(f'縦下: {total_vertical_iidxs}')

#                     if not now_vertical_iidxs:
#                         break
                
#                 if not now_vertical_iidxs:
#                     now_vertical_iidxs = vertical_iidxs

#                     # 下方向が終わったら上方向
#                     n = 0
#                     while n < 200:
#                         n += 1
#                         vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, now_vertical_iidxs \
#                             = self.fill_vertical_indices(model, param_option, duplicate_indices, duplicate_vertices, \
#                                                          vertex_axis_maps[-1], vertex_coordinate_maps[-1], indices_by_vpos, indices_by_vidx, \
#                                                          registed_iidxs, now_vertical_iidxs, -1)
#                         total_vertical_iidxs.extend(now_vertical_iidxs)
#                         logger.debug(f'縦上: {total_vertical_iidxs}')

#                         if not now_vertical_iidxs:
#                             break

#                 logger.debug(f'縦一列: {total_vertical_iidxs} --------------')
                
#                 # 縦が終わった場合、横に移動する
#                 min_x, min_y, max_x, max_y = self.get_axis_range(model, vertex_coordinate_maps[-1], registed_iidxs)
#                 logger.debug(f'axis_range: min_x[{min_x}], min_y[{min_y}], max_x[{max_x}], max_y[{max_y}]')
                
#                 if not now_vertical_iidxs:
#                     # 左方向
#                     registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
#                                                                                       duplicate_vertices, registed_iidxs, min_x, min_y, max_y, -1)

#                     logger.debug(f'横左: {now_vertical_iidxs}')
                    
#                 if not now_vertical_iidxs:
#                     # 右方向
#                     registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
#                                                                                       duplicate_vertices, registed_iidxs, max_x, min_y, max_y, 1)
#                     logger.debug(f'横右: {now_vertical_iidxs}')
                
#                 vertical_iidxs = now_vertical_iidxs
                
#             if not vertical_iidxs:
#                 remaining_iidxs = list(set(model.material_indices[material_name]) - set(registed_iidxs))
#                 # 全頂点登録済みの面を潰していく
#                 for index_idx in remaining_iidxs:
#                     iv0, iv1, iv2 = model.indices[index_idx]
#                     if iv0 in vertex_axis_maps[-1] and iv1 in vertex_axis_maps[-1] and iv2 in vertex_axis_maps[-1]:
#                         registed_iidxs.append(index_idx)
#                         logger.debug(f'頂点潰し: {index_idx}')

#             if len(registed_iidxs) > 0 and len(registed_iidxs) // 200 > prev_index_cnt:
#                 logger.info("-- 面: %s個目:終了", len(registed_iidxs))
#                 prev_index_cnt = len(registed_iidxs) // 200
            
#         logger.info("-- 面: %s個目:終了", len(registed_iidxs))

#         logger.info("%s: 絶対頂点マップの生成", material_name)
#         vertex_maps = []
#         vertex_connecteds = []

#         for midx, (vertex_axis_map, vertex_coordinate_map) in enumerate(zip(vertex_axis_maps, vertex_coordinate_maps)):
#             logger.info("-- 絶対頂点マップ: %s個目: ---------", midx + 1)

#             # XYの最大と最小の抽出
#             xs = [k[0] for k in vertex_coordinate_map.keys()]
#             ys = [k[1] for k in vertex_coordinate_map.keys()]

#             # それぞれの出現回数から大体全部埋まってるのを抽出。その中の最大と最小を選ぶ
#             xu, x_counts = np.unique(xs, return_counts=True)
#             full_xs = [i for i, x in zip(xu, x_counts) if x >= max(x_counts) * 0.6]
#             logger.debug(f'絶対axis_range: xu[{xu}], x_counts[{x_counts}], full_xs[{full_xs}]')

#             min_x = min(full_xs)
#             max_x = max(full_xs)

#             yu, y_counts = np.unique(ys, return_counts=True)
#             full_ys = [i for i, y in zip(yu, y_counts) if y >= max(y_counts) * 0.2]
#             logger.debug(f'絶対axis_range: yu[{yu}], y_counts[{y_counts}], full_ys[{full_ys}]')

#             min_y = min(full_ys)
#             max_y = max(full_ys)

#             logger.debug(f'絶対axis_range: min_x[{min_x}], min_y[{min_y}], max_x[{max_x}], max_y[{max_y}]')
            
#             # 存在しない頂点INDEXで二次元配列初期化
#             vertex_map = np.full((max_y - min_y + 1, max_x - min_x + 1), -1)
#             vertex_display_map = np.full((max_y - min_y + 1, max_x - min_x + 1), ' None ')
#             vertex_connected = []
#             logger.debug(f'vertex_map.shape: {vertex_map.shape}')

#             for vmap in vertex_axis_map.values():
#                 if vertex_map.shape[0] > vmap['y'] - min_y and vertex_map.shape[1] > vmap['x'] - min_x:
#                     logger.debug(f"vertex_map: y[{vmap['y'] - min_y}], x[{vmap['x'] - min_x}]: vidx[{vmap['vidx']}] orgx[{vmap['x']}] orgy[{vmap['y']}] pos[{vmap['position'].to_log()}]")

#                     try:
#                         vertex_map[vmap['y'] - min_y, vmap['x'] - min_x] = vertex_coordinate_map[(vmap['x'], vmap['y'])][0]
#                         vertex_display_map[vmap['y'] - min_y, vmap['x'] - min_x] = ':'.join([str(v) for v in vertex_coordinate_map[(vmap['x'], vmap['y'])]])
#                     except Exception:
#                         # はみ出した頂点はスルーする
#                         pass

#             # 左端と右端で面が連続しているかチェック
#             for yi in range(vertex_map.shape[0]):
#                 is_connect = False
#                 if vertex_map[yi, 0] in model.vertex_dict and vertex_map[yi, -1] in model.vertex_dict:
#                     for (iv1, iv2) in list(itertools.product(duplicate_vertices[model.vertex_dict[vertex_map[yi, 0]].position.to_log()], \
#                                                              duplicate_vertices[model.vertex_dict[vertex_map[yi, -1]].position.to_log()])):
#                         if (min(iv1, iv2), max(iv1, iv2)) in duplicate_indices:
#                             is_connect = True
#                             break
#                 vertex_connected.append(is_connect)

#             logger.debug(f'vertex_connected: {vertex_connected}')

#             vertex_maps.append(vertex_map)
#             vertex_connecteds.append(vertex_connected)

#             logger.info('\n'.join([', '.join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]), translate=False)
#             logger.info("-- 絶対頂点マップ: %s個目:終了 ---------", midx + 1)

#         return vertex_maps, duplicate_vertices, registed_iidxs, duplicate_indices, index_combs_by_vpos
    
#     def get_axis_range(self, model: PmxModel, vertex_coordinate_map: dict, registed_iidxs: list):
#         xs = [k[0] for k in vertex_coordinate_map.keys()]
#         ys = [k[1] for k in vertex_coordinate_map.keys()]

#         min_x = min(xs)
#         max_x = max(xs)

#         min_y = min(ys)
#         max_y = max(ys)
        
#         return min_x, min_y, max_x, max_y
    
#     def fill_horizonal_now_idxs(self, model: PmxModel, param_option: dict, vertex_axis_map: dict, vertex_coordinate_map: dict, duplicate_indices: dict, \
#                                 duplicate_vertices: dict, registed_iidxs: list, first_x: int, min_y: int, max_y: int, offset: int):
#         now_iidxs = []
#         first_vidxs = None
#         second_vidxs = None
#         for first_y in range(min_y + int((max_y - min_y) / 2), min_y - 1, -1):
#             if (first_x, first_y) in vertex_coordinate_map:
#                 first_vidxs = vertex_coordinate_map[(first_x, first_y)]
#                 break

#         if first_vidxs:
#             for second_y in range(first_y + 1, max_y + 1):
#                 if (first_x, second_y) in vertex_coordinate_map:
#                     second_vidxs = vertex_coordinate_map[(first_x, second_y)]
#                     break

#         if first_vidxs and second_vidxs:
#             # 小さいINDEX・大きい頂点INDEXのセットでキー生成
#             for (iv1, iv2) in list(itertools.product(first_vidxs, second_vidxs)):
#                 key = (min(iv1, iv2), max(iv1, iv2))
#                 if key in duplicate_indices:
#                     for index_idx in duplicate_indices[key]:
#                         if index_idx in registed_iidxs + now_iidxs:
#                             continue
                        
#                         # 登録されてない残りの頂点INDEX
#                         remaining_vidx = tuple(set(model.indices[index_idx]) - set(duplicate_vertices[model.vertex_dict[iv1].position.to_log()]) \
#                             - set(duplicate_vertices[model.vertex_dict[iv2].position.to_log()]))[0]     # noqa
#                         remaining_vidxs = duplicate_vertices[model.vertex_dict[remaining_vidx].position.to_log()]
#                         if abs(model.vertex_dict[iv1].position.y() - model.vertex_dict[remaining_vidx].position.y()) == \
#                             abs(model.vertex_dict[iv2].position.y() - model.vertex_dict[remaining_vidx].position.y()):  # noqa
#                             ivy = vertex_axis_map[iv1]['y'] if model.vertex_dict[iv1].position.distanceToPoint(model.vertex_dict[remaining_vidx].position) < \
#                                 model.vertex_dict[iv2].position.distanceToPoint(model.vertex_dict[remaining_vidx].position) else vertex_axis_map[iv2]['y']
#                         else:
#                             ivy = vertex_axis_map[iv1]['y'] if abs(model.vertex_dict[iv1].position.y() - model.vertex_dict[remaining_vidx].position.y()) < \
#                                 abs(model.vertex_dict[iv2].position.y() - model.vertex_dict[remaining_vidx].position.y()) else vertex_axis_map[iv2]['y']
                        
#                         iv1_map = (vertex_axis_map[iv1]['x'] + offset, ivy)
#                         if iv1_map not in vertex_coordinate_map:
#                             is_regist = False
#                             for vidx in remaining_vidxs:
#                                 if vidx not in vertex_axis_map:
#                                     is_regist = True
#                                     vertex_axis_map[vidx] = {'vidx': vidx, 'x': iv1_map[0], 'y': iv1_map[1], 'position': model.vertex_dict[vidx].position}
#                                     logger.debug(f"fill_horizonal_now_idxs: vidx[{vidx}], axis[{vertex_axis_map[vidx]}]")
#                             if is_regist:
#                                 vertex_coordinate_map[iv1_map] = remaining_vidxs
#                                 logger.debug(f"fill_horizonal_now_idxs: key[{iv1_map}], v[{remaining_vidxs}], axis[{vertex_axis_map[vidx]}]")
#                             now_iidxs.append(index_idx)
                        
#                         if len(now_iidxs) > 0:
#                             break
#                 if len(now_iidxs) > 0:
#                     break
        
#         registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))
        
#         for index_idx in now_iidxs:
#             # 斜めが埋まってる場合、残りの一点を埋める
#             vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
#                 self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
#                                                        vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs)

#         return registed_iidxs, now_iidxs
    
#     def fill_diagonal_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
#                                           vertex_axis_map: dict, vertex_coordinate_map: dict, registed_iidxs: list, now_iidxs: list):

#         # 斜めが埋まっている場合、残りの一点を求める（四角形を求められる）
#         for index_idx in now_iidxs:
#             # 面の辺を抽出
#             _, _, diagonal_vs = self.judge_index_edge(model, vertex_axis_map, index_idx)

#             if diagonal_vs and diagonal_vs in duplicate_indices:
#                 for iidx in duplicate_indices[diagonal_vs]:
#                     edge_size = len(set(model.indices[iidx]) & set(vertex_axis_map.keys()))
#                     if edge_size >= 2:
#                         if edge_size == 2:
#                             # 重複頂点(2つの頂点)を持つ面(=連続面)
#                             vertex_axis_map, vertex_coordinate_map = \
#                                 self.create_vertex_map_by_index(model, param_option, duplicate_vertices, \
#                                                                 vertex_axis_map, vertex_coordinate_map, iidx)
                        
#                         # 登録済みでなければ保持
#                         if iidx not in now_iidxs:
#                             now_iidxs.append(iidx)

#         registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))

#         return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs
    
#     def fill_vertical_indices(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
#                               vertex_axis_map: dict, vertex_coordinate_map: dict, indices_by_vpos: dict, indices_by_vidx: dict, \
#                               registed_iidxs: list, vertical_iidxs: list, offset: int):
#         vertical_vs_list = []

#         for index_idx in vertical_iidxs:
#             # 面の辺を抽出
#             vertical_vs, _, _ = self.judge_index_edge(model, vertex_axis_map, index_idx)
#             if not vertical_vs:
#                 continue

#             if vertical_vs not in vertical_vs_list:
#                 vertical_vs_list.append(vertical_vs)

#         now_iidxs = []

#         if vertical_vs_list:
#             # 縦が埋まっている場合、重複頂点から縦方向のベクトルが近いものを抽出する
#             vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
#                 self.fill_vertical_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, \
#                                                        vertex_axis_map, vertex_coordinate_map, indices_by_vpos, \
#                                                        indices_by_vidx, vertical_vs_list, registed_iidxs, vertical_iidxs, offset)

#         return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs

#     def fill_vertical_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_indices: dict, duplicate_vertices: dict, \
#                                           vertex_axis_map: dict, vertex_coordinate_map: dict, indices_by_vpos: dict, indices_by_vidx: dict, \
#                                           vertical_vs_list: list, registed_iidxs: list, vertical_iidxs: list, offset: int):
#         horizonaled_duplicate_indexs = []
#         horizonaled_index_combs = []
#         horizonaled_duplicate_dots = []
#         horizonaled_vertical_above_v = []
#         horizonaled_vertical_below_v = []
#         not_horizonaled_duplicate_indexs = []
#         not_horizonaled_index_combs = []
#         not_horizonaled_duplicate_dots = []
#         not_horizonaled_vertical_above_v = []
#         not_horizonaled_vertical_below_v = []

#         now_iidxs = []
#         for vertical_vs in vertical_vs_list:
#             # 該当縦辺の頂点(0が上(＋大きい))
#             v0 = model.vertex_dict[vertical_vs[0]]
#             v1 = model.vertex_dict[vertical_vs[1]]

#             if offset > 0:
#                 # 下方向の走査
#                 vertical_vec = v0.position - v1.position
#                 vertical_above_v = v0
#                 vertical_below_v = v1
#             else:
#                 # 上方向の走査
#                 vertical_vec = v1.position - v0.position
#                 vertical_above_v = v1
#                 vertical_below_v = v0

#             if vertical_below_v.position.to_log() in indices_by_vpos:
#                 for duplicate_index_idx in indices_by_vpos[vertical_below_v.position.to_log()]:
#                     if duplicate_index_idx in registed_iidxs + vertical_iidxs + now_iidxs:
#                         # 既に登録済みの面である場合、スルー
#                         continue

#                     # 面の辺を抽出
#                     vertical_in_vs, horizonal_in_vs, _ = self.judge_index_edge(model, vertex_axis_map, duplicate_index_idx)

#                     if vertical_in_vs and horizonal_in_vs:
#                         if ((offset > 0 and vertical_in_vs[0] in duplicate_vertices[vertical_below_v.position.to_log()]) \
#                            or (offset < 0 and vertical_in_vs[1] in duplicate_vertices[vertical_below_v.position.to_log()])):
#                             # 既に縦辺が求められていてそれに今回算出対象が含まれている場合
#                             # 縦も横も求められているなら、該当面は必ず対象となる
#                             horizonaled_duplicate_indexs.append(duplicate_index_idx)
#                             horizonaled_vertical_below_v.append(vertical_below_v)
#                             if offset > 0:
#                                 horizonaled_index_combs.append((vertical_in_vs[0], vertical_in_vs[1]))
#                             else:
#                                 horizonaled_index_combs.append((vertical_in_vs[1], vertical_in_vs[0]))
#                             horizonaled_duplicate_dots.append(1)
#                         else:
#                             # 既に縦辺が求められていてそれに今回算出対象が含まれていない場合、スルー
#                             continue

#                     # 重複辺（2点）の組み合わせ
#                     index_combs = list(itertools.combinations(model.indices[duplicate_index_idx], 2))
#                     for (iv0_comb_idx, iv1_comb_idx) in index_combs:
#                         if horizonal_in_vs:
#                             horizonaled_duplicate_indexs.append(duplicate_index_idx)
#                             horizonaled_vertical_below_v.append(vertical_below_v)
#                             horizonaled_vertical_above_v.append(vertical_above_v)

#                             iv0 = None
#                             iv1 = None

#                             if iv0_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv1_comb_idx) not in horizonaled_index_combs:
#                                 iv0 = model.vertex_dict[iv0_comb_idx]
#                                 iv1 = model.vertex_dict[iv1_comb_idx]
#                                 horizonaled_index_combs.append((vertical_below_v.index, iv1_comb_idx))
#                             elif iv1_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv0_comb_idx) not in horizonaled_index_combs:
#                                 iv0 = model.vertex_dict[iv1_comb_idx]
#                                 iv1 = model.vertex_dict[iv0_comb_idx]
#                                 horizonaled_index_combs.append((vertical_below_v.index, iv0_comb_idx))
#                             else:
#                                 horizonaled_index_combs.append((-1, -1))

#                             if iv0 and iv1:
#                                 if iv0.index in vertex_axis_map and (vertex_axis_map[iv0.index]['x'], vertex_axis_map[iv0.index]['y'] + offset) not in vertex_coordinate_map:
#                                     # v1から繋がる辺のベクトル
#                                     iv0 = model.vertex_dict[iv0.index]
#                                     iv1 = model.vertex_dict[iv1.index]
#                                     duplicate_vec = (iv0.position - iv1.position)
#                                     horizonaled_duplicate_dots.append(MVector3D.dotProduct(vertical_vec.normalized(), duplicate_vec.normalized()))
#                                 else:
#                                     horizonaled_duplicate_dots.append(0)
#                             else:
#                                 horizonaled_duplicate_dots.append(0)
#                         else:
#                             not_horizonaled_duplicate_indexs.append(duplicate_index_idx)
#                             not_horizonaled_vertical_below_v.append(vertical_below_v)
#                             not_horizonaled_vertical_above_v.append(vertical_above_v)

#                             iv0 = None
#                             iv1 = None

#                             if iv0_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv1_comb_idx) not in not_horizonaled_index_combs \
#                                    and (vertical_below_v.index, iv1_comb_idx) not in horizonaled_index_combs:   # noqa
#                                 iv0 = model.vertex_dict[iv0_comb_idx]
#                                 iv1 = model.vertex_dict[iv1_comb_idx]
#                                 not_horizonaled_index_combs.append((vertical_below_v.index, iv1_comb_idx))
#                             elif iv1_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv0_comb_idx) not in not_horizonaled_index_combs \
#                                     and (vertical_below_v.index, iv0_comb_idx) not in horizonaled_index_combs:  # noqa
#                                 iv0 = model.vertex_dict[iv1_comb_idx]
#                                 iv1 = model.vertex_dict[iv0_comb_idx]
#                                 not_horizonaled_index_combs.append((vertical_below_v.index, iv0_comb_idx))
#                             else:
#                                 not_horizonaled_index_combs.append((-1, -1))

#                             if iv0 and iv1:
#                                 if iv0.index in vertex_axis_map and (vertex_axis_map[iv0.index]['x'], vertex_axis_map[iv0.index]['y'] + offset) not in vertex_coordinate_map:
#                                     # v1から繋がる辺のベクトル
#                                     iv0 = model.vertex_dict[iv0.index]
#                                     iv1 = model.vertex_dict[iv1.index]
#                                     duplicate_vec = (iv0.position - iv1.position)
#                                     not_horizonaled_duplicate_dots.append(MVector3D.dotProduct(vertical_vec.normalized(), duplicate_vec.normalized()))
#                                 else:
#                                     not_horizonaled_duplicate_dots.append(0)
#                             else:
#                                 not_horizonaled_duplicate_dots.append(0)

#         if len(horizonaled_duplicate_dots) > 0 and np.max(horizonaled_duplicate_dots) >= param_option['similarity']:
#             logger.debug(f"fill_vertical: vertical_vs_list[{vertical_vs_list}], horizonaled_duplicate_dots[{horizonaled_duplicate_dots}], horizonaled_index_combs[{horizonaled_index_combs}]")

#             full_d = [i for i, d in enumerate(horizonaled_duplicate_dots) if np.round(d, decimals=5) == np.max(np.round(horizonaled_duplicate_dots, decimals=5))]  # noqa
#             not_full_d = [i for i, d in enumerate(not_horizonaled_duplicate_dots) if np.round(d, decimals=5) > np.max(np.round(horizonaled_duplicate_dots, decimals=5)) + 0.05]  # noqa
#             # not_full_d = []
#             if full_d:
#                 if not_full_d:
#                     # 平行辺の内積より一定以上近い内積のINDEX組合せがあった場合、臨時採用
#                     for vidx in not_full_d:
#                         # 正方向に繋がる重複辺があり、かつそれが一定以上の場合、採用
#                         vertical_vidxs = not_horizonaled_index_combs[vidx]
#                         duplicate_index_idx = not_horizonaled_duplicate_indexs[vidx]
#                         vertical_below_v = not_horizonaled_vertical_below_v[vidx]
#                         vertical_above_v = not_horizonaled_vertical_above_v[vidx]

#                         remaining_x = vertex_axis_map[vertical_below_v.index]['x']
#                         remaining_y = vertex_axis_map[vertical_below_v.index]['y'] + offset
#                         remaining_vidx = tuple(set(vertical_vidxs) - {vertical_below_v.index})[0]
#                         remaining_v = model.vertex_dict[remaining_vidx]
#                         # ほぼ同じベクトルを向いていたら、垂直頂点として登録
#                         is_regist = False
#                         for below_vidx in duplicate_vertices[remaining_v.position.to_log()]:
#                             if below_vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
#                                 is_regist = True
#                                 vertex_axis_map[below_vidx] = {'vidx': below_vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[below_vidx].position}
#                                 logger.debug(f"fill_vertical1: vidx[{below_vidx}], axis[{vertex_axis_map[below_vidx]}]")
#                         if is_regist:
#                             vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
#                             logger.debug(f"fill_vertical1: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}], axis[{vertex_axis_map[below_vidx]}]")

#                         now_iidxs.append(duplicate_index_idx)
#                 else:
#                     vidx = full_d[0]

#                     # 正方向に繋がる重複辺があり、かつそれが一定以上の場合、採用
#                     vertical_vidxs = horizonaled_index_combs[vidx]
#                     duplicate_index_idx = horizonaled_duplicate_indexs[vidx]
#                     vertical_below_v = horizonaled_vertical_below_v[vidx]

#                     remaining_x = vertex_axis_map[vertical_below_v.index]['x']
#                     remaining_y = vertex_axis_map[vertical_below_v.index]['y'] + offset
#                     remaining_vidx = tuple(set(vertical_vidxs) - {vertical_below_v.index})[0]
#                     remaining_v = model.vertex_dict[remaining_vidx]
#                     # ほぼ同じベクトルを向いていたら、垂直頂点として登録
#                     is_regist = False
#                     for below_vidx in duplicate_vertices[remaining_v.position.to_log()]:
#                         if below_vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
#                             is_regist = True
#                             vertex_axis_map[below_vidx] = {'vidx': below_vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[below_vidx].position}
#                             logger.debug(f"fill_vertical1: vidx[{below_vidx}], axis[{vertex_axis_map[below_vidx]}]")
#                     if is_regist:
#                         vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
#                         logger.debug(f"fill_vertical1: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}], axis[{vertex_axis_map[below_vidx]}]")

#                     now_iidxs.append(duplicate_index_idx)

#                     if vertical_vidxs[0] in vertex_axis_map and vertical_vidxs[1] in vertex_axis_map:
#                         vertical_v0 = vertex_axis_map[vertical_vidxs[0]]
#                         vertical_v1 = vertex_axis_map[vertical_vidxs[1]]
#                         remaining_v = model.vertex_dict[tuple(set(model.indices[duplicate_index_idx]) - set(vertical_vidxs))[0]]

#                         if remaining_v.index not in vertex_axis_map:
#                             # 残り一点のマップ位置
#                             remaining_x, remaining_y = self.get_remaining_vertex_vec(vertical_v0['vidx'], vertical_v0['x'], vertical_v0['y'], vertical_v0['position'], \
#                                                                                      vertical_v1['vidx'], vertical_v1['x'], vertical_v1['y'], vertical_v1['position'], \
#                                                                                      remaining_v, vertex_coordinate_map)

#                             is_regist = False
#                             for vidx in duplicate_vertices[remaining_v.position.to_log()]:
#                                 if vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
#                                     is_regist = True
#                                     vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
#                                     logger.debug(f"fill_vertical2: vidx[{vidx}], axis[{vertex_axis_map[vidx]}]")
#                             if is_regist:
#                                 vertex_coordinate_map[(remaining_x, remaining_y)] = duplicate_vertices[remaining_v.position.to_log()]
#                                 logger.debug(f"fill_vertical2: key[{(remaining_x, remaining_y)}], v[{duplicate_vertices[remaining_v.position.to_log()]}], axis[{vertex_axis_map[vidx]}]")

#                         # 斜めが埋められそうなら埋める
#                         vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs = \
#                             self.fill_diagonal_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, vertex_axis_map, \
#                                                                    vertex_coordinate_map, registed_iidxs, now_iidxs)
        
#         registed_iidxs = list(set(registed_iidxs) | set(now_iidxs))

#         return vertex_axis_map, vertex_coordinate_map, registed_iidxs, now_iidxs

#     def judge_index_edge(self, model: PmxModel, vertex_axis_map: dict, index_idx: int):
#         # 該当面の頂点
#         v0 = model.vertex_dict[model.indices[index_idx][0]]
#         v1 = model.vertex_dict[model.indices[index_idx][1]]
#         v2 = model.vertex_dict[model.indices[index_idx][2]]

#         # 縦の辺を抽出
#         vertical_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map and vertex_axis_map[v0.index]['x'] == vertex_axis_map[v1.index]['x'] \
#             else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v0.index]['x'] == vertex_axis_map[v2.index]['x'] \
#             else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v1.index]['x'] == vertex_axis_map[v2.index]['x'] else None
#         if vertical_vs:
#             vertical_vs = (vertical_vs[0], vertical_vs[1]) if vertex_axis_map[vertical_vs[0]]['y'] < vertex_axis_map[vertical_vs[1]]['y'] else (vertical_vs[1], vertical_vs[0])

#         # 横の辺を抽出
#         horizonal_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v1.index]['y'] \
#             else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v2.index]['y'] \
#             else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v1.index]['y'] == vertex_axis_map[v2.index]['y'] else None
#         if horizonal_vs:
#             horizonal_vs = (horizonal_vs[0], horizonal_vs[1]) if vertex_axis_map[horizonal_vs[0]]['x'] < vertex_axis_map[horizonal_vs[1]]['x'] else (horizonal_vs[1], horizonal_vs[0])

#         # 斜めの辺を抽出
#         diagonal_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map \
#             and vertex_axis_map[v0.index]['x'] != vertex_axis_map[v1.index]['x'] and vertex_axis_map[v0.index]['y'] != vertex_axis_map[v1.index]['y'] \
#             else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map \
#             and vertex_axis_map[v0.index]['x'] != vertex_axis_map[v2.index]['x'] and vertex_axis_map[v0.index]['y'] != vertex_axis_map[v2.index]['y'] \
#             else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map \
#             and vertex_axis_map[v1.index]['x'] != vertex_axis_map[v2.index]['x'] and vertex_axis_map[v1.index]['y'] != vertex_axis_map[v2.index]['y'] else None
#         if diagonal_vs:
#             diagonal_vs = (min(diagonal_vs[0], diagonal_vs[1]), max(diagonal_vs[0], diagonal_vs[1]))

#         return vertical_vs, horizonal_vs, diagonal_vs

#     def create_vertex_map_by_index(self, model: PmxModel, param_option: dict, duplicate_vertices: dict, \
#                                    vertex_axis_map: dict, vertex_coordinate_map: dict, index_idx: int):
#         # 該当面の頂点
#         v0 = model.vertex_dict[model.indices[index_idx][0]]
#         v1 = model.vertex_dict[model.indices[index_idx][1]]
#         v2 = model.vertex_dict[model.indices[index_idx][2]]

#         # 重複を含む頂点一覧
#         vs_duplicated = {}
#         vs_duplicated[v0.index] = duplicate_vertices[v0.position.to_log()]
#         vs_duplicated[v1.index] = duplicate_vertices[v1.position.to_log()]
#         vs_duplicated[v2.index] = duplicate_vertices[v2.position.to_log()]

#         if not vertex_axis_map:
#             # 空の場合、原点として0番目を設定する
#             # 表向き=時計回りで当てはめていく
#             for vidx in vs_duplicated[v0.index]:
#                 vertex_axis_map[vidx] = {'vidx': vidx, 'x': 0, 'y': 0, 'position': model.vertex_dict[vidx].position}
#             vertex_coordinate_map[(0, 0)] = vs_duplicated[v0.index]

#             for vidx in vs_duplicated[v1.index]:

#                 # 方向に応じて判定値を変える
#                 if param_option['direction'] == '上':
#                     v0v = -v0.position.y()
#                     v1v = -v1.position.y()
#                     v2v = -v2.position.y()
#                     base_vertical_axis = MVector3D(0, 1, 0)
#                     base_horizonal_axis = MVector3D(1, 0, 0)
#                 elif param_option['direction'] == '右':
#                     v0v = v0.position.x()
#                     v1v = v1.position.x()
#                     v2v = v2.position.x()
#                     base_vertical_axis = MVector3D(-1, 0, 0)
#                     base_horizonal_axis = MVector3D(0, -1, 0)
#                 elif param_option['direction'] == '左':
#                     v0v = -v0.position.x()
#                     v1v = -v1.position.x()
#                     v2v = -v2.position.x()
#                     base_vertical_axis = MVector3D(1, 0, 0)
#                     base_horizonal_axis = MVector3D(0, -1, 0)
#                 else:
#                     # デフォルトは下
#                     v0v = v0.position.y()
#                     v1v = v1.position.y()
#                     v2v = v2.position.y()
#                     base_vertical_axis = MVector3D(0, -1, 0)
#                     base_horizonal_axis = MVector3D(1, 0, 0)
                
#                 parent_bone = model.bones[param_option['parent_bone_name']]
#                 is_horizonal = round(v0.position.y(), 2) == round(v1.position.y(), 2) == round(v2.position.y(), 2)

#                 v21_axis = (v2.position - v1.position).normalized()

#                 v10_axis = (v1.position - v0.position).normalized()
#                 v10_axis_cross = MVector3D.crossProduct(v10_axis, v21_axis).normalized()
#                 v10_axis_qq = MQuaternion.fromDirection(base_vertical_axis, v10_axis_cross)

#                 v10_mat = MMatrix4x4()
#                 v10_mat.setToIdentity()
#                 v10_mat.translate(v0.position)
#                 v10_mat.rotate(v10_axis_qq)

#                 v1_local_position = v10_mat.inverted() * v1.position
#                 v2_local_position = v10_mat.inverted() * v2.position

#                 v1_vertical_dot = MVector3D.dotProduct(v1_local_position.normalized(), base_vertical_axis)
#                 v2_vertical_dot = MVector3D.dotProduct(v2_local_position.normalized(), base_vertical_axis)
#                 v1_horizonal_dot = MVector3D.dotProduct(v1_local_position.normalized(), base_horizonal_axis)
#                 v2_horizonal_dot = MVector3D.dotProduct(v2_local_position.normalized(), base_horizonal_axis)

#                 vertical_didx = np.argmax(np.abs([v1_vertical_dot, v2_vertical_dot]))
#                 horizonal_didx = np.argmax(np.abs([v1_horizonal_dot, v2_horizonal_dot]))
#                 direction_idxs = tuple(np.argsort(np.abs([v1_vertical_dot, v2_vertical_dot, v1_horizonal_dot, v2_horizonal_dot])))

#                 v1_vertical_sign = round(v1_vertical_dot, 2)
#                 v2_vertical_sign = round(v2_vertical_dot, 2)
#                 v1_horizonal_sign = round(v1_horizonal_dot, 2)
#                 v2_horizonal_sign = round(v2_horizonal_dot, 2)

#                 # より親ボーンに近い方が上
#                 v1_vertical_direction = 1 if parent_bone.position.distanceToPoint(v0.position) < parent_bone.position.distanceToPoint(v1.position) else -1
#                 v2_vertical_direction = 1 if parent_bone.position.distanceToPoint(v0.position) < parent_bone.position.distanceToPoint(v2.position) else -1

#                 logger.debug(f"direction[{param_option['direction']}], v0v[{v0v}], v1v[{v1v}], v2v[{v2v}], is_horizonal[{is_horizonal}]")

#                 logger.debug(f"v1[{v1.position.to_log()}], vertical[{v1_local_position.to_log()}], " \
#                              + f"v1_vertical_dot[{v1_vertical_dot}], v1_horizonal_dot[{v1_horizonal_dot}]")
#                 logger.debug(f"v2[{v2.position.to_log()}], vertical[{v2_local_position.to_log()}], " \
#                              + f"v2_vertical_dot[{v2_vertical_dot}], v2_horizonal_dot[{v2_horizonal_dot}]")

#                 logger.debug(f"vertical_didx[{vertical_didx}], horizonal_didx[{horizonal_didx}], direction_idxs[{direction_idxs}]")
#                 logger.debug(f"v1_vertical_direction[{v1_vertical_direction}], v2_vertical_direction[{v2_vertical_direction}]")

#                 if v1_vertical_sign == 0 and v2_vertical_sign == 0:
#                     # vertical がどちらも0の場合このルート(垂直にメッシュが並んでいる場合)
#                     if v2_horizonal_sign == 0:
#                         if v1_horizonal_sign > 0:
#                             # v1-v0: 水平, v2-v1: 垂直, v0-v2: 斜め
#                             remaining_x = 0
#                             remaining_y = v2_vertical_direction

#                             vx = 1 if v2_vertical_direction == 1 else -1
#                             vy = 0
#                         else:
#                             # v1-v0: 斜め, v2-v1: 垂直, v0-v2: 水平
#                             remaining_x = 0
#                             remaining_y = v2_vertical_direction

#                             vx = 1 if v2_vertical_direction == 1 else -1
#                             vy = v2_vertical_direction
#                     elif abs(v1_horizonal_sign) < abs(v2_horizonal_sign):
#                         # v1-v0: 垂直, v2-v1: 水平, v0-v2: 斜め
#                         vx = 0
#                         vy = v1_vertical_direction

#                         remaining_x = -1 if v1_vertical_direction == 1 else 1
#                         remaining_y = 0
#                     else:
#                         # v1-v0: 水平, v2-v1: 斜め, v0-v2: 垂直
#                         remaining_x = 1 if v2_vertical_direction == 1 else -1
#                         remaining_y = v2_vertical_direction

#                         vx = int(remaining_x)
#                         vy = 0
#                 elif (vertical_didx, horizonal_didx) == (0, 1):
#                     if abs(v2_horizonal_sign) == 1:
#                         if abs(v1_vertical_sign) < abs(v1_horizonal_sign):
#                             # v1-v0: 斜め, v2-v1: 水平, v0-v2: 垂直
#                             vx = -1 if v1_vertical_direction == 1 else 1
#                             vy = v1_vertical_direction

#                             remaining_x = int(vx)
#                             remaining_y = 0
#                         else:
#                             # v1-v0: 垂直, v2-v1: 水平, v0-v2: 斜め
#                             vx = 0
#                             vy = v1_vertical_direction

#                             remaining_x = -1 if v1_vertical_direction == 1 else 1
#                             remaining_y = 0
#                     elif abs(v1_vertical_sign) >= abs(v1_horizonal_sign):
#                         # v1-v0: 垂直, v2-v1: 斜め, v0-v2: 水平
#                         vx = 0
#                         vy = v1_vertical_direction

#                         remaining_x = -1 if v1_vertical_direction == 1 else 1
#                         remaining_y = int(vy)
#                     elif abs(v2_vertical_sign) < abs(v2_horizonal_sign):
#                         # v1-v0: 斜め, v2-v1: 水平, v0-v2: 垂直
#                         vx = -1 if v1_vertical_direction == 1 else 1
#                         vy = v1_vertical_direction

#                         remaining_x = int(vx)
#                         remaining_y = 0
#                     elif abs(v1_vertical_sign) < abs(v1_horizonal_sign):
#                         # v1-v0: 斜め, v2-v1: 水平, v0-v2: 垂直
#                         vx = -1 if v1_vertical_direction == 1 else 1
#                         vy = v1_vertical_direction

#                         remaining_x = int(vx)
#                         remaining_y = 0
#                     else:
#                         # v1-v0: 垂直, v2-v1: 水平, v0-v2: 斜め
#                         vx = 0
#                         vy = v1_vertical_direction

#                         remaining_x = -1 if v1_vertical_direction == 1 else 1
#                         remaining_y = 0
#                 elif (vertical_didx, horizonal_didx) == (1, 0):
#                     if abs(v1_horizonal_sign) == 1:
#                         if abs(v2_vertical_sign) < abs(v2_horizonal_sign) or (v1_horizonal_sign < 0 and v2_horizonal_sign < 0):
#                             # v1-v0: 水平, v2-v1: 垂直, v0-v2: 斜め
#                             remaining_x = 1 if v2_vertical_direction == 1 else -1
#                             remaining_y = v2_vertical_direction

#                             vx = int(remaining_x)
#                             vy = 0
#                         else:
#                             # v1-v0: 水平, v2-v1: 斜め, v0-v2: 垂直
#                             remaining_x = 0
#                             remaining_y = v2_vertical_direction

#                             vx = 1 if v2_vertical_direction == 1 else -1
#                             vy = 0
#                     elif abs(v2_vertical_sign) >= abs(v2_horizonal_sign):
#                         # v1-v0: 斜め, v2-v1: 垂直, v0-v2: 水平
#                         remaining_x = 0
#                         remaining_y = v2_vertical_direction

#                         vx = 1 if v2_vertical_direction == 1 else -1
#                         vy = int(remaining_y)
#                     elif abs(v1_vertical_sign) < abs(v1_horizonal_sign):
#                         # v1-v0: 水平, v2-v1: 斜め, v0-v2: 垂直
#                         remaining_x = 1 if v2_vertical_direction == 1 else -1
#                         remaining_y = v2_vertical_direction

#                         vx = int(remaining_x)
#                         vy = 0
#                     elif abs(v2_vertical_sign) < abs(v2_horizonal_sign):
#                         # v1-v0: 垂直, v2-v1: 水平, v0-v2: 斜め
#                         vx = 0
#                         vy = v1_vertical_direction

#                         remaining_x = -1 if v1_vertical_direction == 1 else 1
#                         remaining_y = 0
#                     else:
#                         # v1-v0: 水平, v2-v1: 斜め, v0-v2: 垂直
#                         remaining_x = 1 if v2_vertical_direction == 1 else -1
#                         remaining_y = v2_vertical_direction

#                         vx = int(remaining_x)
#                         vy = 0

#                 vertex_axis_map[vidx] = {'vidx': vidx, 'x': vx, 'y': vy, 'position': model.vertex_dict[vidx].position, 'duplicate': duplicate_vertices[model.vertex_dict[vidx].position.to_log()]}
#             vertex_coordinate_map[(vx, vy)] = vs_duplicated[v1.index]

#             for vidx in vs_duplicated[v2.index]:
#                 vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
#             vertex_coordinate_map[(remaining_x, remaining_y)] = vs_duplicated[v2.index]

#             logger.debug(f"初期iidx: iidx[{index_idx}], coodinate[{vertex_coordinate_map}]")
#         else:
#             # 残りの頂点INDEX
#             remaining_v = None
            
#             # 重複辺のマップ情報（時計回りで設定する）
#             v_duplicated_maps = []
#             if v0.index not in vertex_axis_map:
#                 remaining_v = v0
#                 v_duplicated_maps.append(vertex_axis_map[v1.index])
#                 v_duplicated_maps.append(vertex_axis_map[v2.index])

#             if v1.index not in vertex_axis_map:
#                 remaining_v = v1
#                 v_duplicated_maps.append(vertex_axis_map[v2.index])
#                 v_duplicated_maps.append(vertex_axis_map[v0.index])

#             if v2.index not in vertex_axis_map:
#                 remaining_v = v2
#                 v_duplicated_maps.append(vertex_axis_map[v0.index])
#                 v_duplicated_maps.append(vertex_axis_map[v1.index])
            
#             # 残り一点のマップ位置
#             remaining_x, remaining_y = self.get_remaining_vertex_vec(v_duplicated_maps[0]['vidx'], v_duplicated_maps[0]['x'], v_duplicated_maps[0]['y'], v_duplicated_maps[0]['position'], \
#                                                                      v_duplicated_maps[1]['vidx'], v_duplicated_maps[1]['x'], v_duplicated_maps[1]['y'], v_duplicated_maps[1]['position'], \
#                                                                      remaining_v, vertex_coordinate_map)

#             is_regist = False
#             for vidx in vs_duplicated[remaining_v.index]:
#                 if vidx not in vertex_axis_map and (remaining_x, remaining_y) not in vertex_coordinate_map:
#                     is_regist = True
#                     vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
#                     logger.debug(f"create_vertex_map_by_index: vidx[{vidx}], axis[{vertex_axis_map[vidx]}]")
#             if is_regist:
#                 vertex_coordinate_map[(remaining_x, remaining_y)] = vs_duplicated[remaining_v.index]

#         return vertex_axis_map, vertex_coordinate_map

#     def get_remaining_vertex_vec(self, vv0_idx: int, vv0_x: int, vv0_y: int, vv0_vec: MVector3D, \
#                                  vv1_idx: int, vv1_x: int, vv1_y: int, vv1_vec: MVector3D, remaining_v: Vertex, vertex_coordinate_map: dict):
#         # 時計回りと見なして位置を合わせる
#         if vv0_x == vv1_x:
#             # 元が縦方向に一致している場合
#             if vv0_y > vv1_y:
#                 remaining_x = vv0_x + 1
#             else:
#                 remaining_x = vv0_x - 1

#             if (remaining_x, vv0_y) in vertex_coordinate_map:
#                 remaining_y = vv1_y
#                 logger.debug(f"get_remaining_vertex_vec(縦): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#             elif (remaining_x, vv1_y) in vertex_coordinate_map:
#                 remaining_y = vv0_y
#                 logger.debug(f"get_remaining_vertex_vec(縦): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#             else:
#                 remaining_y = vv1_y if vv1_vec.distanceToPoint(remaining_v.position) < vv0_vec.distanceToPoint(remaining_v.position) else vv0_y
#                 logger.debug(f"get_remaining_vertex_vec(縦計算): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")

#         elif vv0_y == vv1_y:
#             # 元が横方向に一致している場合

#             if vv0_x < vv1_x:
#                 remaining_y = vv0_y + 1
#             else:
#                 remaining_y = vv0_y - 1
            
#             if (vv0_x, remaining_y) in vertex_coordinate_map:
#                 remaining_x = vv1_x
#                 logger.debug(f"get_remaining_vertex_vec(横): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#             elif (vv1_x, remaining_y) in vertex_coordinate_map:
#                 remaining_x = vv0_x
#                 logger.debug(f"get_remaining_vertex_vec(横): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#             else:
#                 remaining_x = vv1_x if vv1_vec.distanceToPoint(remaining_v.position) < vv0_vec.distanceToPoint(remaining_v.position) else vv0_x
#                 logger.debug(f"get_remaining_vertex_vec(横計算): [{remaining_v.index}], {remaining_x}, {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#         else:
#             # 斜めが一致している場合
#             if (vv0_x > vv1_x and vv0_y < vv1_y) or (vv0_x < vv1_x and vv0_y > vv1_y):
#                 # ／↑の場合、↓、↓／の場合、↑、／←の場合、→
#                 remaining_x = vv1_x
#                 remaining_y = vv0_y
#                 logger.debug(f"get_remaining_vertex_vec(斜1): ([{remaining_v.index}], {remaining_x}), {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
#             else:
#                 # ＼←の場合、→、／→の場合、←
#                 remaining_x = vv0_x
#                 remaining_y = vv1_y
#                 logger.debug(f"get_remaining_vertex_vec(斜2): ([{remaining_v.index}], {remaining_x}), {remaining_y} (v0[{vv0_idx}], ({vv0_x}, {vv0_y})) (v0[{vv1_idx}], ({vv1_x}, {vv1_y}))")
        
#         return remaining_x, remaining_y

#     def get_rigidbody(self, model: PmxModel, bone_name: str):
#         if bone_name not in model.bones:
#             return None

#         for rigidbody in model.rigidbodies.values():
#             if rigidbody.bone_index == model.bones[bone_name].index:
#                 return rigidbody
        
#         return None


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
