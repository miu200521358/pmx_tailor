# -*- coding: utf-8 -*-
#
import logging
import os
import traceback
import numpy as np
import itertools

from module.MOptions import MExportOptions
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, RigidBodyParam, IkLink, Ik # noqa
from mmd.PmxWriter import PmxWriter
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException

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
                service_data_txt = f"{service_data_txt}\n　　空気抵抗: {param_option['air_resistance']}"    # noqa
                service_data_txt = f"{service_data_txt}\n　　形状維持: {param_option['shape_maintenance']}"    # noqa

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

        logger.info(f"{param_option['material_name']}: 頂点マップ生成", decoration=MLogger.DECORATION_LINE)

        vertex_maps, vertex_connecteds, duplicate_vertices = self.create_vertex_map(model, param_option)
        
        # 各頂点の有効INDEX数が最も多いものをベースとする
        map_cnt = []
        for vertex_map in vertex_maps:
            map_cnt.append(np.count_nonzero(vertex_map >= 0))
        
        if len(map_cnt) == 0:
            logger.warning("有効な頂点マップが生成されななかった為、処理を終了します", decoration=MLogger.DECORATION_BOX)
            return False
        
        base_map_idx = np.argmax(map_cnt)

        logger.info(f"{param_option['material_name']}: 基準頂点マップ = No.{base_map_idx + 1}", decoration=MLogger.DECORATION_LINE)

        logger.info(f"{param_option['material_name']}(No.{base_map_idx + 1}): ボーン生成", decoration=MLogger.DECORATION_LINE)

        root_bone, registed_bone_indexs, bone_horizonal_distances, bone_vertical_distances \
            = self.create_bone(model, param_option, vertex_maps[base_map_idx], vertex_connecteds[base_map_idx])

        logger.info(f"{param_option['material_name']}(No.{base_map_idx + 1}): ウェイト分布", decoration=MLogger.DECORATION_LINE)

        self.create_weight(model, param_option, base_map_idx, vertex_maps, vertex_connecteds, duplicate_vertices, registed_bone_indexs, bone_horizonal_distances, bone_vertical_distances)

        return True

    def create_weight(self, model: PmxModel, param_option: dict, base_map_idx: int, vertex_maps: list, vertex_connecteds: dict, duplicate_vertices: dict, \
                      registed_bone_indexs: dict, bone_horizonal_distances: dict, bone_vertical_distances: dict):
        # ウェイト分布

        # 略称
        abb_name = param_option['abb_name']

        v_yidxs = list(reversed(list(registed_bone_indexs.keys())))
        for above_v_yidx, below_v_yidx in zip(v_yidxs[1:], v_yidxs[:-1]):
            above_v_xidxs = registed_bone_indexs[above_v_yidx]
            below_v_xidxs = registed_bone_indexs[below_v_yidx]
            # 繋がってる場合、最後に最初のボーンを追加する
            if vertex_connecteds[base_map_idx][above_v_yidx]:
                above_v_xidxs += [registed_bone_indexs[above_v_yidx][0]]
            if vertex_connecteds[base_map_idx][below_v_yidx]:
                below_v_xidxs += [registed_bone_indexs[below_v_yidx][0]]

            for xi, (prev_below_v_xidx, next_below_v_xidx) in enumerate(zip(below_v_xidxs[:-1], below_v_xidxs[1:])):
                prev_below_v_xno = prev_below_v_xidx + 1
                next_below_v_xno = next_below_v_xidx + 1
                below_v_yno = below_v_yidx + 1

                prev_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, prev_below_v_xno)]
                next_below_bone = model.bones[self.get_bone_name(abb_name, below_v_yno, next_below_v_xno)]
                prev_above_bone = model.bones[model.bone_indexes[prev_below_bone.parent_index]]
                next_above_bone = model.bones[model.bone_indexes[next_below_bone.parent_index]]

                _, prev_above_v_xidx = self.disassemble_bone_name(prev_above_bone.name)
                _, next_above_v_xidx = self.disassemble_bone_name(next_above_bone.name)

                if xi > 0 and (next_below_v_xidx == registed_bone_indexs[below_v_yidx][0] or next_above_v_xidx == registed_bone_indexs[above_v_yidx][0]):
                    # nextが最初のボーンである場合、最後まで
                    v_map = vertex_maps[base_map_idx][above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                    b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                    b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):]
                else:
                    v_map = vertex_maps[base_map_idx][above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                    b_h_distances = bone_horizonal_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                    b_v_distances = bone_vertical_distances[above_v_yidx:(below_v_yidx + 1), min(prev_below_v_xidx, prev_above_v_xidx):(max(next_below_v_xidx, next_above_v_xidx) + 1)]
                
                for vi, v_vertices in enumerate(v_map):
                    for vhi, vertex_idx in enumerate(v_vertices):
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

        # 基準頂点マップ以外の頂点が残っていたら、それも割り当てる
        for vertex_index in list(set(model.material_vertices[param_option['material_name']]) - set(vertex_maps[base_map_idx].tolist())):
            bone_names = []
            bone_distances = []
            for byidx in registed_bone_indexs.keys():
                v_yno = byidx + 1
                for bxidx in registed_bone_indexs[byidx]:
                    v_xno = bxidx + 1
                    bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                    bone_names.append(bone_name)
                    bone_distances.append(model.vertex_dict[vertex_index].position.distanceToPoint(model.bones[bone_name].position))

            for bone_name, b_distance in zip(bone_names, bone_distances):
                logger.debug(f"({bone_name}): {b_distance}")
                pass
            
    def create_bone(self, model: PmxModel, param_option: dict, vertex_map: np.ndarray, vertex_connected: list):
        # 中心ボーン生成

        # 略称
        abb_name = param_option['abb_name']
        # 材質名
        material_name = param_option['material_name']

        # 表示枠定義
        model.display_slots[material_name] = DisplaySlot(material_name, material_name, 0, 0)

        root_bone = Bone(f'{abb_name}中心', f'{abb_name}中心', model.bones[param_option['parent_bone_name']].position, \
                         model.bones[param_option['parent_bone_name']].index, 0, 0x0000 | 0x0002 | 0x0008 | 0x0010)
        root_bone.index = len(list(model.bones.keys()))

        # ボーン
        model.bones[root_bone.name] = root_bone
        model.bone_indexes[root_bone.index] = root_bone.name
        # 表示枠
        model.display_slots[material_name].references.append((0, model.bones[root_bone.name].index))

        # Map用INDEXリスト(根元はボーン追従にするため、常に一段だけ)
        v_yidxs = [0] + list(range(1, vertex_map.shape[0] - param_option["vertical_bone_density"] + 1, param_option["vertical_bone_density"]))
        if v_yidxs[-1] < vertex_map.shape[0] - 1:
            # 最下段は必ず登録
            v_yidxs = v_yidxs + [vertex_map.shape[0] - 1]
        
        all_v_xidxs = []
        for yi in range(vertex_map.shape[0]):
            v_xidxs = list(range(0, vertex_map.shape[1], param_option["horizonal_bone_density"]))
            if not vertex_connected[yi] and v_xidxs[-1] < vertex_map.shape[1] - 1:
                # 繋がってなくて、かつ端っこが登録されていない場合、登録
                v_xidxs = v_xidxs + [vertex_map.shape[1] - 1]
            all_v_xidxs.append(v_xidxs)

        tmp_all_bones = {}
        bone_horizonal_distances = np.zeros(vertex_map.shape)
        bone_vertical_distances = np.zeros(vertex_map.shape)

        for yi, v_yidx in enumerate(v_yidxs):
            for v_xidx in all_v_xidxs[yi]:
                if vertex_map[v_yidx, v_xidx] < 0:
                    # 存在しない頂点はスルー
                    continue

                v = model.vertex_dict[vertex_map[v_yidx, v_xidx]]
                v_xno = v_xidx + 1
                v_yno = v_yidx + 1

                # ボーン仮登録
                bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                bone = Bone(bone_name, bone_name, v.position, root_bone.index, 0, 0x0000 | 0x0002 | 0x0008 | 0x0010)
                tmp_all_bones[bone.name] = bone

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

        # 最下段の横幅平均値(段数単位)
        edge_size = np.mean(bone_horizonal_distances[-1, :]) * param_option["horizonal_bone_density"]

        registed_bone_indexs = {}

        for yi, v_yidx in enumerate(v_yidxs):
            prev_xidx = 0
            registed_bone_indexs[v_yidx] = []

            for v_xidx in all_v_xidxs[yi]:
                if v_xidx == 0 or (not vertex_connected[yi] and v_xidx == all_v_xidxs[yi][-1]) or \
                    not param_option['bone_thinning_out'] or (param_option['bone_thinning_out'] and \
                    np.sum(bone_horizonal_distances[v_yidx, prev_xidx:(v_xidx + 1)]) >= edge_size * 0.9):  # noqa
                    #  and v_xidxs[-1] - v_xidx >= param_option["horizonal_bone_density"] - 1
                    # 前ボーンとの間隔が最下段の横幅平均値より開いている場合、登録対象
                    v_xno = v_xidx + 1
                    v_yno = v_yidx + 1

                    # ボーン名
                    bone_name = self.get_bone_name(abb_name, v_yno, v_xno)
                    if bone_name not in tmp_all_bones:
                        continue

                    # ボーン本登録
                    bone = tmp_all_bones[bone_name]
                    bone.index = len(list(model.bones.keys()))

                    if yi > 0:
                        parent_v_xidx_diff = np.abs(np.array(registed_bone_indexs[v_yidxs[yi - 1]]) - v_xidx)

                        # 2段目以降は最も近い親段でv_xidxを探す
                        parent_v_xidx = registed_bone_indexs[v_yidxs[yi - 1]][(0 if vertex_connected[yi] and (v_xidxs[-1] + 1) - v_xidx < np.min(parent_v_xidx_diff) else np.argmin(parent_v_xidx_diff))]   # noqa

                        parent_bone = model.bones[self.get_bone_name(abb_name, v_yidxs[yi - 1] + 1, parent_v_xidx + 1)]
                        bone.parent_index = parent_bone.index

                        # 親ボーンの表示先も同時設定
                        parent_bone.tail_index = bone.index
                        parent_bone.flag |= 0x0001

                    model.bones[bone.name] = bone
                    model.bone_indexes[bone.index] = bone.name

                    registed_bone_indexs[v_yidx].append(v_xidx)

                    # 表示枠
                    model.display_slots[material_name].references.append((0, bone.index))

                    # 前ボーンとして設定
                    prev_xidx = v_xidx
        
        return root_bone, registed_bone_indexs, bone_horizonal_distances, bone_vertical_distances

    def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
        return f'{abb_name}-{(v_yno):03d}-{(v_xno):03d}'

    def disassemble_bone_name(self, bone_name: str):
        v_xidx = int(bone_name[-3:]) - 1
        v_yidx = int(bone_name[-7:-4]) - 1
        return v_yidx, v_xidx

    # 頂点を展開した図を作成
    def create_vertex_map(self, model: PmxModel, param_option: dict):
        logger.info(f"{param_option['material_name']}: 面の抽出")

        logger.info(f"{param_option['material_name']}: 重複頂点の抽出")

        # 位置ベースで重複頂点の抽出
        duplicate_vertices = {}
        for vertex_idx in model.material_vertices[param_option['material_name']]:
            # 重複頂点の抽出
            vertex = model.vertex_dict[vertex_idx]
            key = vertex.position.to_log()
            if key not in duplicate_vertices:
                duplicate_vertices[key] = []
            if vertex.index not in duplicate_vertices[key]:
                duplicate_vertices[key].append(vertex.index)
            # 一旦ルートボーンにウェイトを一括置換
            vertex.deform = Bdef1(model.bones[param_option['parent_bone_name']].index)

        # 面組み合わせの生成
        indices_by_vidx = {}
        indices_by_vpos = {}
        index_combs_by_vpos = {}
        duplicate_indices = {}
        below_iidx = None
        max_below_x = 99999999
        for index_idx in model.material_indices[param_option['material_name']]:
            # 頂点の組み合わせから面INDEXを引く
            indices_by_vidx[tuple(sorted(model.indices[index_idx]))] = index_idx
            # 重複辺（2点）の組み合わせ
            index_combs = list(itertools.combinations(model.indices[index_idx], 2))
            for (iv1, iv2) in index_combs:
                if abs((model.vertex_dict[iv1].position - model.vertex_dict[iv1].position).normalized().x()) < max_below_x:
                    below_iidx = index_idx
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

        logger.info(f"{param_option['material_name']}: 相対頂点マップの生成")

        # 頂点マップ生成(最初の頂点が(0, 0))
        vertex_axis_maps = []
        vertex_coordinate_maps = []
        registed_iidxs = []
        vertical_iidxs = []
        prev_index_cnt = 0

        while len(registed_iidxs) < len(model.material_indices[param_option['material_name']]):
            if not vertical_iidxs:
                # 切替時はとりあえず一面取り出して判定(二次元配列になる)
                if not registed_iidxs:
                    # 初回は出来るだけ真っ直ぐの辺がある面とする
                    index_idx = below_iidx
                else:
                    remaining_iidxs = list(set(model.material_indices[param_option['material_name']]) - set(registed_iidxs))
                    index_idx = remaining_iidxs[0]
                first_vertex_axis_map, first_vertex_coordinate_map = \
                    self.create_vertex_map_by_index(model, param_option, duplicate_indices, duplicate_vertices, {}, {}, index_idx)
                vertex_axis_maps.append(first_vertex_axis_map)
                vertex_coordinate_maps.append(first_vertex_coordinate_map)
                registed_iidxs.append(index_idx)
                vertical_iidxs.append(index_idx)
                
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

                # 縦辺がいる場合（まずは下方向）
                n = 0
                while n < 200:
                    n += 1
                    vertex_axis_maps[-1], vertex_coordinate_maps[-1], registed_iidxs, now_vertical_iidxs \
                        = self.fill_vertical_indices(model, param_option, duplicate_indices, duplicate_vertices, \
                                                     vertex_axis_maps[-1], vertex_coordinate_maps[-1], indices_by_vpos, indices_by_vidx, \
                                                     registed_iidxs, now_vertical_iidxs, 1)
                    total_vertical_iidxs.extend(now_vertical_iidxs)
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
                        if not now_vertical_iidxs:
                            break
                
                # 縦が終わった場合、横に移動する
                min_x, min_y, max_x, max_y = self.get_axis_range(model, vertex_coordinate_maps[-1], registed_iidxs)
                
                if not now_vertical_iidxs:
                    # 左方向
                    registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
                                                                                      duplicate_vertices, registed_iidxs, min_x, min_y, max_y, -1)
                    
                if not now_vertical_iidxs:
                    # 右方向
                    registed_iidxs, now_vertical_iidxs = self.fill_horizonal_now_idxs(model, param_option, vertex_axis_maps[-1], vertex_coordinate_maps[-1], duplicate_indices, \
                                                                                      duplicate_vertices, registed_iidxs, max_x, min_y, max_y, 1)
                
                vertical_iidxs = now_vertical_iidxs
                
            if not vertical_iidxs:
                remaining_iidxs = list(set(model.material_indices[param_option['material_name']]) - set(registed_iidxs))
                # 全頂点登録済みの面を潰していく
                for index_idx in remaining_iidxs:
                    iv0, iv1, iv2 = model.indices[index_idx]
                    if iv0 in vertex_axis_maps[-1] and iv1 in vertex_axis_maps[-1] and iv2 in vertex_axis_maps[-1]:
                        registed_iidxs.append(index_idx)

            if len(registed_iidxs) > 0 and len(registed_iidxs) // 200 > prev_index_cnt:
                logger.info(f"-- 面: {len(registed_iidxs)}個目:終了")
                prev_index_cnt = len(registed_iidxs) // 200
            
        logger.info(f"-- 面: {len(registed_iidxs)}個目:終了")

        logger.info(f"{param_option['material_name']}: 絶対頂点マップの生成")
        vertex_maps = []
        vertex_connecteds = []

        for midx, (vertex_axis_map, vertex_coordinate_map) in enumerate(zip(vertex_axis_maps, vertex_coordinate_maps)):
            logger.info(f"-- 絶対頂点マップ: {midx + 1}個目: ---------")

            # XYの最大と最小の抽出
            coordinate_x_key = list(sorted(vertex_coordinate_map.keys(), key=lambda x: x[0]))
            coordinate_y_key = list(sorted(vertex_coordinate_map.keys(), key=lambda x: x[1]))
            min_x = coordinate_x_key[0][0]
            min_y = coordinate_y_key[0][1]
            max_x = coordinate_x_key[-1][0]
            max_y = coordinate_y_key[-1][1]

            # 存在しない頂点INDEXで二次元配列初期化
            vertex_map = np.full((max_y - min_y + 1, max_x - min_x + 1), -1)
            vertex_display_map = np.full((max_y - min_y + 1, max_x - min_x + 1), 'None')
            vertex_connected = []

            for vmap in vertex_axis_map.values():
                vertex_map[vmap['y'] - min_y, vmap['x'] - min_x] = vmap['vidx']
                vertex_display_map[vmap['y'] - min_y, vmap['x'] - min_x] = ':'.join([str(v) for v in vertex_coordinate_map[(vmap['x'], vmap['y'])]])

            # 左端と右端で面が連続しているかチェック
            for yi in range(vertex_map.shape[0]):
                if (min(vertex_map[yi, 0], vertex_map[yi, -1]), max(vertex_map[yi, 0], vertex_map[yi, -1])) in duplicate_indices:
                    vertex_connected.append(True)
                else:
                    vertex_connected.append(False)

            vertex_maps.append(vertex_map)
            vertex_connecteds.append(vertex_connected)

            logger.info('\n'.join([', '.join(vertex_display_map[vx, :]) for vx in range(vertex_display_map.shape[0])]))
            logger.info(f"-- 絶対頂点マップ: {midx + 1}個目:終了 ---------")

        return vertex_maps, vertex_connecteds, duplicate_vertices
    
    def get_axis_range(self, model: PmxModel, vertex_coordinate_map: dict, registed_iidxs: list):
        xs = [k[0] for k in vertex_coordinate_map.keys()]
        ys = [k[1] for k in vertex_coordinate_map.keys()]

        # # それぞれの出現回数から大体全部埋まってるのを抽出。その中の最大と最小を選ぶ
        # xu, x_counts = np.unique(xs, return_counts=True)
        # full_x = [i for i, x in zip(xu, x_counts) if x >= max(x_counts) * 0.6]
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
                        # rkey1 = (min(iv1, remaining_vidx), max(iv1, remaining_vidx))
                        # rkey2 = (min(iv2, remaining_vidx), max(iv2, remaining_vidx))
                        # if rkey1 in duplicate_indices and len(duplicate_indices[rkey1]) > 1 and \
                        #    rkey2 in duplicate_indices and len(duplicate_indices[rkey2]) > 1:
                        #     # 両辺とも重複してる場合、開始の下辺であると見なす
                        #     ivy = max(vertex_axis_map[iv1]['y'], vertex_axis_map[iv2]['y'])
                        # else:
                        #     # 両辺のいずれかが重複してない場合、Y位置が近い方を採用する
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
        all_duplicate_indexs = []
        all_index_combs = []
        all_duplicate_dots = []
        all_vertical_below_v = []

        now_iidxs = []
        for vertical_vs in vertical_vs_list:
            # 該当縦辺の頂点(0が上(＋大きい))
            v0 = model.vertex_dict[vertical_vs[0]]
            v1 = model.vertex_dict[vertical_vs[1]]

            if offset > 0:
                # 下方向の走査
                vertical_vec = v0.position - v1.position
                vertical_below_v = v1
            else:
                # 上方向の走査
                vertical_vec = v1.position - v0.position
                vertical_below_v = v0

            if vertical_below_v.position.to_log() in indices_by_vpos:
                for duplicate_index_idx in indices_by_vpos[vertical_below_v.position.to_log()]:
                    if duplicate_index_idx in registed_iidxs + vertical_iidxs + now_iidxs:
                        # 既に登録済みの面である場合、スルー
                        continue

                    # 面の辺を抽出
                    vertical_in_vs, horizonal_in_vs, _ = self.judge_index_edge(model, vertex_axis_map, duplicate_index_idx)
                    if not horizonal_in_vs:
                        # まだ横が登録されていない場合、スルー
                        continue

                    #     if vertical_in_vs:
                    #         # 縦が求められているなら、残り一点は横方向である
                    #         pass
                    #         # all_duplicate_indexs.append(duplicate_index_idx)
                    #         # all_vertical_below_v.append(vertical_below_v)
                    #         # all_index_combs.append((vertical_below_v.index, list(set(vertical_in_vs) - {vertical_below_v.index})[0]))
                    #         # all_duplicate_dots.append(0)
                    #     else:
                    #         # 縦も横もまだならスルー
                    #         continue

                    if vertical_in_vs:
                        if ((offset > 0 and vertical_in_vs[0] in duplicate_vertices[vertical_below_v.position.to_log()]) \
                           or (offset < 0 and vertical_in_vs[1] in duplicate_vertices[vertical_below_v.position.to_log()])):
                            # 既に縦辺が求められていてそれに今回算出対象が含まれている場合
                            # 縦も横も求められているなら、該当面は必ず対象となる
                            all_duplicate_indexs.append(duplicate_index_idx)
                            all_vertical_below_v.append(vertical_below_v)
                            if offset > 0:
                                all_index_combs.append((vertical_in_vs[0], vertical_in_vs[1]))
                            else:
                                all_index_combs.append((vertical_in_vs[1], vertical_in_vs[0]))
                            all_duplicate_dots.append(1)
                        else:
                            # 既に縦辺が求められていてそれに今回算出対象が含まれていない場合、スルー
                            continue

                    # # 重複辺（2点）の組み合わせ(平行のどちらかと合致するはず)
                    # index_combs = list(itertools.product(list(set(model.indices[duplicate_index_idx]) - set(horizonal_in_vs)), horizonal_in_vs))
                    # 重複辺（2点）の組み合わせ
                    index_combs = list(itertools.combinations(model.indices[duplicate_index_idx], 2))
                    for (iv0_comb_idx, iv1_comb_idx) in index_combs:
                        all_duplicate_indexs.append(duplicate_index_idx)
                        all_vertical_below_v.append(vertical_below_v)

                        iv0 = None
                        iv1 = None
                        if iv0_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv1_comb_idx) not in all_index_combs:
                            iv0 = model.vertex_dict[iv0_comb_idx]
                            iv1 = model.vertex_dict[iv1_comb_idx]
                            all_index_combs.append((vertical_below_v.index, iv1_comb_idx))
                        elif iv1_comb_idx in duplicate_vertices[vertical_below_v.position.to_log()] and (vertical_below_v.index, iv0_comb_idx) not in all_index_combs:
                            iv0 = model.vertex_dict[iv1_comb_idx]
                            iv1 = model.vertex_dict[iv0_comb_idx]
                            all_index_combs.append((vertical_below_v.index, iv0_comb_idx))
                        else:
                            all_index_combs.append((-1, -1))

                        if iv0 and iv1:
                            if iv0.index in vertex_axis_map and (vertex_axis_map[iv0.index]['x'], vertex_axis_map[iv0.index]['y'] + offset) not in vertex_coordinate_map:
                                # v1から繋がる辺のベクトル
                                iv0 = model.vertex_dict[iv0.index]
                                iv1 = model.vertex_dict[iv1.index]
                                duplicate_vec = (iv0.position - iv1.position)
                                all_duplicate_dots.append(MVector3D.dotProduct(vertical_vec.normalized(), duplicate_vec.normalized()))
                            else:
                                all_duplicate_dots.append(0)
                        else:
                            all_duplicate_dots.append(0)

        if len(all_duplicate_dots) > 0 and np.max(all_duplicate_dots) >= param_option['similarity']:
            full_d = [i for i, (di, d) in enumerate(zip(all_duplicate_indexs, all_duplicate_dots)) if np.round(d, decimals=5) == np.max(np.round(all_duplicate_dots, decimals=5))]
            if full_d:
                vertical_idx = full_d[0]

                # 正方向に繋がる重複辺があり、かつそれが一定以上の場合、採用
                vertical_vidxs = all_index_combs[vertical_idx]
                duplicate_index_idx = all_duplicate_indexs[vertical_idx]
                vertical_below_v = all_vertical_below_v[vertical_idx]

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
            vertical_vs = (vertical_vs[0], vertical_vs[1]) if model.vertex_dict[vertical_vs[0]].position.y() >= model.vertex_dict[vertical_vs[1]].position.y() else (vertical_vs[1], vertical_vs[0])

        # 横の辺を抽出
        horizonal_vs = (v0.index, v1.index) if v0.index in vertex_axis_map and v1.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v1.index]['y'] \
            else (v0.index, v2.index) if v0.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v0.index]['y'] == vertex_axis_map[v2.index]['y'] \
            else (v1.index, v2.index) if v1.index in vertex_axis_map and v2.index in vertex_axis_map and vertex_axis_map[v1.index]['y'] == vertex_axis_map[v2.index]['y'] else None

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
                if abs(v0.position.y() - v1.position.y()) < abs(v2.position.y() - v1.position.y()):
                    # v1は横展開とみなす
                    vx = int(np.sign(MVector3D.crossProduct((v1.position - v0.position).normalized(), (v2.position - v0.position).normalized()).z()))
                    vy = 0
                else:
                    vy = -1 if v0.position.y() < v1.position.y() else 1
                    v1dot = MVector3D.dotProduct(MVector3D(0, -vy), (v1.position - v0.position).normalized())
                    v2dot = MVector3D.dotProduct(MVector3D(0, -vy), (v2.position - v0.position).normalized())
                    if v1dot > v2dot:
                        # v1の方がv0に近い場合、xは0
                        vx = 0
                    else:
                        # v2の方がv0に近い場合、斜めと見なして外積から時計回りで求める
                        vx = int(np.sign(MVector3D.crossProduct((v1.position - v0.position).normalized(), (v2.position - v0.position).normalized()).z()))

                vertex_axis_map[vidx] = {'vidx': vidx, 'x': vx, 'y': vy, 'position': model.vertex_dict[vidx].position, 'duplicate': duplicate_vertices[model.vertex_dict[vidx].position.to_log()]}
            vertex_coordinate_map[(vx, vy)] = vs_duplicated[v1.index]

            # 残り一点のマップ位置
            remaining_x, remaining_y = self.get_remaining_vertex_vec(v0.index, vertex_axis_map[v0.index]['x'], vertex_axis_map[v0.index]['y'], vertex_axis_map[v0.index]['position'], \
                                                                     v1.index, vertex_axis_map[v1.index]['x'], vertex_axis_map[v1.index]['y'], vertex_axis_map[v1.index]['position'], \
                                                                     v2, vertex_coordinate_map, duplicate_indices, duplicate_vertices)

            for vidx in vs_duplicated[v2.index]:
                vertex_axis_map[vidx] = {'vidx': vidx, 'x': remaining_x, 'y': remaining_y, 'position': model.vertex_dict[vidx].position}
            vertex_coordinate_map[(remaining_x, remaining_y)] = vs_duplicated[v2.index]
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
                logger.debug(f"{remaining_x}, {remaining_y}")
            elif (remaining_x, vv1_y) in vertex_coordinate_map:
                remaining_y = vv0_y
                logger.debug(f"{remaining_x}, {remaining_y}")
            else:
                # Yのみで距離判定
                remaining_y = vv0_y if abs(remaining_v.position.y() - vv0_vec.y()) < abs(remaining_v.position.y() - vv1_vec.y()) else vv1_y
                logger.debug(f"{remaining_x}, {remaining_y}")

        elif vv0_y == vv1_y:
            # 元が横方向に一致している場合

            if vv0_x < vv1_x:
                remaining_y = vv0_y + 1
            else:
                remaining_y = vv0_y - 1
            
            if (vv0_x, remaining_y) in vertex_coordinate_map:
                remaining_x = vv1_x
                logger.debug(f"{remaining_x}, {remaining_y}")
            elif (vv1_x, remaining_y) in vertex_coordinate_map:
                remaining_x = vv0_x
                logger.debug(f"{remaining_x}, {remaining_y}")
            else:
                # Y抜きで距離判定
                remaining_x = vv1_x if np.sign(MVector3D.crossProduct((vv1_vec - remaining_v.position).normalized(), (vv0_vec - remaining_v.position).normalized()).z()) else vv0_x
                # remaining_x = vv0_x if MVector3D(vv0_vec.x(), 0, vv0_vec.z()).distanceToPoint(MVector3D(remaining_v.position.x(), 0, remaining_v.position.z())) \
                #     < MVector3D(vv1_vec.x(), 0, vv1_vec.z()).distanceToPoint(MVector3D(remaining_v.position.x(), 0, remaining_v.position.z())) else vv1_x
                logger.debug(f"{remaining_x}, {remaining_y}")
        else:
            # 斜めが一致している場合
            if (vv0_x > vv1_x and vv0_y < vv1_y) or (vv0_x < vv1_x and vv0_y > vv1_y):
                # ／↑の場合、↓、↓／の場合、↑、／←の場合、→
                remaining_x = vv1_x
                remaining_y = vv0_y
                logger.debug(f"{remaining_x}, {remaining_y}")
            else:
                # ＼←の場合、→、／→の場合、←
                remaining_x = vv0_x
                remaining_y = vv1_y
                logger.debug(f"{remaining_x}, {remaining_y}")
        
        return remaining_x, remaining_y

