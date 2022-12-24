"""
あぴミク髪細分化
"""
from datetime import datetime
import sys
import pathlib
import os
import numpy as np
import random
from glob import glob

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
from service.PmxTailorExportService import read_vertices_from_file, VirtualVertex, calc_ratio


MLogger.initialize(level=MLogger.DEBUG, is_file=True)
logger = MLogger(__name__, level=MLogger.DEBUG)


def exec():
    model = PmxReader(
        "E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル1223/APmiku_nakedhair_IKx_onlyee_start.pmx",
        is_check=False,
        is_sizing=False,
    ).read_data()
    print(model.name)

    for csv_file_path in glob("E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル1223/*.csv"):
        abb_name = os.path.basename(csv_file_path.replace(".csv", ""))
        target_vertices = read_vertices_from_file(csv_file_path, model, "髪")

        params = {}
        # 簡易版オプションデータ -------------
        params["material_name"] = "髪"
        params["back_material_name"] = ""
        params["back_extend_material_names"] = []
        params["edge_material_name"] = ""
        params["edge_extend_material_names"] = []
        params["parent_bone_name"] = "頭"
        params["abb_name"] = abb_name
        params["direction"] = "下"
        params["exist_physics_clear"] = "すべて表面"
        params["special_shape"] = "なし"
        params["vertices_csv"] = ""
        params["vertices_edge_csv"] = ""
        params["vertices_back_csv"] = ""
        params["top_vertices_csv"] = ""
        params["mass"] = 0
        params["air_resistance"] = 0
        params["shape_maintenance"] = 0
        params["vertical_bone_density"] = 4

        vertex_map, virtual_vertices, threshold, max_pos, remaining_vertices = create_vertex_map(
            model, params, params["material_name"], target_vertices
        )

        (
            root_bone,
            virtual_vertices,
            registered_bones,
            bone_vertical_distances,
            bone_horizonal_distances,
            bone_connected,
            root_yidx,
        ) = create_bone(model, params, params["material_name"], virtual_vertices, vertex_map, threshold, max_pos)

        remaining_vertices = create_weight(
            model,
            virtual_vertices,
            vertex_map,
            registered_bones,
            bone_vertical_distances,
            root_yidx,
            remaining_vertices,
        )

        create_remaining_weight(model, virtual_vertices, vertex_map, remaining_vertices)

    PmxWriter().write(
        model,
        f"E:/MMD/MikuMikuDance_v926x64/Work/202101_vroid/_報告/ラワイル1223/APmiku_nakedhair_IKx_onlyee_{params['vertical_bone_density']}_{datetime.now():%Y%m%d_%H%M%S}.pmx",
    )


def get_rigidbody(model: PmxModel, bone_name: str):
    if bone_name not in model.bones:
        return None

    for rigidbody in model.rigidbodies.values():
        if rigidbody.bone_index == model.bones[bone_name].index:
            return rigidbody

    return None


def create_remaining_weight(
    model: PmxModel,
    virtual_vertices: dict,
    vertex_map: np.ndarray,
    remaining_vertices: dict,
):
    # ウェイト塗り終わった実頂点リスト
    weighted_vidxs = []

    # 塗り終わった頂点リスト
    weighted_vkeys = list(set(list(virtual_vertices.keys())) - set(list(remaining_vertices.keys())))

    weighted_poses = {}
    for vkey in weighted_vkeys:
        vv = virtual_vertices[vkey]
        if vv.vidxs():
            weighted_poses[vkey] = vv.position().data()
            weighted_vidxs.extend(vv.vidxs())

    # 登録済みのボーンの位置リスト
    bone_poses = {}
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

            # ボーンが登録されてない箇所かつまだウェイトが塗られてないのは残頂点に入れる
            if v_key not in remaining_vertices and not vv.deform:
                remaining_vertices[v_key] = vv

    remain_cnt = len(remaining_vertices) * 2
    while remaining_vertices.items() and remain_cnt > 0:
        remain_cnt -= 1
        # ランダムで選ぶ
        vkey = list(remaining_vertices.keys())[random.randrange(len(remaining_vertices))]
        vv = remaining_vertices[vkey]

        if not vv.vidxs():
            # vidx がないのはそもそも対象外
            del remaining_vertices[vkey]
            continue

        # ウェイト済み頂点のうち、最も近いのを抽出
        weighted_diff_distances = np.linalg.norm(
            np.array(list(weighted_poses.values())) - vv.position().data(), ord=2, axis=1
        )

        nearest_vkey = list(weighted_poses.keys())[np.argmin(weighted_diff_distances)]
        nearest_vv = virtual_vertices[nearest_vkey]
        nearest_deform = nearest_vv.deform
        # nearest_vv.connected_vvs.extend(vv.vidxs())

        if not nearest_vv.deform:
            # デフォームの対象がない場合、他で埋まる可能性があるので、一旦据え置き
            continue

        if type(nearest_deform) is Bdef1:
            logger.debug(
                f"remaining1 nearest_vv: {nearest_vv.vidxs()}, weight_names: [{model.bone_indexes[nearest_deform.index0]}], total_weights: [1]"
            )

            for rv in vv.real_vertices:
                weighted_vidxs.append(rv.index)
                rv.deform = Bdef1(nearest_deform.index0)
                vv.deform = Bdef1(nearest_deform.index0)

            del remaining_vertices[vkey]
        elif type(nearest_deform) is Bdef2:
            vv.deform = nearest_deform.copy()
            # weight_bone1 = model.bones[model.bone_indexes[nearest_deform.index0]]
            # weight_bone2 = model.bones[model.bone_indexes[nearest_deform.index1]]

            # bone1_distance = vv.position().distanceToPoint(weight_bone1.position)
            # bone2_distance = vv.position().distanceToPoint(weight_bone2.position) if nearest_deform.weight0 < 1 else 0
            # weight_names = np.array([weight_bone1.name, weight_bone2.name])
            # if bone1_distance + bone2_distance != 0:
            #     total_weights = np.array(
            #         [
            #             bone1_distance / (bone1_distance + bone2_distance),
            #             bone2_distance / (bone1_distance + bone2_distance),
            #         ]
            #     )
            # else:
            #     total_weights = np.array([1, 0])
            #     logger.warning("残ウェイト計算で意図せぬ値が入ったため、BDEF1を設定します。: 対象頂点[%s]", vv.vidxs())
            # weights = total_weights / total_weights.sum(axis=0, keepdims=1)
            # weight_idxs = np.argsort(weights)

            # logger.debug(
            #     f"remaining2 nearest_vv: {vv.vidxs()}, weight_names: [{weight_names}], total_weights: [{total_weights}]"
            # )

            # if np.count_nonzero(weights) == 1:
            #     vv.deform = Bdef1(model.bones[weight_names[weight_idxs[-1]]].index)
            # elif np.count_nonzero(weights) == 2:
            #     vv.deform = Bdef2(
            #         model.bones[weight_names[weight_idxs[-1]]].index,
            #         model.bones[weight_names[weight_idxs[-2]]].index,
            #         weights[weight_idxs[-1]],
            #     )
            # else:
            #     vv.deform = Bdef4(
            #         model.bones[weight_names[weight_idxs[-1]]].index,
            #         model.bones[weight_names[weight_idxs[-2]]].index,
            #         model.bones[weight_names[weight_idxs[-3]]].index,
            #         model.bones[weight_names[weight_idxs[-4]]].index,
            #         weights[weight_idxs[-1]],
            #         weights[weight_idxs[-2]],
            #         weights[weight_idxs[-3]],
            #         weights[weight_idxs[-4]],
            #     )

            for rv in vv.real_vertices:
                weighted_vidxs.append(rv.index)
                rv.deform = vv.deform

            del remaining_vertices[vkey]

    return weighted_vidxs


def create_weight(
    model: PmxModel,
    virtual_vertices: dict,
    vertex_map: np.ndarray,
    registered_bones: dict,
    bone_vertical_distances: dict,
    root_yidx: int,
    remaining_vertices: dict,
):

    for v_xidx in range(vertex_map.shape[1]):
        for v_yidx in range(vertex_map.shape[0]):
            if np.isnan(vertex_map[v_yidx, v_xidx]).any():
                continue

            vkey = tuple(vertex_map[v_yidx, v_xidx])
            vv = virtual_vertices[vkey]

            if v_yidx < root_yidx:
                # 根元より上の場合、そのまま根元ウェイト
                vv.deform = Bdef1(registered_bones[root_yidx].index)

                for rv in vv.real_vertices:
                    rv.deform = vv.deform

                # 登録対象の場合、残対象から削除
                if vkey in remaining_vertices:
                    del remaining_vertices[vkey]

            elif registered_bones[v_yidx]:
                # 頂点位置にボーンが登録されている場合、BDEF1登録対象
                vv.deform = Bdef1(registered_bones[v_yidx].index)

                for rv in vv.real_vertices:
                    rv.deform = vv.deform

                # 登録対象の場合、残対象から削除
                if vkey in remaining_vertices:
                    del remaining_vertices[vkey]
            else:
                above_yidx = root_yidx
                if v_yidx > root_yidx:
                    above_yidx = np.max(np.where(registered_bones[:v_yidx])[0])
                if len(np.where(registered_bones[v_yidx + 1 :])[0]):
                    below_yidx = v_yidx + 1 + np.min(np.where(registered_bones[v_yidx + 1 :])[0])
                else:
                    below_yidx = np.max(np.where(registered_bones)[0])

                if above_yidx == below_yidx:
                    # 末端はBDEF1でウェイトを塗っておく
                    vv.deform = Bdef1(registered_bones[below_yidx].index)

                    for rv in vv.real_vertices:
                        rv.deform = vv.deform

                    # 登録対象の場合、残対象から削除
                    if vkey in remaining_vertices:
                        del remaining_vertices[vkey]

                    continue

                # 登録外のはBDEF2
                above_weight = np.nan_to_num(
                    (
                        np.sum(bone_vertical_distances[v_yidx:below_yidx, v_xidx])
                        / np.sum(bone_vertical_distances[above_yidx:below_yidx, v_xidx])
                    )
                )

                below_weight = np.nan_to_num(
                    (
                        np.sum(bone_vertical_distances[above_yidx:v_yidx, v_xidx])
                        / np.sum(bone_vertical_distances[above_yidx:below_yidx, v_xidx])
                    )
                )

                # ほぼ0のものは0に置換（円周用）
                total_weights = np.array([above_weight, below_weight])
                total_weights[np.isclose(total_weights, 0, equal_nan=True)] = 0

                if np.count_nonzero(total_weights):
                    deform_weights = total_weights / total_weights.sum(axis=0, keepdims=1)

                    vv.deform = Bdef2(
                        registered_bones[above_yidx].index,
                        registered_bones[below_yidx].index,
                        deform_weights[0],
                    )

                    for rv in vv.real_vertices:
                        rv.deform = vv.deform

                    # 登録対象の場合、残対象から削除
                    if vkey in remaining_vertices:
                        del remaining_vertices[vkey]
                else:
                    pass

    return remaining_vertices


def create_bone(
    model: PmxModel,
    param_option: dict,
    material_name: str,
    virtual_vertices: dict,
    vertex_map: dict,
    threshold: float,
    max_pos: MVector3D,
):
    logger.info("【%s:%s】ボーン生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

    # 中心ボーン生成

    # 略称
    abb_name = param_option["abb_name"]
    # 親ボーン
    parent_bone = model.bones[param_option["parent_bone_name"]]

    # 中心ボーン
    display_name, root_bone = create_root_bone(model, param_option, material_name, max_pos.copy())

    logger.info("【%s】頂点距離の算出", material_name)

    bone_horizonal_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))
    bone_vertical_distances = np.zeros((vertex_map.shape[0], vertex_map.shape[1]))
    bone_connected = np.zeros((vertex_map.shape[0], vertex_map.shape[1]), dtype=np.int)

    # 各頂点の距離（円周っぽい可能性があるため、頂点一個ずつで測る）
    for v_yidx in range(vertex_map.shape[0]):
        v_xidx = -1
        for v_xidx in range(0, vertex_map.shape[1] - 1):
            if not np.isnan(vertex_map[v_yidx, v_xidx]).any() and not np.isnan(vertex_map[v_yidx, v_xidx + 1]).any():
                now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx + 1])].position()
                bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)

                if tuple(vertex_map[v_yidx, v_xidx]) in virtual_vertices[
                    tuple(vertex_map[v_yidx, v_xidx + 1])
                ].connected_vvs or tuple(vertex_map[v_yidx, v_xidx]) == tuple(vertex_map[v_yidx, v_xidx + 1]):
                    # 前の仮想頂点と同じか繋がっている場合、True
                    bone_connected[v_yidx, v_xidx] = True

            if (
                v_yidx < vertex_map.shape[0] - 1
                and not np.isnan(vertex_map[v_yidx, v_xidx]).any()
                and not np.isnan(vertex_map[v_yidx + 1, v_xidx]).any()
            ):
                now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx + 1, v_xidx])].position()
                bone_vertical_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)

        v_xidx += 1
        if (
            not np.isnan(vertex_map[v_yidx, v_xidx]).any()
            and not np.isnan(vertex_map[v_yidx, 0]).any()
            and vertex_map.shape[1] > 2
        ):
            # 輪を描いたのも入れとく(ウェイト対象取得の時に範囲指定入るからここでは強制)
            if tuple(vertex_map[v_yidx, 0]) in virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].connected_vvs:
                # 横の仮想頂点と繋がっている場合、Trueで有効な距離を入れておく
                bone_connected[v_yidx, v_xidx] = True

                now_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, v_xidx])].position()
                next_v_vec = virtual_vertices[tuple(vertex_map[v_yidx, 0])].position()
                bone_horizonal_distances[v_yidx, v_xidx] = now_v_vec.distanceToPoint(next_v_vec)
            else:
                # とりあえずINT最大値を入れておく
                bone_horizonal_distances[v_yidx, v_xidx] = np.iinfo(np.int).max

    logger.debug("bone_horizonal_distances ------------")
    logger.debug(bone_horizonal_distances.tolist())
    logger.debug("bone_vertical_distances ------------")
    logger.debug(bone_vertical_distances.tolist())
    logger.debug("bone_connected ------------")
    logger.debug(bone_connected.tolist())

    # ボーン登録有無
    regist_bones = np.zeros(vertex_map.shape[0], dtype=np.int)

    root_yidx = np.where(
        np.array([not np.isnan(vertex_map[yidx, :]).any() for yidx in range(vertex_map.shape[0])]) == True
    )[0][0]

    # 間隔が頂点タイプの場合、規則的に間を空ける
    for v_yidx in list(range(root_yidx, vertex_map.shape[0], param_option["vertical_bone_density"])) + [
        vertex_map.shape[0] - 1,
    ]:
        regist_bones[v_yidx] = True

    registered_bones = np.full(vertex_map.shape[0], fill_value=None)
    prev_bone = None
    for v_yidx in range(vertex_map.shape[0]):
        if not regist_bones[v_yidx]:
            # 登録対象ではない場合、スルー
            continue

        # 親は既にモデルに登録済みのものを選ぶ
        parent_bone = root_bone if v_yidx == root_yidx else prev_bone

        bone_name = f"{abb_name}-{(v_yidx + 1):03d}"

        bone_pos = MVector3D(
            np.mean(vertex_map[v_yidx, np.unique(np.where(np.isnan(vertex_map[v_yidx, :]) == False)[0])], axis=0)
            * threshold
        )
        bone = Bone(bone_name, bone_name, bone_pos, parent_bone.index, 0, 0x0000 | 0x0002 | 0x0800)

        bone.parent_index = parent_bone.index
        bone.local_x_vector = (bone.position - parent_bone.position).normalized()
        bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, MVector3D(1, 0, 0))

        # 登録対象の場合のみボーン保持
        bone.index = len(model.bones)
        parent_bone.tail_index = bone.index
        if v_yidx < vertex_map.shape[0] - 1:
            bone.flag |= 0x0008 | 0x0010 | 0x0001
            registered_bones[v_yidx] = bone

        model.display_slots[display_name].references.append((0, bone.index))

        logger.debug(f"tmp_all_bones: {bone.name}, pos: {bone.position.to_log()}")

        model.bones[bone_name] = bone
        model.bone_indexes[bone.index] = bone_name

        prev_bone = bone

    return (
        root_bone,
        virtual_vertices,
        registered_bones,
        bone_vertical_distances,
        bone_horizonal_distances,
        bone_connected,
        root_yidx,
    )


def get_bone_name(self, abb_name: str, v_yno: int, v_xno: int):
    return f"{abb_name}-{(v_yno):03d}-{(v_xno):03d}"


def create_root_bone(model: PmxModel, param_option: dict, material_name: str, root_pos: MVector3D):
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
        root_pos,
        parent_bone.index,
        0,
        0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010,
    )

    root_bone.index = len(model.bones)
    model.bones[root_bone.name] = root_bone
    model.bone_indexes[root_bone.index] = root_bone.name

    # 表示枠
    model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)
    model.display_slots[display_name].references.append((0, model.bones[root_bone.name].index))

    return display_name, root_bone


def create_ring(virtual_vertices: dict, vkey: tuple, rings: list):
    rings.append(vkey)

    vv: VirtualVertex = virtual_vertices[vkey]
    connected_vecs = {}
    for connected_vkey in vv.connected_vvs:
        if connected_vkey not in rings and connected_vkey in virtual_vertices:
            connected_vec = (vv.position() - virtual_vertices[connected_vkey].position()).normalized()
            connected_vecs[connected_vkey] = connected_vec.data()
    next_vkey = list(connected_vecs.keys())[np.argmin(np.abs(list(connected_vecs.values())), axis=0)[1]]
    if np.min(np.abs(list(connected_vecs.values())), axis=0)[1] > 0.3:
        return rings

    return create_ring(virtual_vertices, next_vkey, rings)


def create_vertex_map(
    model: PmxModel,
    param_option: dict,
    material_name: str,
    target_vertices: list,
):
    logger.info("【%s:%s】頂点マップ生成", material_name, param_option["abb_name"], decoration=MLogger.DECORATION_LINE)

    # 残頂点リスト
    remaining_vertices = {}

    parent_bone = model.bones[param_option["parent_bone_name"]]

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
        .data()[np.where(np.abs([0, 1, 0]))]
    )[0]
    logger.info("%s: 材質頂点の傾き算出: %s", material_name, round(material_direction, 5))

    # 頂点間の距離
    # https://blog.shikoan.com/distance-without-for-loop/
    all_vertex_diffs = np.expand_dims(np.array(list(vertex_positions.values())), axis=1) - np.expand_dims(
        np.array(list(vertex_positions.values())), axis=0
    )
    all_vertex_distances = np.sqrt(np.sum(all_vertex_diffs**2, axis=-1))
    # 頂点同士の距離から閾値生成
    threshold = np.min(all_vertex_distances[all_vertex_distances > 0]) * 0.8

    logger.info("%s: 材質頂点の閾値算出: %s", material_name, round(threshold, 5))

    logger.info("%s: 仮想頂点リストの生成", material_name)

    virtual_vertices = {}
    all_poses = {}
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

        for vv0_key, vv1_key, vv2_key, vv0 in [
            (v0_key, v1_key, v2_key, v0),
            (v1_key, v2_key, v0_key, v1),
            (v2_key, v0_key, v1_key, v2),
        ]:
            # 仮想頂点登録（該当頂点対象）
            if vv0_key not in virtual_vertices:
                virtual_vertices[vv0_key] = VirtualVertex(vv0_key)
            all_poses[vv0_key] = vv0.position.data()
            virtual_vertices[vv0_key].append([vv0], [vv1_key, vv2_key], [index_idx])

            # 残頂点リストにまずは登録
            if vv0_key not in remaining_vertices:
                remaining_vertices[vv0_key] = virtual_vertices[vv0_key]

        logger.debug(
            f"☆表 index[{index_idx}], v0[{v0.index}:{v0_key}], v1[{v1.index}:{v1_key}], v2[{v2.index}:{v2_key}], pa[{parent_bone.position.to_log()}]"
        )

        n += 1

        if n > 0 and n % 500 == 0:
            logger.info("-- メッシュ確認: %s個目:終了", n)

    max_pos = np.max(list(all_poses.values()), axis=0)
    max_key = MVector3D(max_pos).to_key(threshold)

    min_pos = np.min(list(all_poses.values()), axis=0)
    min_key = MVector3D(min_pos).to_key(threshold)

    median_pos = np.median(list(all_poses.values()), axis=0)
    median_key = MVector3D(median_pos).to_key(threshold)

    median_vkey = min_vkey = max_vkey = None
    for x, y, z in all_poses.keys():
        if y == median_key[1] and not median_vkey:
            median_vkey = (x, y, z)
        if y == min_key[1] and not min_vkey:
            min_vkey = (x, y, z)
        if y == max_key[1] and not max_vkey:
            max_vkey = (x, y, z)

    min_pos = virtual_vertices[min_vkey].position()
    max_pos = virtual_vertices[max_vkey].position()

    # リング取得
    median_rings = create_ring(virtual_vertices, median_vkey, [])

    all_upper_vkeys = []
    all_lower_vkeys = []
    for mi, target_vkey in enumerate(median_rings):
        # 根元に向けてのライン
        upper_vkeys, upper_vscores = create_vertex_line_map(
            target_vkey,
            max_pos,
            target_vkey,
            virtual_vertices,
            [target_vkey],
            [],
            [],
        )

        # 髪上部の進行方向に合わせて下を検出する
        # ボーン進行方向(x)
        top_x_pos = (
            virtual_vertices[upper_vkeys[-2]].position() - virtual_vertices[upper_vkeys[-1]].position()
        ).normalized()
        # ボーン進行方向に対しての縦軸(y)
        top_y_pos = MVector3D(1, 0, 0)
        # ボーン進行方向に対しての横軸(z)
        top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
        top_qq = MQuaternion.fromDirection(top_z_pos, top_y_pos)

        mat = MMatrix4x4()
        mat.setToIdentity()
        mat.translate(virtual_vertices[target_vkey].position())
        mat.rotate(top_qq)

        below_pos = mat * MVector3D(-3, 0, 0)

        lower_vkeys, lower_vscores = create_vertex_line_map(
            target_vkey,
            below_pos,
            target_vkey,
            virtual_vertices,
            [target_vkey],
            [],
            [],
        )

        all_upper_vkeys.append(upper_vkeys)
        all_lower_vkeys.append(lower_vkeys)

    # XYの最大と最小の抽出
    xu = len(all_upper_vkeys)
    upper_y = np.max([len(vk) for i, vk in enumerate(all_upper_vkeys)])
    lower_y = np.max([len(vk) for i, vk in enumerate(all_lower_vkeys)])

    # 存在しない頂点INDEXで二次元配列初期化
    vertex_map = np.full((upper_y + lower_y, xu, 3), (np.nan, np.nan, np.nan))

    # 根元から順にマップに埋めていく
    for x, vkeys in enumerate(all_upper_vkeys):
        for y, vkey in enumerate(reversed(vkeys)):
            vv = virtual_vertices[vkey]
            if not vv.vidxs():
                continue

            logger.debug(f"x: {x}, y: {y}, vv: {vkey}, vidxs: {vv.vidxs()}")

            vertex_map[upper_y - y, x] = vkey

    for x, vkeys in enumerate(all_lower_vkeys):
        for y, vkey in enumerate(reversed(vkeys)):
            vv = virtual_vertices[vkey]
            if not vv.vidxs():
                continue

            logger.debug(f"x: {x}, y: {y}, vv: {vkey}, vidxs: {vv.vidxs()}")

            vertex_map[upper_y + y, x] = vkey

    return vertex_map, virtual_vertices, threshold, max_pos, remaining_vertices


def create_vertex_line_map(
    bottom_key: tuple,
    top_pos: MVector3D,
    from_key: tuple,
    virtual_vertices: dict,
    vkeys: list,
    vscores: list,
    registed_vkeys: list,
    loop=0,
):

    if loop > 500:
        return None, None

    from_vv = virtual_vertices[from_key]
    from_pos = from_vv.position()

    bottom_vv = virtual_vertices[bottom_key]
    bottom_pos = bottom_vv.position()

    local_next_base_pos = MVector3D(1, 0, 0)

    # ボーン進行方向(x)
    top_x_pos = (top_pos - bottom_pos).normalized()
    # ボーン進行方向に対しての縦軸(y)
    top_y_pos = MVector3D(1, 0, 0)
    # ボーン進行方向に対しての横軸(z)
    top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
    top_qq = MQuaternion.fromDirection(top_z_pos, top_y_pos)
    logger.debug(
        f" - top({bottom_vv.vidxs()}): x[{top_x_pos.to_log()}], y[{top_y_pos.to_log()}], z[{top_z_pos.to_log()}]"
    )

    scores = []
    prev_dots = []
    for n, to_key in enumerate(from_vv.connected_vvs):
        if to_key not in virtual_vertices:
            scores.append(0)
            prev_dots.append(0)
            logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_key}], 対象外")
            continue

        if to_key in registed_vkeys:
            scores.append(0)
            prev_dots.append(0)
            logger.debug(f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_key}], 登録済み")
            continue

        to_vv = virtual_vertices[to_key]
        to_pos = to_vv.position()

        direction_dot = MVector3D.dotProduct(top_x_pos, (to_pos - from_pos).normalized())

        if to_key in vkeys:
            # 到達済みのベクトルには行かせない
            scores.append(0)
            prev_dots.append(0)
            logger.debug(
                f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], 到達済み"
            )
            continue

        if direction_dot < 0.3:
            # 反対方向のベクトルには行かせない
            scores.append(0)
            prev_dots.append(0)
            logger.debug(
                f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], 反対方向"
            )
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

        # 前頂点との内積差を考慮する場合
        prev_dot = (
            MVector3D.dotProduct(
                (virtual_vertices[vkeys[0]].position() - virtual_vertices[vkeys[1]].position()).normalized(),
                (to_pos - virtual_vertices[vkeys[0]].position()).normalized(),
            )
            if len(vkeys) > 1
            else 1
        )

        if prev_dot < 0.5:
            # ズレた方向には行かせない
            scores.append(0)
            prev_dots.append(0)
            logger.debug(
                f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], yaw_score: {round(yaw_score, 5)}, pitch_score: {round(pitch_score, 5)}, roll_score: {round(roll_score, 5)}, 前ズレ方向"
            )
            continue

        score = ((yaw_score * 20) + pitch_score + (roll_score * 2)) * (prev_dot**3)

        # scores.append(score * (2 if to_key in top_keys else 1))
        scores.append(score)
        prev_dots.append(prev_dot * direction_dot)

        logger.debug(
            f" - get_vertical_key({n}): from[{from_vv.vidxs()}], to[{to_vv.vidxs()}], direction_dot[{direction_dot}], prev_dot[{prev_dot}], local_next_vpos[{local_next_vpos.to_log()}], score: [{score}], yaw_score: {round(yaw_score, 5)}, pitch_score: {round(pitch_score, 5)}, roll_score: {round(roll_score, 5)}"
        )

    if np.count_nonzero(scores) == 0:
        # スコアが付けられなくなったら終了
        return vkeys, vscores

    # 最もスコアの高いINDEXを採用
    nearest_idx = np.argmax(scores)
    vertical_key = from_vv.connected_vvs[nearest_idx]

    # 前の辺との内積差を考慮する
    prev_diff_dot = (
        MVector3D.dotProduct(
            (virtual_vertices[vkeys[0]].position() - virtual_vertices[vkeys[1]].position()).normalized(),
            (virtual_vertices[vertical_key].position() - virtual_vertices[vkeys[0]].position()).normalized(),
        )
        if len(vkeys) > 1
        else 1
    )

    logger.debug(
        f"direction: from: [{virtual_vertices[from_key].vidxs()}], to: [{virtual_vertices[vertical_key].vidxs()}], prev_diff_dot[{round(prev_diff_dot, 4)}]"
    )

    vkeys.insert(0, vertical_key)
    vscores.insert(0, np.max(scores) * prev_diff_dot)

    # 髪上部の進行方向に合わせて進行方向を検出する
    # ボーン進行方向(x)
    top_x_pos = (virtual_vertices[vertical_key].position() - virtual_vertices[from_key].position()).normalized()
    # ボーン進行方向に対しての縦軸(y)
    top_y_pos = MVector3D(1, 0, 0)
    # ボーン進行方向に対しての横軸(z)
    top_z_pos = MVector3D.crossProduct(top_x_pos, top_y_pos)
    top_qq = MQuaternion.fromDirection(top_z_pos, top_y_pos)

    mat = MMatrix4x4()
    mat.setToIdentity()
    mat.translate(virtual_vertices[from_key].position())
    mat.rotate(top_qq)

    top_pos = mat * MVector3D(3, 0, 0)

    return create_vertex_line_map(
        bottom_key,
        top_pos,
        vertical_key,
        virtual_vertices,
        vkeys,
        vscores,
        registed_vkeys,
        loop + 1,
    )


def get_edge_lines(edge_line_pairs: dict, start_vkey: tuple, edge_lines: list, edge_vkeys: list, loop=0):
    if len(edge_vkeys) >= len(edge_line_pairs.keys()) or loop > 500:
        return start_vkey, edge_lines, edge_vkeys

    if not start_vkey:
        # 下: Y(降順) - X(中央揃え) - Z(降順)
        sorted_edge_line_pairs = sorted(
            list(set(edge_line_pairs.keys()) - set(edge_vkeys)), key=lambda x: (-x[1], abs(x[0]), -x[2])
        )
        start_vkey = sorted_edge_line_pairs[0]
        edge_lines.append([start_vkey])
        edge_vkeys.append(start_vkey)

    # 下: X(中央揃え) - Z(降順) - Y(降順)
    sorted_edge_line_pairs = sorted(edge_line_pairs[start_vkey], key=lambda x: (abs(x[0]), -x[2], -x[1]))

    for next_vkey in sorted_edge_line_pairs:
        if next_vkey not in edge_vkeys:
            edge_lines[-1].append(next_vkey)
            edge_vkeys.append(next_vkey)
            start_vkey, edge_lines, edge_vkeys = get_edge_lines(
                edge_line_pairs, next_vkey, edge_lines, edge_vkeys, loop + 1
            )

    return None, edge_lines, edge_vkeys


if __name__ == "__main__":
    exec()
