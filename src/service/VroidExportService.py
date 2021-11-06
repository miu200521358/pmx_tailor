# -*- coding: utf-8 -*-
#
import logging
import traceback
from PIL import Image, ImageChops
import struct
import os
import json
from pathlib import Path
import shutil
import numpy as np
import re
import math
import urllib.parse

from module.MOptions import MExportOptions
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, RigidBodyParam, IkLink, Ik # noqa
from mmd.PmxData import Bdef1, Bdef2, Bdef4, VertexMorphOffset, GroupMorphData # noqa
from mmd.PmxWriter import PmxWriter
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException

logger = MLogger(__name__, level=1)

MIME_TYPE = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/ktx': 'ktx',
    'image/ktx2': 'ktx2',
    'image/webp': 'webp',
    'image/vnd-ms.dds': 'dds',
    'audio/wav': 'wav'
}

# MMDにおける1cm＝0.125(ミクセル)、1m＝12.5
MIKU_METER = 12.5


class VroidExportService():
    def __init__(self, options: MExportOptions):
        self.options = options
        self.offset = 0
        self.buffer = None

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = "Vroid2Pmx処理実行\n------------------------\nexeバージョン: {version_name}\n".format(version_name=self.options.version_name) \

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

            model = self.vroid2pmx()

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            os.makedirs(os.path.dirname(self.options.output_path), exist_ok=True)
            PmxWriter().write(model, self.options.output_path)

            logger.info("出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title="成功")

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical("Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
        finally:
            logging.shutdown()

    def vroid2pmx(self):
        try:
            model = self.convert_texture(PmxModel())
            if not model:
                return False

            model, bone_dict, node_name_dict = self.convert_bone(model)
            if not model:
                return False

            # model = self.convert_vertex(model)
            # if not model:
            #     return False

            

            return model
        except MKilledException as ke:
            # 終了命令
            raise ke
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
            return se
        except Exception as e:
            import traceback
            logger.critical("Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
            raise e

    def convert_vertex(self, model: PmxModel):
        if 'meshes' not in model.json_data:
            logger.error("変換可能なメッシュ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None

        vertex_accessors = {}
        vertex_idx = 0

        for midx, mesh in enumerate(model.json_data["meshes"]):
            if "primitives" not in mesh:
                continue

            for pidx, primitive in enumerate(mesh["primitives"]):
                if "attributes" not in primitive:
                    continue

                # 頂点データ
                if primitive["attributes"]["POSITION"] not in vertex_accessors:
                    # 位置データ
                    positions = self.read_from_accessor(model, primitive["attributes"]["POSITION"])

                    # 法線データ
                    normals = self.read_from_accessor(model, primitive["attributes"]["NORMAL"])

                    # UVデータ
                    uvs = self.read_from_accessor(model, primitive["attributes"]["TEXCOORD_0"])

                    # ジョイントデータ(MMDのジョイントとは異なる)
                    if "JOINTS_0" in primitive["attributes"]:
                        joints = self.read_from_accessor(model, primitive["attributes"]["JOINTS_0"])
                    else:
                        joints = [MVector4D() for _ in range(len(positions))]
                    
                    # ウェイトデータ
                    if "WEIGHTS_0" in primitive["attributes"]:
                        weights = self.read_from_accessor(model, primitive["attributes"]["WEIGHTS_0"])
                    else:
                        weights = [MVector4D() for _ in range(len(positions))]

                    # 対応するジョイントデータ
                    try:
                        skin_joints = model.json_data["skins"][[s for s in model.json_data["nodes"] if "mesh" in s and s["mesh"] == midx][0]["skin"]]["joints"]
                    except Exception:
                        # 取れない場合はとりあえず空
                        skin_joints = []
                        
                    if "extras" not in primitive or "targetNames" not in primitive["extras"] or "targets" not in primitive:
                        continue

                    for eidx, (extra, target) in enumerate(zip(primitive["extras"]["targetNames"], primitive["targets"])):
                        # 位置データ
                        extra_positions = self.read_from_accessor(model, target["POSITION"])

                        # 法線データ
                        extra_normals = self.read_from_accessor(model, target["NORMAL"])

                        morph = Morph(extra, extra, 1, 1)
                        morph.index = eidx

                        morph_vertex_idx = vertex_idx
                        for vidx, (eposition, enormal) in enumerate(zip(extra_positions, extra_normals)):
                            model_eposition = eposition * MIKU_METER * MVector3D(-1, 1, 1)

                            morph.offsets.append(VertexMorphOffset(morph_vertex_idx, model_eposition))
                            morph_vertex_idx += 1

                        model.morphs[extra] = morph

                    for vidx, (position, normal, uv, joint, weight) in enumerate(zip(positions, normals, uvs, joints, weights)):
                        pmx_position = position * MIKU_METER * MVector3D(-1, 1, 1)

                        # 有効なINDEX番号と実際のボーンINDEXを取得
                        joint_idxs, weight_values = self.get_deform_index(vertex_idx, model, pmx_position, joint, skin_joints, node_pairs, weight)
                        if len(joint_idxs) > 1:
                            if len(joint_idxs) == 2:
                                # ウェイトが2つの場合、Bdef2
                                deform = Bdef2(joint_idxs[0], joint_idxs[1], weight_values[0])
                            else:
                                # それ以上の場合、Bdef4
                                deform = Bdef4(joint_idxs[0], joint_idxs[1], joint_idxs[2], joint_idxs[3], \
                                               weight_values[0], weight_values[1], weight_values[2], weight_values[3])
                        elif len(joint_idxs) == 1:
                            # ウェイトが1つのみの場合、Bdef1
                            deform = Bdef1(joint_idxs[0])
                        else:
                            # とりあえず除外
                            deform = Bdef1(0)

                        vertex = Vertex(vertex_idx, pmx_position, normal * MVector3D(-1, 1, 1), uv, None, deform, 1)
                        if vidx == 0:
                            # ブロック毎の開始頂点INDEXを保持
                            vertex_accessors[primitive["attributes"]["POSITION"]] = vertex_idx

                        if primitive["material"] not in model.vertices:
                            model.vertices[primitive["material"]] = []
                        model.vertices[primitive["material"]].append(vertex)

                        vertex_idx += 1

                    logger.info(f'-- 頂点データ解析[{primitive["material"]}-{primitive["attributes"]["NORMAL"]}-{primitive["attributes"]["TEXCOORD_0"]}]')

                    logger.debug(f'{midx}-{pidx}: start({model.vertices[primitive["material"]][0].index}): {[v.position.to_log() for v in model.vertices[primitive["material"]][:3]]}')
                    logger.debug(f'{midx}-{pidx}: end({model.vertices[primitive["material"]][-1].index}): {[v.position.to_log() for v in model.vertices[primitive["material"]][-3:-1]]}')
                


        return model

    def convert_bone(self, model: PmxModel):
        if 'nodes' not in model.json_data:
            logger.error("変換可能なボーン情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None, None
        
        node_dict = {}
        node_name_dict = {}
        for nidx, node in enumerate(model.json_data['nodes']):
            node = model.json_data['nodes'][nidx]
            logger.debug(f'[{nidx:03d}] node: {node}')

            node_name = node['name']
            
            # 位置
            position = MVector3D(*node['translation']) * MIKU_METER * MVector3D(-1, 1, 1)

            children = node['children'] if 'children' in node else []

            node_dict[nidx] = {'name': node_name, 'position': position, 'parent': -1, 'children': children}
            node_name_dict[node_name] = nidx

        # 親子関係設定
        for nidx, bone_param in node_dict.items():
            for midx, parent_bone_param in node_dict.items():
                if nidx in parent_bone_param['children']:
                    node_dict[nidx]['parent'] = midx
        
        # まずは人体ボーン
        bone_dict = {}
        for node_name, bone_param in BONE_PAIRS.items():
            parent_index = model.bones[bone_param['parent']].index if bone_param['parent'] else -1

            position = MVector3D()
            if parent_index >= 0:
                if node_name in node_name_dict:
                    if node_name == 'J_Bip_C_Hips':
                        position = node_dict[node_name_dict[node_name]]['position']
                    else:
                        position = model.bones[bone_param['parent']].position + node_dict[node_name_dict[node_name]]['position']
                elif node_name == 'Center':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.7
                elif node_name == 'Groove':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.8

            bone = Bone(bone_param['name'], node_name, position, parent_index, 0, bone_param['flag'])
            bone.index = len(model.bones)

            model.bones[bone.name] = bone
            bone_dict[node_name] = {'bone': bone, 'parent': bone.parent_index}

        return model, bone_dict, node_name_dict

    def convert_texture(self, model: PmxModel):
        # テクスチャ用ディレクトリ
        tex_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "tex")
        os.makedirs(tex_dir_path, exist_ok=True)
        # 展開用ディレクトリ作成
        glft_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "glTF")
        os.makedirs(glft_dir_path, exist_ok=True)

        with open(self.options.pmx_model.path, "rb") as f:
            self.buffer = f.read()

            signature = self.unpack(12, "12s")
            logger.debug("signature: %s (%s)", signature, self.offset)

            # JSON文字列読み込み
            json_buf_size = self.unpack(8, "L")
            json_text = self.read_text(json_buf_size)

            model.json_data = json.loads(json_text)
            
            # JSON出力
            jf = open(os.path.join(glft_dir_path, "gltf.json"), "w", encoding='utf-8')
            json.dump(model.json_data, jf, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
            logger.info("-- JSON出力終了")

            # binデータ
            bin_buf_size = self.unpack(8, "L")
            logger.debug(f'bin_buf_size: {bin_buf_size}')

            with open(os.path.join(glft_dir_path, "data.bin"), "wb") as bf:
                bf.write(self.buffer[self.offset:(self.offset + bin_buf_size)])

            # 空値をスフィア用に登録
            model.textures.append("")

            if "images" not in model.json_data:
                logger.error("変換可能な画像情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None

            # jsonデータの中に画像データの指定がある場合
            image_offset = 0
            for image in model.json_data['images']:
                if int(image["bufferView"]) < len(model.json_data['bufferViews']):
                    image_buffer = model.json_data['bufferViews'][int(image["bufferView"])]
                    # 画像の開始位置はオフセット分ずらす
                    image_start = self.offset + image_buffer["byteOffset"]
                    # 拡張子
                    ext = MIME_TYPE[image["mimeType"]]
                    # 画像名
                    image_name = f"{image['name']}.{ext}"
                    with open(os.path.join(glft_dir_path, image_name), "wb") as ibf:
                        ibf.write(self.buffer[image_start:(image_start + image_buffer["byteLength"])])
                    # オフセット加算
                    image_offset += image_buffer["byteLength"]
                    # PMXに追記
                    model.textures.append(os.path.join("tex", image_name))
                    # テクスチャコピー
                    shutil.copy(os.path.join(glft_dir_path, image_name), os.path.join(tex_dir_path, image_name))
            
            logger.info("-- テクスチャデータ解析終了")

        return model

    # アクセサ経由で値を取得する
    # https://github.com/ft-lab/Documents_glTF/blob/master/structure.md
    def read_from_accessor(self, model: PmxModel, accessor_idx: int):
        bresult = None
        aidx = 0
        if accessor_idx < len(model.json_data['accessors']):            
            accessor = model.json_data['accessors'][accessor_idx]
            acc_type = accessor['type']                        
            if accessor['bufferView'] < len(model.json_data['bufferViews']):
                buffer = model.json_data['bufferViews'][accessor['bufferView']]
                logger.debug('accessor: %s, %s', accessor_idx, buffer)
                if 'count' in accessor:
                    bresult = []
                    if acc_type == "VEC3":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 3) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)
                            zresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 2))

                            if buf_type == "f":
                                bresult.append(MVector3D(float(xresult[0]), float(yresult[0]), float(zresult[0])))
                            else:
                                bresult.append(MVector3D(int(xresult[0]), int(yresult[0]), int(zresult[0])))
                            
                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.debug("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "VEC2":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 2) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)

                            bresult.append(MVector2D(float(xresult[0]), float(yresult[0])))
                            
                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.debug("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "VEC4":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 4) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)
                            zresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 2))
                            wresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 3))

                            if buf_type == "f":
                                bresult.append(MVector4D(float(xresult[0]), float(yresult[0]), float(zresult[0]), float(wresult[0])))
                            else:
                                bresult.append(MVector4D(int(xresult[0]), int(yresult[0]), int(zresult[0]), int(wresult[0])))
                            
                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.debug("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "SCALAR":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
                            buf_start = self.offset + buffer["byteOffset"] + (buf_num * n)
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)

                            if buf_type == "f":
                                bresult.append(float(xresult[0]))
                            else:
                                bresult.append(int(xresult[0]))
                            
                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.debug("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

        return bresult

    def define_buf_type(self, componentType: int):
        if componentType == 5120:
            return "b", 1
        elif componentType == 5121:
            return "B", 1
        elif componentType == 5122:
            return "h", 2
        elif componentType == 5123:
            return "H", 2
        elif componentType == 5124:
            return "i", 4
        elif componentType == 5125:
            return "I", 4
        
        return "f", 4

    def read_text(self, format_size):
        bresult = self.unpack(format_size, "{0}s".format(format_size))
        return bresult.decode("UTF8")

    # 解凍して、offsetを更新する
    def unpack(self, format_size, format):
        bresult = struct.unpack_from(format, self.buffer, self.offset)

        # オフセットを更新する
        self.offset += format_size

        if bresult:
            result = bresult[0]
        else:
            result = None

        return result


BONE_PAIRS = {
    'Root': {'name': '全ての親', 'parent': None, 'tail': 'センター', 'display': '全ての親', 'flag': 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'Center': {'name': 'センター', 'parent': '全ての親', 'tail': -1, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'Groove': {'name': 'グルーブ', 'parent': 'センター', 'tail': -1, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'J_Bip_C_Hips': {'name': '腰', 'parent': 'グルーブ', 'tail': -1, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Spine': {'name': '下半身', 'parent': '腰', 'tail': -1, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Chest': {'name': '上半身', 'parent': '腰', 'tail': '上半身2', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_UpperChest': {'name': '上半身2', 'parent': '上半身', 'tail': '首', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Neck': {'name': '首', 'parent': '上半身2', 'tail': '頭', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Head': {'name': '頭', 'parent': '首', 'tail': -1, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_FaceEye': {'name': '両目', 'parent': '頭', 'tail': -1, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_L_FaceEye': {'name': '左目', 'parent': '頭', 'tail': -1, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_R_FaceEye': {'name': '右目', 'parent': '頭', 'tail': -1, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Sec_L_Bust1': {'name': '左胸', 'parent': '上半身2', 'tail': '左胸先', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Sec_L_Bust2': {'name': '左胸先', 'parent': '左胸', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Sec_R_Bust1': {'name': '右胸', 'parent': '上半身2', 'tail': '右胸先', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Sec_R_Bust2': {'name': '右胸先', 'parent': '右胸', 'tail': -1, 'display': None, 'flag': 0x0002},
    'shoulderP_L': {'name': '左肩P', 'parent': '上半身2', 'tail': -1, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Shoulder': {'name': '左肩', 'parent': '左肩P', 'tail': '左腕', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'shoulderC_L': {'name': '左肩C', 'parent': '左肩', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_UpperArm': {'name': '左腕', 'parent': '左肩C', 'tail': '左ひじ', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'arm_twist_L': {'name': '左腕捩', 'parent': '左腕', 'tail': -1, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400},
    'arm_twist_L1': {'name': '左腕捩1', 'parent': '左腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'arm_twist_L2': {'name': '左腕捩2', 'parent': '左腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'arm_twist_L3': {'name': '左腕捩3', 'parent': '左腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_LowerArm': {'name': '左ひじ', 'parent': '左腕捩', 'tail': '左手首', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'wrist_twist_L': {'name': '左手捩', 'parent': '左ひじ', 'tail': -1, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400},
    'wrist_twist_L1': {'name': '左手捩1', 'parent': '左ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'wrist_twist_L2': {'name': '左手捩2', 'parent': '左ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'wrist_twist_L3': {'name': '左手捩3', 'parent': '左ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Hand': {'name': '左手首', 'parent': '左手捩', 'tail': -1, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Thumb1': {'name': '左親指０', 'parent': '左手首', 'tail': '左親指１', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Thumb2': {'name': '左親指１', 'parent': '左親指０', 'tail': '左親指２', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Thumb3': {'name': '左親指２', 'parent': '左親指１', 'tail': '左親指先', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Thumb3_end': {'name': '左親指先', 'parent': '左親指２', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Index1': {'name': '左人指１', 'parent': '左手首', 'tail': '左人指２', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Index2': {'name': '左人指２', 'parent': '左人指１', 'tail': '左人指３', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Index3': {'name': '左人指３', 'parent': '左人指２', 'tail': '左人指先', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Index3_end': {'name': '左人指先', 'parent': '左人指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Middle1': {'name': '左中指１', 'parent': '左手首', 'tail': '左中指２', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Middle2': {'name': '左中指２', 'parent': '左中指１', 'tail': '左中指３', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Middle3': {'name': '左中指３', 'parent': '左中指２', 'tail': '左中指先', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Middle3_end': {'name': '左中指先', 'parent': '左中指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Ring1': {'name': '左薬指１', 'parent': '左手首', 'tail': '左薬指２', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Ring2': {'name': '左薬指２', 'parent': '左薬指１', 'tail': '左薬指３', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Ring3': {'name': '左薬指３', 'parent': '左薬指２', 'tail': '左薬指先', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Ring3_end': {'name': '左薬指先', 'parent': '左薬指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Little1': {'name': '左小指１', 'parent': '左手首', 'tail': '左小指２', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Little2': {'name': '左小指２', 'parent': '左小指１', 'tail': '左小指３', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Little3': {'name': '左小指３', 'parent': '左小指２', 'tail': '左小指先', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Little3_end': {'name': '左小指先', 'parent': '左小指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'shoulderP_R': {'name': '右肩P', 'parent': '上半身2', 'tail': -1, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Shoulder': {'name': '右肩', 'parent': '右肩P', 'tail': '右腕', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'shoulderC_R': {'name': '右肩C', 'parent': '右肩', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_UpperArm': {'name': '右腕', 'parent': '右肩C', 'tail': '右ひじ', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'arm_twist_R': {'name': '右腕捩', 'parent': '右腕', 'tail': -1, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400},
    'arm_twist_R1': {'name': '右腕捩1', 'parent': '右腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'arm_twist_R2': {'name': '右腕捩2', 'parent': '右腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'arm_twist_R3': {'name': '右腕捩3', 'parent': '右腕', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_LowerArm': {'name': '右ひじ', 'parent': '右腕捩', 'tail': '右手首', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'wrist_twist_R': {'name': '右手捩', 'parent': '右ひじ', 'tail': -1, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400},
    'wrist_twist_R1': {'name': '右手捩1', 'parent': '右ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'wrist_twist_R2': {'name': '右手捩2', 'parent': '右ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'wrist_twist_R3': {'name': '右手捩3', 'parent': '右ひじ', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Hand': {'name': '右手首', 'parent': '右手捩', 'tail': -1, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Thumb1': {'name': '右親指０', 'parent': '右手首', 'tail': '右親指１', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Thumb2': {'name': '右親指１', 'parent': '右親指０', 'tail': '右親指２', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Thumb3': {'name': '右親指２', 'parent': '右親指１', 'tail': '右親指先', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Thumb3_end': {'name': '右親指先', 'parent': '右親指２', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Index1': {'name': '右人指１', 'parent': '右手首', 'tail': '右人指２', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Index2': {'name': '右人指２', 'parent': '右人指１', 'tail': '右人指３', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Index3': {'name': '右人指３', 'parent': '右人指２', 'tail': '右人指先', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Index3_end': {'name': '右人指先', 'parent': '右人指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Middle1': {'name': '右中指１', 'parent': '右手首', 'tail': '右中指２', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Middle2': {'name': '右中指２', 'parent': '右中指１', 'tail': '右中指３', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Middle3': {'name': '右中指３', 'parent': '右中指２', 'tail': '右中指先', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Middle3_end': {'name': '右中指先', 'parent': '右中指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Ring1': {'name': '右薬指１', 'parent': '右手首', 'tail': '右薬指２', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Ring2': {'name': '右薬指２', 'parent': '右薬指１', 'tail': '右薬指３', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Ring3': {'name': '右薬指３', 'parent': '右薬指２', 'tail': '右薬指先', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Ring3_end': {'name': '右薬指先', 'parent': '右薬指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Little1': {'name': '右小指１', 'parent': '右手首', 'tail': '右小指２', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Little2': {'name': '右小指２', 'parent': '右小指１', 'tail': '右小指３', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Little3': {'name': '右小指３', 'parent': '右小指２', 'tail': '右小指先', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Little3_end': {'name': '右小指先', 'parent': '右小指３', 'tail': -1, 'display': None, 'flag': 0x0002},
    'leftWaistCancel': {'name': '腰キャンセル左', 'parent': '下半身', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_L_UpperLeg': {'name': '左足', 'parent': '腰キャンセル左', 'tail': '左ひざ', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_LowerLeg': {'name': '左ひざ', 'parent': '左足', 'tail': '左足首', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Foot': {'name': '左足首', 'parent': '左ひざ', 'tail': '左つま先', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_ToeBase_end': {'name': '左つま先', 'parent': '左足首', 'tail': -1, 'display': None, 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_IK_L': {'name': '左足ＩＫ', 'parent': '全ての親', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'toe_IK_L': {'name': '左つま先ＩＫ', 'parent': '左足ＩＫ', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'rightWaistCancel': {'name': '腰キャンセル右', 'parent': '下半身', 'tail': -1, 'display': None, 'flag': 0x0002},
    'J_Bip_R_UpperLeg': {'name': '右足', 'parent': '腰キャンセル右', 'tail': '右ひざ', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_LowerLeg': {'name': '右ひざ', 'parent': '右足', 'tail': '右足首', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Foot': {'name': '右足首', 'parent': '右ひざ', 'tail': '右つま先', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_ToeBase_end': {'name': '右つま先', 'parent': '右足首', 'tail': -1, 'display': None, 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_IK_R': {'name': '右足ＩＫ', 'parent': '全ての親', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'toe_IK_R': {'name': '右つま先ＩＫ', 'parent': '右足ＩＫ', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'leg_LD': {'name': '左足D', 'parent': '腰キャンセル左', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'knee_LD': {'name': '左ひざD', 'parent': '左足D', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'ankle_LD': {'name': '左足首D', 'parent': '左ひざD', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_ToeBase': {'name': '左足先EX', 'parent': '左足首D', 'tail': -1, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_RD': {'name': '右足D', 'parent': '腰キャンセル右', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'knee_RD': {'name': '右ひざD', 'parent': '右足D', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'ankle_RD': {'name': '右足首D', 'parent': '右ひざD', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_ToeBase': {'name': '右足先EX', 'parent': '右足首D', 'tail': -1, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010},
}
