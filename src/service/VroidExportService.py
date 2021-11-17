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
            model, tex_dir_path = self.convert_texture(PmxModel())
            if not model:
                return False

            model, bone_name_dict, node_name_dict = self.convert_bone(model)
            if not model:
                return False

            model = self.convert_mesh(model, bone_name_dict, node_name_dict, tex_dir_path)
            if not model:
                return False
          
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

    def convert_mesh(self, model: PmxModel, bone_name_dict: dict, node_name_dict: dict, tex_dir_path: str):
        if 'meshes' not in model.json_data:
            logger.error("変換可能なメッシュ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None

        vertex_blocks = {}
        vertex_idx = 0

        for midx, mesh in enumerate(model.json_data["meshes"]):
            if "primitives" not in mesh:
                continue
            
            for pidx, primitive in enumerate(mesh["primitives"]):
                if "attributes" not in primitive or "indices" not in primitive or "material" not in primitive:
                    continue
                
                # 頂点ブロック
                vertex_key = f'{primitive["attributes"]["JOINTS_0"]}-{primitive["attributes"]["NORMAL"]}-{primitive["attributes"]["POSITION"]}-{primitive["attributes"]["TEXCOORD_0"]}-{primitive["attributes"]["WEIGHTS_0"]}'  # noqa

                # 頂点データ
                if vertex_key not in vertex_blocks:
                    vertex_blocks[vertex_key] = {'vertices': [], 'start': vertex_idx, 'indices': [], 'materials': []}

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
                        
                    # if "extras" not in primitive or "targetNames" not in primitive["extras"] or "targets" not in primitive:
                    #     continue

                    # for eidx, (extra, target) in enumerate(zip(primitive["extras"]["targetNames"], primitive["targets"])):
                    #     # 位置データ
                    #     extra_positions = self.read_from_accessor(model, target["POSITION"])

                    #     # 法線データ
                    #     extra_normals = self.read_from_accessor(model, target["NORMAL"])

                    #     morph = Morph(extra, extra, 1, 1)
                    #     morph.index = eidx

                    #     morph_vertex_idx = vertex_idx
                    #     for vidx, (eposition, enormal) in enumerate(zip(extra_positions, extra_normals)):
                    #         model_eposition = eposition * MIKU_METER * MVector3D(-1, 1, 1)

                    #         morph.offsets.append(VertexMorphOffset(morph_vertex_idx, model_eposition))
                    #         morph_vertex_idx += 1

                    #     model.morphs[extra] = morph

                    for position, normal, uv, joint, weight in zip(positions, normals, uvs, joints, weights):
                        model_position = position * MIKU_METER * MVector3D(-1, 1, 1)

                    #     # 有効なINDEX番号と実際のボーンINDEXを取得
                    #     joint_idxs, weight_values = self.get_deform_index(vertex_idx, model, pmx_position, joint, skin_joints, node_pairs, weight)
                    #     if len(joint_idxs) > 1:
                    #         if len(joint_idxs) == 2:
                    #             # ウェイトが2つの場合、Bdef2
                    #             deform = Bdef2(joint_idxs[0], joint_idxs[1], weight_values[0])
                    #         else:
                    #             # それ以上の場合、Bdef4
                    #             deform = Bdef4(joint_idxs[0], joint_idxs[1], joint_idxs[2], joint_idxs[3], \
                    #                            weight_values[0], weight_values[1], weight_values[2], weight_values[3])
                    #     elif len(joint_idxs) == 1:
                    #         # ウェイトが1つのみの場合、Bdef1
                    #         deform = Bdef1(joint_idxs[0])
                    #     else:
                    #         # とりあえず除外
                    #         deform = Bdef1(0)
                        deform = Bdef1(0)

                        vertex = Vertex(vertex_idx, model_position, normal * MVector3D(-1, 1, 1), uv, None, deform, 1)

                        model.vertex_dict[vertex_idx] = vertex
                        vertex_blocks[vertex_key]['vertices'].append(vertex_idx)
                        vertex_idx += 1

                    logger.info('-- 頂点データ解析[%s]', vertex_key)
                
                vertex_blocks[vertex_key]['indices'].append(primitive["indices"])
                vertex_blocks[vertex_key]['materials'].append(primitive["material"])

        hair_regexp = r'((N\d+_\d+_Hair_\d+)_HAIR)'
        hair_tex_regexp = r'_(\d+)'

        indices_by_materials = {}
        materials_by_type = {}

        for vertex_key, vertex_dict in vertex_blocks.items():
            start_vidx = vertex_dict['start']
            indices = vertex_dict['indices']
            materials = vertex_dict['materials']

            for index_accessor, material_accessor in zip(indices, materials):
                # 材質データ ---------------
                vrm_material = model.json_data["materials"][material_accessor]
                material_name = vrm_material['name']

                # 材質順番を決める
                material_key = vrm_material["alphaMode"]
                if "EyeIris" in material_name:
                    material_key = "EyeIris"
                if "EyeHighlight" in material_name:
                    material_key = "EyeHighlight"
                if "EyeWhite" in material_name:
                    material_key = "EyeWhite"
                if "Eyelash" in material_name:
                    material_key = "Eyelash"
                if "Eyeline" in material_name:
                    material_key = "Eyeline"
                if "FaceBrow" in material_name:
                    material_key = "FaceBrow"
                if "Lens" in material_name:
                    material_key = "Lens"

                if material_key not in materials_by_type:
                    materials_by_type[material_key] = {}

                if material_name not in materials_by_type[material_key]:
                    # VRMの材質拡張情報
                    material_ext = [m for m in model.json_data["extensions"]["VRM"]["materialProperties"] if m["name"] == material_name][0]
                    # 拡散色
                    diffuse_color_data = vrm_material["pbrMetallicRoughness"]["baseColorFactor"]
                    diffuse_color = MVector3D(*diffuse_color_data[:3])
                    # 非透過度
                    alpha = diffuse_color_data[3]
                    # 反射色
                    if "emissiveFactor" in vrm_material:
                        specular_color_data = vrm_material["emissiveFactor"]
                        specular_color = MVector3D(*specular_color_data[:3])
                    else:
                        specular_color = MVector3D()
                    specular_factor = 0
                    # 環境色
                    if "vectorProperties" in material_ext and "_ShadeColor" in material_ext["vectorProperties"]:
                        ambient_color = MVector3D(*material_ext["vectorProperties"]["_ShadeColor"][:3])
                    else:
                        ambient_color = diffuse_color / 2
                    # 0x02:地面影, 0x04:セルフシャドウマップへの描画, 0x08:セルフシャドウの描画
                    flag = 0x02 | 0x04 | 0x08
                    if vrm_material["doubleSided"]:
                        # 両面描画
                        flag |= 0x01
                    edge_color = MVector4D(*material_ext["vectorProperties"]["_OutlineColor"])
                    edge_size = material_ext["floatProperties"]["_OutlineWidth"]

                    # 0番目は空テクスチャなので+1で設定
                    m = re.search(hair_regexp, material_name)
                    if m is not None:
                        # 髪材質の場合、合成
                        hair_img_name = os.path.basename(model.textures[material_ext["textureProperties"]["_MainTex"] + 1])
                        hm = re.search(hair_tex_regexp, hair_img_name)
                        hair_img_number = -1
                        if hm is not None:
                            hair_img_number = int(hm.groups()[0])
                        hair_spe_name = f'_{(hair_img_number + 1):02d}.png'
                        hair_blend_name = f'_{hair_img_number:02d}_blend.png'

                        if os.path.exists(os.path.join(tex_dir_path, hair_img_name)) and os.path.exists(os.path.join(tex_dir_path, hair_spe_name)):
                            # スペキュラファイルがある場合
                            hair_img = Image.open(os.path.join(tex_dir_path, hair_img_name))
                            hair_ary = np.array(hair_img)

                            spe_img = Image.open(os.path.join(tex_dir_path, hair_spe_name))
                            spe_ary = np.array(spe_img)

                            # 拡散色の画像
                            diffuse_ary = np.array(material_ext["vectorProperties"]["_Color"])
                            diffuse_img = Image.fromarray(np.tile(diffuse_ary * 255, (hair_ary.shape[0], hair_ary.shape[1], 1)).astype(np.uint8))
                            hair_diffuse_img = ImageChops.multiply(hair_img, diffuse_img)

                            # 反射色の画像
                            if "emissiveFactor" in vrm_material:
                                emissive_ary = np.array(vrm_material["emissiveFactor"])
                                emissive_ary = np.append(emissive_ary, 1)
                            else:
                                emissive_ary = np.array([0, 0, 0, 1])
                            emissive_img = Image.fromarray(np.tile(emissive_ary * 255, (spe_ary.shape[0], spe_ary.shape[1], 1)).astype(np.uint8))
                            # 乗算
                            hair_emissive_img = ImageChops.multiply(spe_img, emissive_img)
                            # スクリーン
                            dest_img = ImageChops.screen(hair_diffuse_img, hair_emissive_img)
                            dest_img.save(os.path.join(tex_dir_path, hair_blend_name))

                            model.textures.append(os.path.join("tex", hair_blend_name))
                            texture_index = len(model.textures) - 1

                            # 拡散色と環境色は固定
                            diffuse_color = MVector3D(1, 1, 1)
                            specular_color = MVector3D()
                            ambient_color = diffuse_color / 2
                        else:
                            # スペキュラがない場合、ないし反映させない場合、そのまま設定
                            texture_index = material_ext["textureProperties"]["_MainTex"] + 1
                    else:
                        # そのまま出力
                        texture_index = material_ext["textureProperties"]["_MainTex"] + 1
                    
                    sphere_texture_index = 0
                    sphere_mode = 0
                    if "_SphereAdd" in material_ext["textureProperties"]:
                        sphere_texture_index = material_ext["textureProperties"]["_SphereAdd"] + 1
                        # 加算スフィア
                        sphere_mode = 2

                    if "vectorProperties" in material_ext and "_ShadeColor" in material_ext["vectorProperties"]:
                        toon_sharing_flag = 0
                        toon_img_name = f'{material_name}_TOON.bmp'
                        
                        toon_light_ary = np.tile(np.array([255, 255, 255, 255]), (24, 32, 1))
                        toon_shadow_ary = np.tile(np.array(material_ext["vectorProperties"]["_ShadeColor"]) * 255, (8, 32, 1))
                        toon_ary = np.concatenate((toon_light_ary, toon_shadow_ary), axis=0)
                        toon_img = Image.fromarray(toon_ary.astype(np.uint8))

                        toon_img.save(os.path.join(tex_dir_path, toon_img_name))
                        model.textures.append(os.path.join("tex", toon_img_name))
                        # 最後に追加したテクスチャをINDEXとして設定
                        toon_texture_index = len(model.textures) - 1
                    else:
                        toon_sharing_flag = 1
                        toon_texture_index = 1

                    material = Material(material_name, material_name, diffuse_color, alpha, specular_factor, specular_color, \
                                        ambient_color, flag, edge_color, edge_size, texture_index, sphere_texture_index, sphere_mode, toon_sharing_flag, \
                                        toon_texture_index, "", 0)
                    materials_by_type[material_key][material.name] = material
                    indices_by_materials[material.name] = {}
                else:
                    material = materials_by_type[material_key][material_name]

                # 面データ ---------------
                indices = self.read_from_accessor(model, index_accessor)
                indices_by_materials[material.name][index_accessor] = (np.array(indices) + start_vidx).tolist()
                material.vertex_count += len(indices)

                logger.info('-- 面・材質データ解析[%s-%s]', index_accessor, material_accessor)
        
        # 材質を不透明(OPAQUE)→透明順(BLEND)に並べ替て設定
        index_idx = 0
        for material_type in ["OPAQUE", "MASK", "BLEND", "FaceBrow", "Eyeline", "Eyelash", "EyeWhite", "EyeIris", "EyeHighlight", "Lens"]:
            if material_type in materials_by_type:
                for material_name, material in materials_by_type[material_type].items():
                    model.materials[material.name] = material
                    for index_accessor, indices in indices_by_materials[material.name].items():
                        for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):
                            # 面の貼り方がPMXは逆
                            model.indices[index_idx] = [v2_idx, v1_idx, v0_idx]
                            index_idx += 1

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

            node_dict[nidx] = {'name': node_name, 'relative_position': position, 'position': position, 'parent': -1, 'children': children}
            node_name_dict[node_name] = nidx

        # 親子関係設定
        for nidx, node_param in node_dict.items():
            for midx, parent_node_param in node_dict.items():
                if nidx in parent_node_param['children']:
                    node_dict[nidx]['parent'] = midx
        
        # 絶対位置計算
        for nidx, node_param in node_dict.items():
            node_dict[nidx]['position'] = self.calc_bone_position(model, node_dict, node_param)
        
        # まずは人体ボーン
        bone_name_dict = {}
        for node_name, bone_param in BONE_PAIRS.items():
            parent_name = BONE_PAIRS[bone_param['parent']]['name'] if bone_param['parent'] else None
            parent_index = model.bones[parent_name].index if parent_name else -1

            position = MVector3D()
            bone = Bone(bone_param['name'], node_name, position, parent_index, 0, bone_param['flag'])
            if parent_index >= 0:
                if node_name in node_name_dict:
                    position = node_dict[node_name_dict[node_name]]['position'].copy()
                elif node_name == 'Center':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.7
                elif node_name == 'Groove':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.8
                elif node_name == 'J_Bip_C_Spine2':
                    position = node_dict[node_name_dict['J_Bip_C_Spine']]['position'].copy()
                elif node_name == 'J_Adj_FaceEye':
                    position = node_dict[node_name_dict['J_Adj_L_FaceEye']]['position'] + \
                                ((node_dict[node_name_dict['J_Adj_R_FaceEye']]['position'] - node_dict[node_name_dict['J_Adj_L_FaceEye']]['position']) * 0.5)   # noqa
                elif 'shoulderP_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Shoulder']]['position'].copy()
                elif 'shoulderC_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'shoulderP_{node_name[-1]}']['index']
                    bone.effect_factor = -1
                elif 'arm_twist_' in node_name:
                    factor = 0.25 if node_name[-2] == '1' else 0.75 if node_name[-2] == '3' else 0.5
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position'] + \
                                ((node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position'] - node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position']) * factor)   # noqa
                    if node_name[-2] in ['1', '2', '3']:
                        bone.effect_index = bone_name_dict[f'arm_twist_{node_name[-1]}']['index']
                        bone.effect_factor = factor
                elif 'wrist_twist_' in node_name:
                    factor = 0.25 if node_name[-2] == '1' else 0.75 if node_name[-2] == '3' else 0.5
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position'] + \
                                ((node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Hand']]['position'] - node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position']) * factor)   # noqa
                    if node_name[-2] in ['1', '2', '3']:
                        bone.effect_index = bone_name_dict[f'wrist_twist_{node_name[-1]}']['index']
                        bone.effect_factor = factor
                elif 'waistCancel_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']]['position'].copy()
                elif 'leg_IK_Parent_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                    position.setY(0)
                elif 'leg_IK_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                elif 'toe_IK_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_ToeBase']]['position'].copy()
                elif 'leg_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'knee_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerLeg']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_LowerLeg']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'ankle_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_Foot']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'toe_EX_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'] + \
                                ((node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_ToeBase']]['position'] - node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position']) * 0.8)   # noqa
                    bone.layer = 1
            bone.position = position
            bone.index = len(model.bones)

            model.bones[bone.name] = bone
            bone_name_dict[node_name] = {'index': bone.index, 'name': bone.name}
        
        # 人体以外のボーン
        for nidx, node_param in node_dict.items():
            if node_param['name'] not in bone_name_dict:
                bone = Bone(node_param['name'], node_param['name'], node_param['position'], -1, 0, 0x0002 | 0x0008 | 0x0010)
                parent_index = bone_name_dict[node_dict[node_param['parent']]['name']]['index'] if node_param['parent'] in node_dict and node_dict[node_param['parent']]['name'] in bone_name_dict else -1   # noqa
                bone.parent_index = parent_index
                bone.index = len(model.bones)
                model.bones[bone.name] = bone
                bone_name_dict[node_param['name']] = {'index': bone.index, 'name': bone.name}

        local_y_vector = MVector3D(0, -1, 0)

        # 表示先・ローカル軸・IK設定
        for bone in model.bones.values():
            # 人体ボーン
            if bone.english_name in BONE_PAIRS:
                # 表示先
                tail = BONE_PAIRS[bone.english_name]['tail']
                if tail:
                    if type(tail) is MVector3D:
                        bone.tail_position = tail.copy()
                    else:
                        bone.tail_index = bone_name_dict[tail]['index']
                if bone.name == '下半身':
                    # 腰は表示順が上なので、相対指定
                    bone.tail_position = model.bones['腰'].position - bone.position

                # ローカル軸
                direction = bone.name[0]
                arm_bone_name = f'{direction}腕'
                elbow_bone_name = f'{direction}ひじ'
                wrist_bone_name = f'{direction}手首'
                
                if bone.name in ['右肩', '左肩'] and arm_bone_name in model.bones:
                    bone.local_x_vector = (model.bones[arm_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector)
                if bone.name in ['右ひじ', '左ひじ', '右手首', '左手首'] and elbow_bone_name in model.bones:
                    bone.local_x_vector = (model.bones[elbow_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector)
                # 捩り
                if bone.name in ['右腕捩', '左腕捩'] and arm_bone_name in model.bones and elbow_bone_name in model.bones:
                    bone.fixed_axis = (model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position).normalized()
                    bone.local_x_vector = (model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector)
                if bone.name in ['右手捩', '左手捩'] and elbow_bone_name in model.bones and wrist_bone_name in model.bones:
                    bone.fixed_axis = (model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position).normalized()
                    bone.local_x_vector = (model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector)
                # 指
                if BONE_PAIRS[bone.english_name]['display'] and '指' in BONE_PAIRS[bone.english_name]['display']:
                    bone.local_x_vector = (model.bones[bone_name_dict[BONE_PAIRS[bone.english_name]['tail']]['name']].position - model.bones[bone_name_dict[BONE_PAIRS[bone.english_name]['parent']]['name']].position).normalized()    # noqa
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector)

                # 足IK
                leg_name = f'{direction}足'
                knee_name = f'{direction}ひざ'
                ankle_name = f'{direction}足首'
                toe_name = f'{direction}つま先'

                if bone.name in ['右足ＩＫ', '左足ＩＫ'] and leg_name in model.bones and knee_name in model.bones and ankle_name in model.bones:
                    leg_ik_link = []
                    leg_ik_link.append(IkLink(model.bones[knee_name].index, 1, MVector3D(math.radians(-180), 0, 0), MVector3D(math.radians(-0.5), 0, 0)))
                    leg_ik_link.append(IkLink(model.bones[leg_name].index, 0))
                    leg_ik = Ik(model.bones[ankle_name].index, 40, 1, leg_ik_link)
                    bone.ik = leg_ik

                if bone.name in ['右つま先ＩＫ', '左つま先ＩＫ'] and ankle_name in model.bones and toe_name in model.bones:
                    toe_ik_link = []
                    toe_ik_link.append(IkLink(model.bones[ankle_name].index, 0))
                    toe_ik = Ik(model.bones[toe_name].index, 40, 1, toe_ik_link)
                    bone.ik = toe_ik
            else:
                # 人体以外
                # 表示先
                node_param = node_dict[node_name_dict[bone.name]]
                tail_index = bone_name_dict[node_dict[node_param['children'][0]]['name']]['index'] if node_param['children'] and node_param['children'][0] in node_dict and node_dict[node_param['children'][0]]['name'] in bone_name_dict else -1   # noqa
                if tail_index >= 0:
                    bone.tail_index = tail_index
                    bone.flag |= 0x0001

        return model, bone_name_dict, node_name_dict
    
    def calc_bone_position(self, model: PmxModel, node_dict: dict, node_param: dict):
        if node_param['parent'] == -1:
            return node_param['relative_position']

        return node_param['relative_position'] + self.calc_bone_position(model, node_dict, node_dict[node_param['parent']])

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

        return model, tex_dir_path

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
    'Root': {'name': '全ての親', 'parent': None, 'tail': 'Center', 'display': '全ての親', 'flag': 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'Center': {'name': 'センター', 'parent': 'Root', 'tail': None, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'Groove': {'name': 'グルーブ', 'parent': 'Center', 'tail': None, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'J_Bip_C_Hips': {'name': '腰', 'parent': 'Groove', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Spine': {'name': '下半身', 'parent': 'J_Bip_C_Hips', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Spine2': {'name': '上半身', 'parent': 'J_Bip_C_Hips', 'tail': 'J_Bip_C_Chest', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Chest': {'name': '上半身2', 'parent': 'J_Bip_C_Spine2', 'tail': 'J_Bip_C_Neck', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Neck': {'name': '首', 'parent': 'J_Bip_C_Chest', 'tail': 'J_Bip_C_Head', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_C_Head': {'name': '頭', 'parent': 'J_Bip_C_Neck', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_FaceEye': {'name': '両目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_L_FaceEye': {'name': '左目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Adj_R_FaceEye': {'name': '右目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Sec_L_Bust1': {'name': '左胸', 'parent': 'J_Bip_C_Chest', 'tail': 'J_Sec_L_Bust2', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Sec_L_Bust2': {'name': '左胸先', 'parent': 'J_Sec_L_Bust1', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Sec_R_Bust1': {'name': '右胸', 'parent': 'J_Bip_C_Chest', 'tail': 'J_Sec_R_Bust2', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Sec_R_Bust2': {'name': '右胸先', 'parent': 'J_Sec_R_Bust1', 'tail': None, 'display': None, 'flag': 0x0002},
    'shoulderP_L': {'name': '左肩P', 'parent': 'J_Bip_C_Chest', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Shoulder': {'name': '左肩', 'parent': 'shoulderP_L', 'tail': 'J_Bip_L_UpperArm', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'shoulderC_L': {'name': '左肩C', 'parent': 'J_Bip_L_Shoulder', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_L_UpperArm': {'name': '左腕', 'parent': 'shoulderC_L', 'tail': 'J_Bip_L_LowerArm', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'arm_twist_L': {'name': '左腕捩', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800 | 0x0800},
    'arm_twist_1L': {'name': '左腕捩1', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'arm_twist_2L': {'name': '左腕捩2', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'arm_twist_3L': {'name': '左腕捩3', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_L_LowerArm': {'name': '左ひじ', 'parent': 'arm_twist_L', 'tail': 'J_Bip_L_Hand', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'wrist_twist_L': {'name': '左手捩', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800},
    'wrist_twist_1L': {'name': '左手捩1', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'wrist_twist_2L': {'name': '左手捩2', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'wrist_twist_3L': {'name': '左手捩3', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_L_Hand': {'name': '左手首', 'parent': 'wrist_twist_L', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Thumb1': {'name': '左親指０', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Thumb2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Thumb2': {'name': '左親指１', 'parent': 'J_Bip_L_Thumb1', 'tail': 'J_Bip_L_Thumb3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Thumb3': {'name': '左親指２', 'parent': 'J_Bip_L_Thumb2', 'tail': 'J_Bip_L_Thumb3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Thumb3_end': {'name': '左親指先', 'parent': 'J_Bip_L_Thumb3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Index1': {'name': '左人指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Index2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Index2': {'name': '左人指２', 'parent': 'J_Bip_L_Index1', 'tail': 'J_Bip_L_Index3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Index3': {'name': '左人指３', 'parent': 'J_Bip_L_Index2', 'tail': 'J_Bip_L_Index3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Index3_end': {'name': '左人指先', 'parent': 'J_Bip_L_Index3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Middle1': {'name': '左中指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Middle2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Middle2': {'name': '左中指２', 'parent': 'J_Bip_L_Middle1', 'tail': 'J_Bip_L_Middle3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Middle3': {'name': '左中指３', 'parent': 'J_Bip_L_Middle2', 'tail': 'J_Bip_L_Middle3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Middle3_end': {'name': '左中指先', 'parent': 'J_Bip_L_Middle3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Ring1': {'name': '左薬指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Ring2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Ring2': {'name': '左薬指２', 'parent': 'J_Bip_L_Ring1', 'tail': 'J_Bip_L_Ring3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Ring3': {'name': '左薬指３', 'parent': 'J_Bip_L_Ring2', 'tail': 'J_Bip_L_Ring3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Ring3_end': {'name': '左薬指先', 'parent': 'J_Bip_L_Ring3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_L_Little1': {'name': '左小指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Little2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Little2': {'name': '左小指２', 'parent': 'J_Bip_L_Little1', 'tail': 'J_Bip_L_Little3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Little3': {'name': '左小指３', 'parent': 'J_Bip_L_Little2', 'tail': 'J_Bip_L_Little3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_L_Little3_end': {'name': '左小指先', 'parent': 'J_Bip_L_Little3', 'tail': None, 'display': None, 'flag': 0x0002},
    'shoulderP_R': {'name': '右肩P', 'parent': 'J_Bip_C_Chest', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Shoulder': {'name': '右肩', 'parent': 'shoulderP_R', 'tail': 'J_Bip_R_UpperArm', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'shoulderC_R': {'name': '右肩C', 'parent': 'J_Bip_R_Shoulder', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_R_UpperArm': {'name': '右腕', 'parent': 'shoulderC_R', 'tail': 'J_Bip_R_LowerArm', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'arm_twist_R': {'name': '右腕捩', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800},
    'arm_twist_1R': {'name': '右腕捩1', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'arm_twist_2R': {'name': '右腕捩2', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'arm_twist_3R': {'name': '右腕捩3', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_R_LowerArm': {'name': '右ひじ', 'parent': 'arm_twist_R', 'tail': 'J_Bip_R_Hand', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'wrist_twist_R': {'name': '右手捩', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800},
    'wrist_twist_1R': {'name': '右手捩1', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'wrist_twist_2R': {'name': '右手捩2', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'wrist_twist_3R': {'name': '右手捩3', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100},
    'J_Bip_R_Hand': {'name': '右手首', 'parent': 'wrist_twist_R', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Thumb1': {'name': '右親指０', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Thumb2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Thumb2': {'name': '右親指１', 'parent': 'J_Bip_R_Thumb1', 'tail': 'J_Bip_R_Thumb3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Thumb3': {'name': '右親指２', 'parent': 'J_Bip_R_Thumb2', 'tail': 'J_Bip_R_Thumb3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Thumb3_end': {'name': '右親指先', 'parent': 'J_Bip_R_Thumb3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Index1': {'name': '右人指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Index2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Index2': {'name': '右人指２', 'parent': 'J_Bip_R_Index1', 'tail': 'J_Bip_R_Index3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Index3': {'name': '右人指３', 'parent': 'J_Bip_R_Index2', 'tail': 'J_Bip_R_Index3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Index3_end': {'name': '右人指先', 'parent': 'J_Bip_R_Index3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Middle1': {'name': '右中指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Middle2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Middle2': {'name': '右中指２', 'parent': 'J_Bip_R_Middle1', 'tail': 'J_Bip_R_Middle3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Middle3': {'name': '右中指３', 'parent': 'J_Bip_R_Middle2', 'tail': 'J_Bip_R_Middle3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Middle3_end': {'name': '右中指先', 'parent': 'J_Bip_R_Middle3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Ring1': {'name': '右薬指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Ring2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Ring2': {'name': '右薬指２', 'parent': 'J_Bip_R_Ring1', 'tail': 'J_Bip_R_Ring3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Ring3': {'name': '右薬指３', 'parent': 'J_Bip_R_Ring2', 'tail': 'J_Bip_R_Ring3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Ring3_end': {'name': '右薬指先', 'parent': 'J_Bip_R_Ring3', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_R_Little1': {'name': '右小指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Little2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Little2': {'name': '右小指２', 'parent': 'J_Bip_R_Little1', 'tail': 'J_Bip_R_Little3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Little3': {'name': '右小指３', 'parent': 'J_Bip_R_Little2', 'tail': 'J_Bip_R_Little3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800},
    'J_Bip_R_Little3_end': {'name': '右小指先', 'parent': 'J_Bip_R_Little3', 'tail': None, 'display': None, 'flag': 0x0002},
    'waistCancel_L': {'name': '腰キャンセル左', 'parent': 'J_Bip_C_Spine', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_L_UpperLeg': {'name': '左足', 'parent': 'waistCancel_L', 'tail': 'J_Bip_L_LowerLeg', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_LowerLeg': {'name': '左ひざ', 'parent': 'J_Bip_L_UpperLeg', 'tail': 'J_Bip_L_Foot', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_Foot': {'name': '左足首', 'parent': 'J_Bip_L_LowerLeg', 'tail': 'J_Bip_L_ToeBase', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_L_ToeBase': {'name': '左つま先', 'parent': 'J_Bip_L_Foot', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_IK_Parent_L': {'name': '左足IK親', 'parent': 'Root', 'tail': 'leg_IK_L', 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'leg_IK_L': {'name': '左足ＩＫ', 'parent': 'leg_IK_Parent_L', 'tail': MVector3D(0, 0, -1), 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020},
    'toe_IK_L': {'name': '左つま先ＩＫ', 'parent': 'leg_IK_L', 'tail': MVector3D(0, -1, 0), 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020},
    'waistCancel_R': {'name': '腰キャンセル右', 'parent': 'J_Bip_C_Spine', 'tail': None, 'display': None, 'flag': 0x0002},
    'J_Bip_R_UpperLeg': {'name': '右足', 'parent': 'waistCancel_R', 'tail': 'J_Bip_R_LowerLeg', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_LowerLeg': {'name': '右ひざ', 'parent': 'J_Bip_R_UpperLeg', 'tail': 'J_Bip_R_Foot', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_Foot': {'name': '右足首', 'parent': 'J_Bip_R_LowerLeg', 'tail': 'J_Bip_R_ToeBase', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010},
    'J_Bip_R_ToeBase': {'name': '右つま先', 'parent': 'J_Bip_R_Foot', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_IK_Parent_R': {'name': '右足IK親', 'parent': 'Root', 'tail': 'leg_IK_R', 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010},
    'leg_IK_R': {'name': '右足ＩＫ', 'parent': 'leg_IK_Parent_R', 'tail': MVector3D(0, 0, -1), 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020},
    'toe_IK_R': {'name': '右つま先ＩＫ', 'parent': 'leg_IK_R', 'tail': MVector3D(0, -1, 0), 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020},
    'leg_D_L': {'name': '左足D', 'parent': 'waistCancel_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'knee_D_L': {'name': '左ひざD', 'parent': 'leg_D_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'ankle_D_L': {'name': '左足首D', 'parent': 'knee_D_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'toe_EX_L': {'name': '左足先EX', 'parent': 'ankle_D_L', 'tail': MVector3D(0, 0, -1), 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010},
    'leg_D_R': {'name': '右足D', 'parent': 'waistCancel_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'knee_D_R': {'name': '右ひざD', 'parent': 'leg_D_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'ankle_D_R': {'name': '右足首D', 'parent': 'knee_D_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100},
    'toe_EX_R': {'name': '右足先EX', 'parent': 'ankle_D_R', 'tail': MVector3D(0, 0, -1), 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010},
}
