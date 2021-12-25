# -*- coding: utf-8 -*-
#
import unittest
import sys
import pathlib
# このソースのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + '/../')
sys.path.append(str(current_dir) + '/../src/')

from mmd.PmxReader import PmxReader # noqa
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint # noqa
from mmd.VmdData import VmdMotion, VmdBoneFrame, VmdCameraFrame, VmdInfoIk, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame # noqa
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from module.MOptions import MExportOptions # noqa
from service.PmxTailorExportService import PmxTailorExportService # noqa
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__, level=1)


class PmxTailorExportServiceTest(unittest.TestCase):

    def test_create_vertex_map_by_index_01(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\_報告\\niki修正中\\日光一文字ver.1.0Tailorテスト版9_クリア.pmx").read_data()

        for index_idx, direction, vertex_map in [(138026, '下', {117371: {'x': 0, 'y': 0}, 115630: {'x': 1, 'y': 1}, 120679: {'x': 0, 'y': 1}}), \
                                                 (138198, '下', {115858: {'x': 0, 'y': 0}, 117461: {'x': -1, 'y': -1}, 120837: {'x': 0, 'y': -1}}), \
                                                 (138968, '下', {115656: {'x': 0, 'y': 0}, 120942: {'x': 0, 'y': 1}, 117507: {'x': -1, 'y': 1}}), \
                                                 (138790, '下', {120750: {'x': 0, 'y': 0}, 115836: {'x': 0, 'y': -1}, 120774: {'x': 1, 'y': -1}}), \
                                                 (138011, '下', {115634: {'x': 0, 'y': 0}, 117369: {'x': -1, 'y': -1}, 120673: {'x': 0, 'y': -1}}), \
                                                 (138183, '下', {117459: {'x': 0, 'y': 0}, 115862: {'x': 1, 'y': 1}, 120859: {'x': 0, 'y': 1}}), \
                                                 (138936, '下', {115658: {'x': 0, 'y': 0}, 120949: {'x': 0, 'y': 1}, 117503: {'x': -1, 'y': 1}}), \
                                                 (138762, '下', {120767: {'x': 0, 'y': 0}, 117413: {'x': 0, 'y': -1}, 120768: {'x': 1, 'y': -1}}), \
                                                 (130318, '下', {114705: {'x': 0, 'y': 0}, 116359: {'x': -1, 'y': 1}, 118613: {'x': -1, 'y': 0}}), \
                                                 (130321, '下', {118614: {'x': 0, 'y': 0}, 118613: {'x': 1, 'y': -1}, 116359: {'x': 1, 'y': 0}}), \
                                                 (129784, '下', {118497: {'x': 0, 'y': 0}, 118498: {'x': -1, 'y': -1}, 114421: {'x': 0, 'y': -1}}), \
                                                 (129792, '下', {118499: {'x': 0, 'y': 0}, 118479: {'x': -1, 'y': 1}, 114673: {'x': -1, 'y': 0}}), \
                                                 (130256, '下', {118599: {'x': 0, 'y': 0}, 118600: {'x': 1, 'y': -1}, 114702: {'x': 1, 'y': 0}}), \
                                                 (129590, '下', {114665: {'x': 0, 'y': 0}, 116268: {'x': -1, 'y': -1}, 118439: {'x': 0, 'y': -1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "首"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_02(self):
        # VRoid
        model = PmxReader("D:\\MMD\\Blender\\スカート\\_報告\\ギン助\\kukurumoon_57_bone_test.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(20179, '下', {1037: {'x': 0, 'y': 0}, 988: {'x': -1, 'y': 1}, 987: {'x': -1, 'y': 0}}), \
                                                 (20785, '下', {1039: {'x': 0, 'y': 0}, 1040: {'x': 0, 'y': 1}, 990: {'x': -1, 'y': 1}}), \
                                                 (20234, '下', {1093: {'x': 0, 'y': 0}, 1069: {'x': -1, 'y': 1}, 1068: {'x': -1, 'y': 0}}), \
                                                 (20399, '下', {270: {'x': 0, 'y': 0}, 898: {'x': -1, 'y': 0}, 274: {'x': -1, 'y': -1}}), \
                                                 (20273, '下', {271: {'x': 0, 'y': 0}, 898: {'x': -1, 'y': -1}, 270: {'x': 0, 'y': -1}}), \
                                                 (19929, '下', {803: {'x': 0, 'y': 0}, 766: {'x': -1, 'y': -1}, 802: {'x': 0, 'y': -1}}), \
                                                 (19809, '下', {356: {'x': 0, 'y': 0}, 520: {'x': 1, 'y': -1}, 521: {'x': 1, 'y': 0}}), \
                                                 (20196, '下', {1056: {'x': 0, 'y': 0}, 1017: {'x': -1, 'y': 1}, 1015: {'x': -1, 'y': 0}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_03(self):
        # 垂直
        model = PmxReader("D:\\MMD\\Blender\\スカート\\_報告\\less\\son-na-kan-ji_20211113_164508.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(163, '下', {86: {'x': 0, 'y': 0}, 96: {'x': 1, 'y': 1}, 97: {'x': 0, 'y': 1}}), \
                                                 (74, '下', {37: {'x': 0, 'y': 0}, 49: {'x': 0, 'y': 1}, 38: {'x': -1, 'y': 0}}), \
                                                 (133, '下', {70: {'x': 0, 'y': 0}, 80: {'x': 1, 'y': 1}, 81: {'x': 0, 'y': 1}}), \
                                                 (148, '下', {78: {'x': 0, 'y': 0}, 89: {'x': 0, 'y': 1}, 79: {'x': -1, 'y': 0}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_04(self):
        # 垂直
        model = PmxReader("D:\\MMD\\Blender\\スカート\\test01.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(76, '下', {186: {'x': 0, 'y': 0}, 188: {'x': 0, 'y': -1}, 187: {'x': 1, 'y': 0}}), \
                                                 (347, '下', {230: {'x': 0, 'y': 0}, 231: {'x': 1, 'y': 0}, 229: {'x': 1, 'y': 1}}), \
                                                 (131, '下', {296: {'x': 0, 'y': 0}, 298: {'x': 0, 'y': -1}, 297: {'x': 1, 'y': 0}}), \
                                                 (420, '下', {376: {'x': 0, 'y': 0}, 377: {'x': 1, 'y': 0}, 375: {'x': 1, 'y': 1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_05(self):
        # 垂直ミラー
        model = PmxReader("D:\\MMD\\Blender\\スカート\\test02.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(168, '下', {173: {'x': 0, 'y': 0}, 174: {'x': 0, 'y': -1}, 172: {'x': 1, 'y': 0}}), \
                                                 (56, '下', {125: {'x': 0, 'y': 0}, 129: {'x': -1, 'y': 0}, 128: {'x': 0, 'y': -1}}), \
                                                 (341, '下', {371: {'x': 0, 'y': 0}, 369: {'x': 0, 'y': 1}, 368: {'x': -1, 'y': 0}}), \
                                                 (227, '下', {321: {'x': 0, 'y': 0}, 318: {'x': -1, 'y': 0}, 320: {'x': -1, 'y': -1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_06(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\Vネックワンピース ねこる.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(7795, '下', {4527: {'x': 0, 'y': 0}, 2074: {'x': -1, 'y': -1}, 4447: {'x': 0, 'y': -1}}), \
                                                 (7794, '下', {4527: {'x': 0, 'y': 0}, 2155: {'x': -1, 'y': 0}, 2074: {'x': -1, 'y': -1}}), \
                                                 (7972, '下', {4618: {'x': 0, 'y': 0}, 4480: {'x': 0, 'y': 1}, 2107: {'x': -1, 'y': 1}}), \
                                                 (7733, '下', {4479: {'x': 0, 'y': 0}, 2107: {'x': -1, 'y': -1}, 4480: {'x': 0, 'y': -1}}), \
                                                 (7747, '下', {4493: {'x': 0, 'y': 0}, 2120: {'x': 1, 'y': -1}, 2121: {'x': 1, 'y': 0}}), \
                                                 (7868, '下', {4565: {'x': 0, 'y': 0}, 4485: {'x': 0, 'y': -1}, 2113: {'x': 1, 'y': -1}}), \
                                                 (8074, '下', {4669: {'x': 0, 'y': 0}, 2294: {'x': 1, 'y': 0}, 2254: {'x': 1, 'y': 1}}), \
                                                 (7805, '下', {4533: {'x': 0, 'y': 0}, 2081: {'x': 1, 'y': -1}, 2160: {'x': 1, 'y': 0}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_07(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\ドット柄の秋っぽい服 緑.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(2425, '下', {648: {'x': 0, 'y': 0}, 597: {'x': -1, 'y': 1}, 575: {'x': -1, 'y': 0}}), \
                                                 (2641, '下', {648: {'x': 0, 'y': 0}, 652: {'x': 0, 'y': 1}, 597: {'x': -1, 'y': 1}}), \
                                                 (2209, '下', {648: {'x': 0, 'y': 0}, 388: {'x': 1, 'y': 1}, 652: {'x': 0, 'y': 1}}), \
                                                 (1993, '下', {648: {'x': 0, 'y': 0}, 366: {'x': 1, 'y': 0}, 388: {'x': 1, 'y': 1}}), \
                                                 (2420, '下', {577: {'x': 0, 'y': 0}, 601: {'x': -1, 'y': 1}, 579: {'x': -1, 'y': 0}}), \
                                                 (2636, '下', {577: {'x': 0, 'y': 0}, 599: {'x': 0, 'y': 1}, 601: {'x': -1, 'y': 1}}), \
                                                 (2366, '下', {627: {'x': 0, 'y': 0}, 644: {'x': -1, 'y': 1}, 659: {'x': -1, 'y': 0}}), \
                                                 (1934, '下', {418: {'x': 0, 'y': 0}, 659: {'x': 1, 'y': 0}, 644: {'x': 1, 'y': 1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_08(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\less_for_debug.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(19, '下', {9: {'x': 0, 'y': 0}, 10: {'x': -1, 'y': 0}, 63: {'x': 0, 'y': -1}}), \
                                                 (91, '下', {10: {'x': 0, 'y': 0}, 74: {'x': 0, 'y': -1}, 63: {'x': 1, 'y': -1}}), \
                                                 (87, '下', {59: {'x': 0, 'y': 0}, 74: {'x': 1, 'y': 0}, 10: {'x': 1, 'y': 1}}), \
                                                 (15, '下', {17: {'x': 0, 'y': 0}, 59: {'x': 0, 'y': -1}, 10: {'x': 1, 'y': 0}}), \
                                                 (51, '下', {93: {'x': 0, 'y': 0}, 92: {'x': 1, 'y': 0}, 33: {'x': 0, 'y': 1}}), \
                                                 (123, '下', {92: {'x': 0, 'y': 0}, 39: {'x': 0, 'y': 1}, 33: {'x': -1, 'y': 1}}), \
                                                 (54, '下', {96: {'x': 0, 'y': 0}, 25: {'x': 0, 'y': 1}, 91: {'x': -1, 'y': 0}}), \
                                                 (142, '下', {90: {'x': 0, 'y': 0}, 26: {'x': -1, 'y': 0}, 29: {'x': -1, 'y': -1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_09(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\collar.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(11234, '下', {235: {'x': 0, 'y': 0}, 238: {'x': 0, 'y': -1}, 237: {'x': 1, 'y': 0}}), \
                                                 (11350, '下', {5: {'x': 0, 'y': 0}, 3: {'x': 1, 'y': 0}, 245: {'x': 1, 'y': 1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_10(self):
        model = PmxReader("D:\\MMD\\Blender\\スカート\\_報告\\しもべ2\\高杉晋作_tailor.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(87076, '下', {11293: {'x': 0, 'y': 0}, 11290: {'x': 1, 'y': 0}, 11292: {'x': 0, 'y': 1}}), \
                                                 (88168, '下', {10725: {'x': 0, 'y': 0}, 10728: {'x': -1, 'y': 0}, 10724: {'x': -1, 'y': -1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "左肩"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])

    def test_create_vertex_map_by_index_horizonal_06(self):
        # 水平
        model = PmxReader("D:\\MMD\\Blender\\スカート\\test03.pmx", is_check=False, is_sizing=False).read_data()

        for index_idx, direction, vertex_map in [(278, '下', {161: {'x': 0, 'y': 0}, 160: {'x': 0, 'y': 1}, 164: {'x': -1, 'y': 1}}), \
                                                 (119, '下', {126: {'x': 0, 'y': 0}, 128: {'x': 1, 'y': 0}, 165: {'x': 0, 'y': 1}}), \
                                                 (155, '下', {109: {'x': 0, 'y': 0}, 111: {'x': 1, 'y': 0}, 197: {'x': 0, 'y': 1}}), \
                                                 (309, '下', {188: {'x': 0, 'y': 0}, 187: {'x': 0, 'y': 1}, 191: {'x': -1, 'y': 1}}), \
                                                 (51, '下', {76: {'x': 0, 'y': 0}, 16: {'x': 0, 'y': -1}, 77: {'x': 1, 'y': 0}}), \
                                                 (218, '下', {81: {'x': 0, 'y': 0}, 85: {'x': 1, 'y': 0}, 84: {'x': 1, 'y': 1}}), \
                                                 (30, '下', {48: {'x': 0, 'y': 0}, 49: {'x': 0, 'y': -1}, 52: {'x': 1, 'y': 0}}), \
                                                 (204, '下', {63: {'x': 0, 'y': 0}, 67: {'x': 1, 'y': 0}, 66: {'x': 1, 'y': 1}})]:
            service = PmxTailorExportService(MExportOptions("0", 10, 3, model, "", [{"direction": direction, "parent_bone_name": "下半身"}], None, False, ""))
            duplicate_vertices = {}
            for vertex_idx in model.indices[index_idx]:
                vertex = model.vertex_dict[vertex_idx]
                key = vertex.position.to_log()
                if key not in duplicate_vertices:
                    duplicate_vertices[key] = []
                if vertex.index not in duplicate_vertices[key]:
                    duplicate_vertices[key].append(vertex.index)
            
            vertex_axis_map, vertex_coordinate_map = \
                service.create_vertex_map_by_index(model, service.options.param_options[0], duplicate_vertices, {}, {}, index_idx)

            for vertex_idx, vmap in vertex_map.items():
                print("vertex_idx: %s, vmap: [%s, %s], result: [%s, %s]" % (vertex_idx, vmap['x'], vmap['y'], vertex_axis_map[vertex_idx]['x'], vertex_axis_map[vertex_idx]['y']))
                # self.assertEqual(vertex_axis_map[vertex_idx]['x'], vmap['x'])
                # self.assertEqual(vertex_axis_map[vertex_idx]['y'], vmap['y'])


if __name__ == "__main__":
    unittest.main()

