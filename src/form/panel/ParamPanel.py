# -*- coding: utf-8 -*-
#
import os
import wx
import wx.lib.newevent
import sys

from form.panel.BasePanel import BasePanel
from form.parts.FloatSliderCtrl import FloatSliderCtrl
from mmd.PmxData import RigidBody, Joint
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils import MFormUtils, MFileUtils
from utils.MLogger import MLogger # noqa

logger = MLogger(__name__)
TIMER_ID = wx.NewId()

# イベント定義
(ParentThreadEvent, EVT_SMOOTH_THREAD) = wx.lib.newevent.NewEvent()


class ParamPanel(BasePanel):
        
    def __init__(self, frame: wx.Frame, export: wx.Notebook, tab_idx: int):
        super().__init__(frame, export, tab_idx)
        self.convert_export_worker = None

        self.header_panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        self.header_sizer = wx.BoxSizer(wx.VERTICAL)

        self.description_txt = wx.StaticText(self, wx.ID_ANY, u"材質を選択して、パラメーターを調整してください。", wx.DefaultPosition, wx.DefaultSize, 0)
        self.header_sizer.Add(self.description_txt, 0, wx.ALL, 5)

        self.static_line01 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL)
        self.header_sizer.Add(self.static_line01, 0, wx.EXPAND | wx.ALL, 5)

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 材質設定クリアボタン
        self.clear_btn_ctrl = wx.Button(self.header_panel, wx.ID_ANY, u"材質設定クリア", wx.DefaultPosition, wx.DefaultSize, 0)
        self.clear_btn_ctrl.SetToolTip(u"既に入力されたデータをすべて空にします。")
        self.clear_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_clear_set)
        self.btn_sizer.Add(self.clear_btn_ctrl, 0, wx.ALL, 5)

        # 材質設定追加ボタン
        self.add_btn_ctrl = wx.Button(self.header_panel, wx.ID_ANY, u"物理設定追加", wx.DefaultPosition, wx.DefaultSize, 0)
        self.add_btn_ctrl.SetToolTip(u"物理設定フォームをパネルに追加します。")
        self.add_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_add)
        self.btn_sizer.Add(self.add_btn_ctrl, 0, wx.ALL, 5)

        self.header_sizer.Add(self.btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        self.header_panel.SetSizer(self.header_sizer)
        self.header_panel.Layout()
        self.sizer.Add(self.header_panel, 0, wx.EXPAND | wx.ALL, 5)

        # 元モデルのハッシュ
        self.org_model_digest = -1
        # 材質名リスト
        self.material_list = []
        # ボーンリスト
        self.bone_list = []
        # 物理リスト
        self.physics_list = []
        # 基本Sizer
        self.simple_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.scrolled_window = wx.ScrolledWindow(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.FULL_REPAINT_ON_RESIZE | wx.VSCROLL | wx.ALWAYS_SHOW_SB)
        self.scrolled_window.SetScrollRate(5, 5)

        self.scrolled_window.SetSizer(self.simple_sizer)
        self.scrolled_window.Layout()
        self.sizer.Add(self.scrolled_window, 1, wx.ALL | wx.EXPAND | wx.FIXED_MINSIZE, 5)
        self.fit()

    def on_add(self, event: wx.Event):
        self.physics_list.append(PhysicsParam(self.frame, self, self.scrolled_window, self.frame.advance_param_panel_ctrl.scrolled_window, len(self.physics_list)))

        # 基本
        self.simple_sizer.Add(self.physics_list[-1].simple_sizer, 0, wx.ALL | wx.EXPAND, 5)
        self.simple_sizer.Layout()

        # 詳細
        self.frame.advance_param_panel_ctrl.advance_sizer.Add(self.physics_list[-1].advance_sizer, 0, wx.ALL | wx.EXPAND, 5)
        self.frame.advance_param_panel_ctrl.advance_sizer.Layout()

        # 初期値設定
        self.physics_list[-1].on_clear(event)

        # スクロールバーの表示のためにサイズ調整
        self.sizer.Layout()
        self.frame.advance_param_panel_ctrl.sizer.Layout()

        event.Skip()

    def on_clear_set(self, event: wx.Event):
        for physics_param in self.physics_list:
            physics_param.on_clear(event)

    # フォーム無効化
    def disable(self):
        self.file_set.disable()

    # フォーム無効化
    def enable(self):
        self.file_set.enable()

    def initialize(self, event: wx.Event):
        if self.frame.file_panel_ctrl.org_model_file_ctrl.data and self.org_model_digest != self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest:
            for physics_param in self.physics_list:
                self.simple_sizer.Hide(physics_param.simple_sizer, recursive=True)
                self.frame.advance_param_panel_ctrl.advance_sizer.Hide(physics_param.advance_sizer, recursive=True)

            # ハッシュ
            self.org_model_digest = self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest
            # 物理リストクリア
            self.physics_list = []
            # プルダウン用材質名リスト
            self.material_list = []
            for material_name in self.frame.file_panel_ctrl.org_model_file_ctrl.data.materials.keys():
                self.material_list.append(material_name)
            # ボーンリストクリア
            self.bone_list = []
            for bone_name in self.frame.file_panel_ctrl.org_model_file_ctrl.data.bones.keys():
                self.bone_list.append(bone_name)
            # セットクリア
            self.on_clear_set(event)
            # 1件追加
            self.on_add(event)
        elif not self.frame.file_panel_ctrl.org_model_file_ctrl.data:
            # ハッシュ
            self.org_model_digest = None
            # 物理リストクリア
            self.physics_list = []
            # プルダウン用材質名リスト
            self.material_list = []
            # セットクリア
            self.on_clear_set(event)

    def get_param_options(self, is_show_error=False):
        params = []

        if self.frame.file_panel_ctrl.org_model_file_ctrl.data and self.org_model_digest == self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest:
            for pidx, physics_param in enumerate(self.physics_list):
                param = physics_param.get_param_options(pidx, is_show_error)
                if param:
                    params.append(param)
        
        if len(params) == 0 and not self.frame.is_vroid:
            logger.error("物理設定が1件も設定されていません。\nモデルを選択しなおした場合、物理設定は初期化されます。", decoration=MLogger.DECORATION_BOX)

        return params


class PhysicsParam():
    def __init__(self, main_frame: wx.Frame, frame: wx.Frame, simple_window: wx.Panel, advance_window: wx.Panel, param_no: int):
        self.main_frame = main_frame
        self.frame = frame
        self.simple_window = simple_window
        self.advance_window = advance_window

        # 簡易版 ------------------
        self.simple_sizer = wx.StaticBoxSizer(wx.StaticBox(self.simple_window, wx.ID_ANY, "【No.{0}】".format(param_no + 1)), orient=wx.VERTICAL)
        self.simple_param_sizer = wx.BoxSizer(wx.VERTICAL)

        self.simple_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # インポートボタン
        self.import_btn_ctrl = wx.Button(self.simple_window, wx.ID_ANY, u"インポート ...", wx.DefaultPosition, wx.DefaultSize, 0)
        self.import_btn_ctrl.SetToolTip(u"材質設定データをjsonファイルから読み込みます。\nファイル選択ダイアログが開きます。")
        self.import_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_import)
        self.simple_btn_sizer.Add(self.import_btn_ctrl, 0, wx.ALL, 5)

        # エクスポートボタン
        self.export_btn_ctrl = wx.Button(self.simple_window, wx.ID_ANY, u"エクスポート ...", wx.DefaultPosition, wx.DefaultSize, 0)
        self.export_btn_ctrl.SetToolTip(u"材質設定データをjsonファイルに出力します。\n元モデルと同じフォルダに出力します。")
        self.export_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_export)
        self.simple_btn_sizer.Add(self.export_btn_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 0)

        self.simple_material_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_material_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"物理材質 *", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_material_txt.SetToolTip(u"物理を設定する材質を選択してください。\n材質全体に物理を設定するため、裾など一部にのみ物理を設定したい場合、材質を一旦分離してください。")
        self.simple_material_txt.Wrap(-1)
        self.simple_material_sizer.Add(self.simple_material_txt, 0, wx.ALL, 5)

        self.simple_material_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.material_list)
        self.simple_material_ctrl.SetToolTip(u"物理を設定する材質を選択してください。\n材質全体に物理を設定するため、裾など一部にのみ物理を設定したい場合、材質を一旦分離してください。")
        self.simple_material_ctrl.Bind(wx.EVT_CHOICE, self.set_material_name)
        self.simple_material_sizer.Add(self.simple_material_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_material_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_header_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        self.simple_abb_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"材質略称 *", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_abb_txt.SetToolTip(u"ボーン名などに使用する材質略称を5文字以内で入力してください。")
        self.simple_abb_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_abb_txt, 0, wx.ALL, 5)

        self.simple_abb_ctrl = wx.TextCtrl(self.simple_window, id=wx.ID_ANY, size=wx.Size(70, -1))
        self.simple_abb_ctrl.SetToolTip(u"ボーン名などに使用する材質略称を5文字以内で入力してください。")
        self.simple_abb_ctrl.SetMaxLength(5)
        self.simple_header_grid_sizer.Add(self.simple_abb_ctrl, 0, wx.ALL, 5)

        self.simple_parent_bone_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"親ボーン *", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_parent_bone_txt.SetToolTip(u"材質の起点となる親ボーン")
        self.simple_parent_bone_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_parent_bone_txt, 0, wx.ALL, 5)

        self.simple_parent_bone_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.bone_list)
        self.simple_parent_bone_ctrl.SetToolTip(u"材質の起点となる親ボーン")
        self.simple_header_grid_sizer.Add(self.simple_parent_bone_ctrl, 0, wx.ALL, 5)

        self.simple_direction_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"物理方向", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_direction_txt.SetToolTip(u"物理材質の向き")
        self.simple_direction_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_direction_txt, 0, wx.ALL, 5)

        self.simple_direction_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=["下", "上", "右", "左"])
        self.simple_direction_ctrl.SetToolTip(u"物理材質の向き")
        self.simple_direction_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_header_grid_sizer.Add(self.simple_direction_ctrl, 0, wx.ALL, 5)

        self.simple_group_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"剛体グループ *", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_group_txt.SetToolTip(u"剛体のグループ。初期設定では、1と自分自身のグループのみ非衝突として設定します。")
        self.simple_group_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_group_txt, 0, wx.ALL, 5)

        self.simple_group_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
        self.simple_group_ctrl.SetToolTip(u"剛体のグループ。初期設定では、1と自分自身のグループのみ非衝突として設定します。")
        self.simple_header_grid_sizer.Add(self.simple_group_ctrl, 0, wx.ALL, 5)

        self.simple_primitive_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"プリセット", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_primitive_txt.SetToolTip(u"物理の参考値プリセット")
        self.simple_primitive_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_primitive_txt, 0, wx.ALL, 5)

        self.simple_primitive_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=["布(コットン)", "布(絹)", "布(レザー)", "布(デニム)", "髪(ロング)", "髪(ショート)"])
        self.simple_primitive_ctrl.SetToolTip(u"物理の参考値プリセット")
        self.simple_primitive_ctrl.Bind(wx.EVT_CHOICE, self.set_simple_primitive)
        self.simple_header_grid_sizer.Add(self.simple_primitive_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_header_grid_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_back_material_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_back_material_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"裏面材質", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_back_material_txt.SetToolTip(u"物理材質の裏面にあたる材質がある場合、選択してください。\n物理材質のボーン割りに応じてウェイトを割り当てます")
        self.simple_back_material_txt.Wrap(-1)
        self.simple_back_material_sizer.Add(self.simple_back_material_txt, 0, wx.ALL, 5)

        self.simple_back_material_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.material_list)
        self.simple_back_material_ctrl.SetToolTip(u"物理材質の裏面にあたる材質がある場合、選択してください。\n物理材質のボーン割りに応じてウェイトを割り当てます")
        self.simple_back_material_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_back_material_sizer.Add(self.simple_back_material_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_back_material_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_grid_sizer = wx.FlexGridSizer(0, 5, 0, 0)

        # 材質頂点類似度
        self.simple_similarity_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"検出度", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_similarity_txt.SetToolTip(u"材質内の頂点を検出する時の傾き等の類似度\n値を小さくすると傾きが違っていても検出しやすくなりますが、誤検知が増える可能性があります。")
        self.simple_similarity_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_similarity_txt, 0, wx.ALL, 5)

        self.simple_similarity_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"（0.75）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_similarity_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_similarity_label, 0, wx.ALL, 5)

        self.simple_similarity_min_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"0.5", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_similarity_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_similarity_min_label, 0, wx.ALL, 5)

        self.simple_similarity_slider = \
            FloatSliderCtrl(self.simple_window, wx.ID_ANY, 0.75, 0.4, 1, 0.01, self.simple_similarity_label, wx.DefaultPosition, (350, 30), wx.SL_HORIZONTAL)
        self.simple_grid_sizer.Add(self.simple_similarity_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_similarity_max_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_similarity_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_similarity_max_label, 0, wx.ALL, 5)

        # 物理の細かさスライダー
        self.simple_fineness_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"細かさ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_fineness_txt.SetToolTip(u"材質の物理の細かさ。ボーン・剛体・ジョイントの細かさ等に影響します。")
        self.simple_fineness_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_fineness_txt, 0, wx.ALL, 5)

        self.simple_fineness_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"（3）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_fineness_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_fineness_label, 0, wx.ALL, 5)

        self.simple_fineness_min_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"小", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_fineness_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_fineness_min_label, 0, wx.ALL, 5)

        self.simple_fineness_slider = \
            FloatSliderCtrl(self.simple_window, wx.ID_ANY, 3, 1, 10, 0.1, self.simple_fineness_label, wx.DefaultPosition, (350, 30), wx.SL_HORIZONTAL)
        self.simple_fineness_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_fineness)
        self.simple_grid_sizer.Add(self.simple_fineness_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_fineness_max_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"大", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_fineness_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_fineness_max_label, 0, wx.ALL, 5)

        # 剛体の質量スライダー
        self.simple_mass_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"質量", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_mass_txt.SetToolTip(u"材質の質量。剛体の質量・減衰等に影響します。")
        self.simple_mass_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_txt, 0, wx.ALL, 5)

        self.simple_mass_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"（0.5）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_mass_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_label, 0, wx.ALL, 5)

        self.simple_mass_min_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"軽", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_mass_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_min_label, 0, wx.ALL, 5)

        self.simple_mass_slider = \
            FloatSliderCtrl(self.simple_window, wx.ID_ANY, 1.5, 0.01, 5, 0.01, self.simple_mass_label, wx.DefaultPosition, (350, 30), wx.SL_HORIZONTAL)
        self.simple_mass_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_mass)
        self.simple_grid_sizer.Add(self.simple_mass_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_mass_max_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"重", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_mass_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_max_label, 0, wx.ALL, 5)

        # 空気抵抗スライダー
        self.simple_air_resistance_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"柔らかさ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_air_resistance_txt.SetToolTip(u"材質の柔らかさ。小さくなるほどすぐに元の形状に戻ります。（減衰が高い）\n剛体の減衰・ジョイントの強さ等に影響します。")
        self.simple_air_resistance_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_txt, 0, wx.ALL, 5)

        self.simple_air_resistance_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"（1.8）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_air_resistance_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_label, 0, wx.ALL, 5)

        self.simple_air_resistance_min_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"柔", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_air_resistance_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_min_label, 0, wx.ALL, 5)

        self.simple_air_resistance_slider = \
            FloatSliderCtrl(self.simple_window, wx.ID_ANY, 1.8, 0.01, 5, 0.01, self.simple_air_resistance_label, wx.DefaultPosition, (350, 30), wx.SL_HORIZONTAL)
        self.simple_air_resistance_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_air_resistance)
        self.simple_grid_sizer.Add(self.simple_air_resistance_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_air_resistance_max_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"硬", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_air_resistance_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_max_label, 0, wx.ALL, 5)

        # 形状維持スライダー
        self.simple_shape_maintenance_txt = wx.StaticText(self.simple_window, wx.ID_ANY, u"張り", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_shape_maintenance_txt.SetToolTip(u"材質の形状維持強度。ジョイントの強さ等に影響します。")
        self.simple_shape_maintenance_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_txt, 0, wx.ALL, 5)

        self.simple_shape_maintenance_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"（1.5）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_shape_maintenance_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_label, 0, wx.ALL, 5)

        self.simple_shape_maintenance_min_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"弱", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_shape_maintenance_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_min_label, 0, wx.ALL, 5)

        self.simple_shape_maintenance_slider = \
            FloatSliderCtrl(self.simple_window, wx.ID_ANY, 1.5, 0.01, 5, 0.01, self.simple_shape_maintenance_label, wx.DefaultPosition, (350, 30), wx.SL_HORIZONTAL)
        self.simple_shape_maintenance_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_shape_maintenance)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_shape_maintenance_max_label = wx.StaticText(self.simple_window, wx.ID_ANY, u"強", wx.DefaultPosition, wx.DefaultSize, 0)
        self.simple_shape_maintenance_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_max_label, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.simple_sizer.Add(self.simple_param_sizer, 1, wx.ALL | wx.EXPAND, 5)

        # 詳細版 ------------------
        self.advance_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "【No.{0}】".format(param_no + 1)), orient=wx.VERTICAL)
        self.advance_param_sizer = wx.BoxSizer(wx.VERTICAL)

        self.advance_material_ctrl = wx.StaticText(self.advance_window, wx.ID_ANY, u"（材質未選択）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_param_sizer.Add(self.advance_material_ctrl, 0, wx.ALL, 5)

        self.advance_sizer.Add(self.advance_param_sizer, 1, wx.ALL | wx.EXPAND, 0)

        # ボーン密度ブロック -------------------------------
        self.advance_bone_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "ボーン密度"), orient=wx.VERTICAL)
        self.advance_bone_grid_sizer = wx.FlexGridSizer(0, 8, 0, 0)

        # 縦密度
        self.vertical_bone_density_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"縦密度", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_bone_density_txt.SetToolTip(u"ボーンの縦方向のメッシュに対する密度")
        self.vertical_bone_density_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.vertical_bone_density_txt, 0, wx.ALL, 5)

        self.vertical_bone_density_spin = wx.SpinCtrl(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=20, initial=1)
        self.vertical_bone_density_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.vertical_bone_density_spin, 0, wx.ALL, 5)

        # 横密度
        self.horizonal_bone_density_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"横密度", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_bone_density_txt.SetToolTip(u"ボーンの横方向のメッシュに対する密度")
        self.horizonal_bone_density_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_density_txt, 0, wx.ALL, 5)

        self.horizonal_bone_density_spin = wx.SpinCtrl(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="2", min=1, max=20, initial=2)
        self.horizonal_bone_density_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_density_spin, 0, wx.ALL, 5)

        # 間引きオプション
        self.bone_thinning_out_check = wx.CheckBox(self.advance_window, wx.ID_ANY, "間引き")
        self.bone_thinning_out_check.SetToolTip("ボーン密度が均一になるよう間引きするか否か")
        self.advance_bone_grid_sizer.Add(self.bone_thinning_out_check, 0, wx.ALL, 5)

        # 物理タイプ
        self.physics_type_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"物理タイプ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.physics_type_txt.SetToolTip(u"物理タイプ")
        self.physics_type_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.physics_type_txt, 0, wx.ALL, 5)

        self.physics_type_ctrl = wx.Choice(self.advance_window, id=wx.ID_ANY, choices=['布', '髪'])
        self.physics_type_ctrl.SetToolTip(u"物理タイプ")
        self.physics_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.physics_type_ctrl, 0, wx.ALL, 5)

        self.advance_bone_sizer.Add(self.advance_bone_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_bone_sizer, 0, wx.ALL, 5)

        # 末端剛体ブロック -------------------------------
        self.advance_rigidbody_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "末端剛体"), orient=wx.VERTICAL)
        self.advance_rigidbody_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 質量
        self.rigidbody_mass_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"質量", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_mass_txt.SetToolTip(u"末端剛体の質量")
        self.rigidbody_mass_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_mass_txt, 0, wx.ALL, 5)

        self.rigidbody_mass_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0.5", min=0.000001, max=1000, initial=0.5, inc=0.01)
        self.rigidbody_mass_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_mass_spin, 0, wx.ALL, 5)

        # 移動減衰
        self.rigidbody_linear_damping_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動減衰", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_linear_damping_txt.SetToolTip(u"末端剛体の移動減衰")
        self.rigidbody_linear_damping_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_linear_damping_txt, 0, wx.ALL, 5)

        self.rigidbody_linear_damping_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0.999", min=0.000001, max=0.9999999, initial=0.999, inc=0.01)
        self.rigidbody_linear_damping_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_linear_damping_spin, 0, wx.ALL, 5)

        # 回転減衰
        self.rigidbody_angular_damping_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転減衰", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_angular_damping_txt.SetToolTip(u"末端剛体の回転減衰")
        self.rigidbody_angular_damping_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_angular_damping_txt, 0, wx.ALL, 5)

        self.rigidbody_angular_damping_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0.999", min=0.000001, max=0.9999999, initial=0.999, inc=0.01)
        self.rigidbody_angular_damping_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_angular_damping_spin, 0, wx.ALL, 5)

        # 反発力
        self.rigidbody_restitution_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"反発力", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_restitution_txt.SetToolTip(u"末端剛体の反発力")
        self.rigidbody_restitution_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_restitution_txt, 0, wx.ALL, 5)

        self.rigidbody_restitution_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=10, initial=0, inc=0.01)
        self.rigidbody_restitution_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_restitution_spin, 0, wx.ALL, 5)

        # 摩擦力
        self.rigidbody_friction_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"摩擦力", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_friction_txt.SetToolTip(u"末端剛体の摩擦力")
        self.rigidbody_friction_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_friction_txt, 0, wx.ALL, 5)

        self.rigidbody_friction_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0.2", min=0, max=1000, initial=0.2, inc=0.01)
        self.rigidbody_friction_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_friction_spin, 0, wx.ALL, 5)

        # 係数
        self.rigidbody_coefficient_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"係数", wx.DefaultPosition, wx.DefaultSize, 0)
        self.rigidbody_coefficient_txt.SetToolTip(u"末端剛体から上の剛体にかけての加算係数")
        self.rigidbody_coefficient_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_coefficient_txt, 0, wx.ALL, 5)

        self.rigidbody_coefficient_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1.2", min=1, max=10, initial=1.2, inc=0.1)
        self.rigidbody_coefficient_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_coefficient_spin, 0, wx.ALL, 5)

        # 剛体形状
        self.advance_rigidbody_shape_type_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"剛体形状", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_rigidbody_shape_type_txt.SetToolTip(u"剛体の形状")
        self.advance_rigidbody_shape_type_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_shape_type_txt, 0, wx.ALL, 5)

        self.advance_rigidbody_shape_type_ctrl = wx.Choice(self.advance_window, id=wx.ID_ANY, choices=['球', '箱', 'カプセル'])
        self.advance_rigidbody_shape_type_ctrl.SetToolTip(u"剛体の形状")
        self.advance_rigidbody_shape_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_shape_type_ctrl, 0, wx.ALL, 5)

        self.advance_rigidbody_sizer.Add(self.advance_rigidbody_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_rigidbody_sizer, 0, wx.ALL, 5)

        # 縦ジョイントブロック -------------------------------
        self.advance_vertical_joint_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "縦ジョイント"), orient=wx.VERTICAL)

        self.advance_vertical_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_vertical_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, "有効")
        self.advance_vertical_joint_valid_check.SetToolTip("縦ジョイントを有効にするか否か")
        self.advance_vertical_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_vertical_joint)
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_vertical_joint_coefficient_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"制限係数", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_vertical_joint_coefficient_txt.SetToolTip(u"根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        self.advance_vertical_joint_coefficient_txt.Wrap(-1)
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_vertical_joint_coefficient_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1.2", min=1.2, max=10, initial=1, inc=0.1)
        self.advance_vertical_joint_coefficient_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_vertical_joint_sizer.Add(self.advance_vertical_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_vertical_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.vertical_joint_mov_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_x_min_txt.SetToolTip(u"末端縦ジョイントの移動X(最小)")
        self.vertical_joint_mov_x_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_mov_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.vertical_joint_mov_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_y_min_txt.SetToolTip(u"末端縦ジョイントの移動Y(最小)")
        self.vertical_joint_mov_y_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_mov_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.vertical_joint_mov_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_z_min_txt.SetToolTip(u"末端縦ジョイントの移動Z(最小)")
        self.vertical_joint_mov_z_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_mov_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.vertical_joint_mov_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_x_max_txt.SetToolTip(u"末端縦ジョイントの移動X(最大)")
        self.vertical_joint_mov_x_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_mov_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.vertical_joint_mov_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_y_max_txt.SetToolTip(u"末端縦ジョイントの移動Y(最大)")
        self.vertical_joint_mov_y_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_mov_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.vertical_joint_mov_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_mov_z_max_txt.SetToolTip(u"末端縦ジョイントの移動Z(最大)")
        self.vertical_joint_mov_z_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_mov_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.vertical_joint_rot_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_x_min_txt.SetToolTip(u"末端縦ジョイントの回転X(最小)")
        self.vertical_joint_rot_x_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_rot_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.vertical_joint_rot_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_y_min_txt.SetToolTip(u"末端縦ジョイントの回転Y(最小)")
        self.vertical_joint_rot_y_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_rot_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.vertical_joint_rot_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_z_min_txt.SetToolTip(u"末端縦ジョイントの回転Z(最小)")
        self.vertical_joint_rot_z_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.vertical_joint_rot_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.vertical_joint_rot_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_x_max_txt.SetToolTip(u"末端縦ジョイントの回転X(最大)")
        self.vertical_joint_rot_x_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_rot_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.vertical_joint_rot_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_y_max_txt.SetToolTip(u"末端縦ジョイントの回転Y(最大)")
        self.vertical_joint_rot_y_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_rot_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.vertical_joint_rot_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_rot_z_max_txt.SetToolTip(u"末端縦ジョイントの回転Z(最大)")
        self.vertical_joint_rot_z_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_rot_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.vertical_joint_spring_mov_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_mov_x_txt.SetToolTip(u"末端縦ジョイントのばね(移動X)")
        self.vertical_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_mov_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.vertical_joint_spring_mov_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_mov_y_txt.SetToolTip(u"末端縦ジョイントのばね(移動Y)")
        self.vertical_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_mov_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.vertical_joint_spring_mov_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_mov_z_txt.SetToolTip(u"末端縦ジョイントのばね(移動Z)")
        self.vertical_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_mov_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.vertical_joint_spring_rot_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_rot_x_txt.SetToolTip(u"末端縦ジョイントのばね(回転X)")
        self.vertical_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_rot_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.vertical_joint_spring_rot_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_rot_y_txt.SetToolTip(u"末端縦ジョイントのばね(回転Y)")
        self.vertical_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_rot_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.vertical_joint_spring_rot_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.vertical_joint_spring_rot_z_txt.SetToolTip(u"末端縦ジョイントのばね(回転Z)")
        self.vertical_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.vertical_joint_spring_rot_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_vertical_joint_sizer.Add(self.advance_vertical_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_vertical_joint_sizer, 0, wx.ALL, 5)

        # 横ジョイントブロック -------------------------------
        self.advance_horizonal_joint_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "横ジョイント"), orient=wx.VERTICAL)

        self.advance_horizonal_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_horizonal_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, "有効")
        self.advance_horizonal_joint_valid_check.SetToolTip("横ジョイントを有効にするか否か")
        self.advance_horizonal_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_horizonal_joint)
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_horizonal_joint_coefficient_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"制限係数", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_horizonal_joint_coefficient_txt.SetToolTip(u"根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        self.advance_horizonal_joint_coefficient_txt.Wrap(-1)
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_horizonal_joint_coefficient_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="2.3", min=1, max=10, initial=2.3, inc=0.1)
        self.advance_horizonal_joint_coefficient_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_horizonal_joint_sizer.Add(self.advance_horizonal_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_horizonal_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.horizonal_joint_mov_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_x_min_txt.SetToolTip(u"末端横ジョイントの移動X(最小)")
        self.horizonal_joint_mov_x_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_mov_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.horizonal_joint_mov_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_y_min_txt.SetToolTip(u"末端横ジョイントの移動Y(最小)")
        self.horizonal_joint_mov_y_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_mov_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.horizonal_joint_mov_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_z_min_txt.SetToolTip(u"末端横ジョイントの移動Z(最小)")
        self.horizonal_joint_mov_z_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_mov_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.horizonal_joint_mov_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_x_max_txt.SetToolTip(u"末端横ジョイントの移動X(最大)")
        self.horizonal_joint_mov_x_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_mov_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.horizonal_joint_mov_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_y_max_txt.SetToolTip(u"末端横ジョイントの移動Y(最大)")
        self.horizonal_joint_mov_y_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_mov_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.horizonal_joint_mov_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_mov_z_max_txt.SetToolTip(u"末端横ジョイントの移動Z(最大)")
        self.horizonal_joint_mov_z_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_mov_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.horizonal_joint_rot_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_x_min_txt.SetToolTip(u"末端横ジョイントの回転X(最小)")
        self.horizonal_joint_rot_x_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_rot_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.horizonal_joint_rot_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_y_min_txt.SetToolTip(u"末端横ジョイントの回転Y(最小)")
        self.horizonal_joint_rot_y_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_rot_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.horizonal_joint_rot_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_z_min_txt.SetToolTip(u"末端横ジョイントの回転Z(最小)")
        self.horizonal_joint_rot_z_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.horizonal_joint_rot_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.horizonal_joint_rot_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_x_max_txt.SetToolTip(u"末端横ジョイントの回転X(最大)")
        self.horizonal_joint_rot_x_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_rot_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.horizonal_joint_rot_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_y_max_txt.SetToolTip(u"末端横ジョイントの回転Y(最大)")
        self.horizonal_joint_rot_y_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_rot_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.horizonal_joint_rot_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_rot_z_max_txt.SetToolTip(u"末端横ジョイントの回転Z(最大)")
        self.horizonal_joint_rot_z_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_rot_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.horizonal_joint_spring_mov_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_mov_x_txt.SetToolTip(u"末端横ジョイントのばね(移動X)")
        self.horizonal_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_mov_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.horizonal_joint_spring_mov_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_mov_y_txt.SetToolTip(u"末端横ジョイントのばね(移動Y)")
        self.horizonal_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_mov_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.horizonal_joint_spring_mov_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_mov_z_txt.SetToolTip(u"末端横ジョイントのばね(移動Z)")
        self.horizonal_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_mov_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.horizonal_joint_spring_rot_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_rot_x_txt.SetToolTip(u"末端横ジョイントのばね(回転X)")
        self.horizonal_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_rot_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.horizonal_joint_spring_rot_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_rot_y_txt.SetToolTip(u"末端横ジョイントのばね(回転Y)")
        self.horizonal_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_rot_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.horizonal_joint_spring_rot_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.horizonal_joint_spring_rot_z_txt.SetToolTip(u"末端横ジョイントのばね(回転Z)")
        self.horizonal_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.horizonal_joint_spring_rot_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_horizonal_joint_sizer.Add(self.advance_horizonal_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_horizonal_joint_sizer, 0, wx.ALL, 5)

        # 斜めジョイントブロック -------------------------------
        self.advance_diagonal_joint_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "斜めジョイント"), orient=wx.VERTICAL)

        self.advance_diagonal_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_diagonal_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, "有効")
        self.advance_diagonal_joint_valid_check.SetToolTip("斜めジョイントを有効にするか否か")
        self.advance_diagonal_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_diagonal_joint)
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_diagonal_joint_coefficient_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"制限係数", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_diagonal_joint_coefficient_txt.SetToolTip(u"根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        self.advance_diagonal_joint_coefficient_txt.Wrap(-1)
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_diagonal_joint_coefficient_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=10, initial=1, inc=0.1)
        self.advance_diagonal_joint_coefficient_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_diagonal_joint_sizer.Add(self.advance_diagonal_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_diagonal_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.diagonal_joint_mov_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_x_min_txt.SetToolTip(u"末端斜めジョイントの移動X(最小)")
        self.diagonal_joint_mov_x_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_mov_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.diagonal_joint_mov_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_y_min_txt.SetToolTip(u"末端斜めジョイントの移動Y(最小)")
        self.diagonal_joint_mov_y_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_mov_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.diagonal_joint_mov_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_z_min_txt.SetToolTip(u"末端斜めジョイントの移動Z(最小)")
        self.diagonal_joint_mov_z_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_mov_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.diagonal_joint_mov_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_x_max_txt.SetToolTip(u"末端斜めジョイントの移動X(最大)")
        self.diagonal_joint_mov_x_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_mov_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.diagonal_joint_mov_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_y_max_txt.SetToolTip(u"末端斜めジョイントの移動Y(最大)")
        self.diagonal_joint_mov_y_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_mov_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.diagonal_joint_mov_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_mov_z_max_txt.SetToolTip(u"末端斜めジョイントの移動Z(最大)")
        self.diagonal_joint_mov_z_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_mov_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.diagonal_joint_rot_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_x_min_txt.SetToolTip(u"末端斜めジョイントの回転X(最小)")
        self.diagonal_joint_rot_x_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_rot_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.diagonal_joint_rot_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_y_min_txt.SetToolTip(u"末端斜めジョイントの回転Y(最小)")
        self.diagonal_joint_rot_y_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_rot_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.diagonal_joint_rot_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_z_min_txt.SetToolTip(u"末端斜めジョイントの回転Z(最小)")
        self.diagonal_joint_rot_z_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.diagonal_joint_rot_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.diagonal_joint_rot_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_x_max_txt.SetToolTip(u"末端斜めジョイントの回転X(最大)")
        self.diagonal_joint_rot_x_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_rot_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.diagonal_joint_rot_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_y_max_txt.SetToolTip(u"末端斜めジョイントの回転Y(最大)")
        self.diagonal_joint_rot_y_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_rot_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.diagonal_joint_rot_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_rot_z_max_txt.SetToolTip(u"末端斜めジョイントの回転Z(最大)")
        self.diagonal_joint_rot_z_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_rot_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.diagonal_joint_spring_mov_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_mov_x_txt.SetToolTip(u"末端斜めジョイントのばね(移動X)")
        self.diagonal_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_mov_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.diagonal_joint_spring_mov_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_mov_y_txt.SetToolTip(u"末端斜めジョイントのばね(移動Y)")
        self.diagonal_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_mov_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.diagonal_joint_spring_mov_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_mov_z_txt.SetToolTip(u"末端斜めジョイントのばね(移動Z)")
        self.diagonal_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_mov_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.diagonal_joint_spring_rot_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_rot_x_txt.SetToolTip(u"末端斜めジョイントのばね(回転X)")
        self.diagonal_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_rot_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.diagonal_joint_spring_rot_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_rot_y_txt.SetToolTip(u"末端斜めジョイントのばね(回転Y)")
        self.diagonal_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_rot_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.diagonal_joint_spring_rot_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.diagonal_joint_spring_rot_z_txt.SetToolTip(u"末端斜めジョイントのばね(回転Z)")
        self.diagonal_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.diagonal_joint_spring_rot_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_diagonal_joint_sizer.Add(self.advance_diagonal_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_diagonal_joint_sizer, 0, wx.ALL, 5)

        # 逆ジョイントブロック -------------------------------
        self.advance_reverse_joint_sizer = wx.StaticBoxSizer(wx.StaticBox(self.advance_window, wx.ID_ANY, "逆ジョイント"), orient=wx.VERTICAL)

        self.advance_reverse_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_reverse_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, "有効")
        self.advance_reverse_joint_valid_check.SetToolTip("逆ジョイントを有効にするか否か")
        self.advance_reverse_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_reverse_joint)
        self.advance_reverse_joint_head_sizer.Add(self.advance_reverse_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_reverse_joint_coefficient_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"制限係数", wx.DefaultPosition, wx.DefaultSize, 0)
        self.advance_reverse_joint_coefficient_txt.SetToolTip(u"根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        self.advance_reverse_joint_coefficient_txt.Wrap(-1)
        self.advance_reverse_joint_head_sizer.Add(self.advance_reverse_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_reverse_joint_coefficient_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=10, initial=1, inc=0.1)
        self.advance_reverse_joint_coefficient_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_head_sizer.Add(self.advance_reverse_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_reverse_joint_sizer.Add(self.advance_reverse_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_reverse_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.reverse_joint_mov_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_x_min_txt.SetToolTip(u"末端逆ジョイントの移動X(最小)")
        self.reverse_joint_mov_x_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_mov_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.reverse_joint_mov_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_y_min_txt.SetToolTip(u"末端逆ジョイントの移動Y(最小)")
        self.reverse_joint_mov_y_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_mov_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.reverse_joint_mov_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_z_min_txt.SetToolTip(u"末端逆ジョイントの移動Z(最小)")
        self.reverse_joint_mov_z_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_mov_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.reverse_joint_mov_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_x_max_txt.SetToolTip(u"末端逆ジョイントの移動X(最大)")
        self.reverse_joint_mov_x_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_mov_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.reverse_joint_mov_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_y_max_txt.SetToolTip(u"末端逆ジョイントの移動Y(最大)")
        self.reverse_joint_mov_y_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_mov_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.reverse_joint_mov_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"移動Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_mov_z_max_txt.SetToolTip(u"末端逆ジョイントの移動Z(最大)")
        self.reverse_joint_mov_z_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_mov_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_mov_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.reverse_joint_rot_x_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_x_min_txt.SetToolTip(u"末端逆ジョイントの回転X(最小)")
        self.reverse_joint_rot_x_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_x_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_rot_x_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.reverse_joint_rot_y_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_y_min_txt.SetToolTip(u"末端逆ジョイントの回転Y(最小)")
        self.reverse_joint_rot_y_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_y_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_rot_y_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.reverse_joint_rot_z_min_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最小)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_z_min_txt.SetToolTip(u"末端逆ジョイントの回転Z(最小)")
        self.reverse_joint_rot_z_min_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_z_min_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=0, initial=0, inc=0.1)
        self.reverse_joint_rot_z_min_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.reverse_joint_rot_x_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転X(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_x_max_txt.SetToolTip(u"末端逆ジョイントの回転X(最大)")
        self.reverse_joint_rot_x_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_x_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_rot_x_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.reverse_joint_rot_y_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Y(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_y_max_txt.SetToolTip(u"末端逆ジョイントの回転Y(最大)")
        self.reverse_joint_rot_y_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_y_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_rot_y_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.reverse_joint_rot_z_max_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"回転Z(最大)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_rot_z_max_txt.SetToolTip(u"末端逆ジョイントの回転Z(最大)")
        self.reverse_joint_rot_z_max_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.reverse_joint_rot_z_max_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_rot_z_max_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.reverse_joint_spring_mov_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_mov_x_txt.SetToolTip(u"末端逆ジョイントのばね(移動X)")
        self.reverse_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_mov_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_mov_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.reverse_joint_spring_mov_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_mov_y_txt.SetToolTip(u"末端逆ジョイントのばね(移動Y)")
        self.reverse_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_mov_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_mov_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.reverse_joint_spring_mov_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(移動Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_mov_z_txt.SetToolTip(u"末端逆ジョイントのばね(移動Z)")
        self.reverse_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_mov_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_mov_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.reverse_joint_spring_rot_x_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転X)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_rot_x_txt.SetToolTip(u"末端逆ジョイントのばね(回転X)")
        self.reverse_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_rot_x_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_rot_x_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.reverse_joint_spring_rot_y_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Y)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_rot_y_txt.SetToolTip(u"末端逆ジョイントのばね(回転Y)")
        self.reverse_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_rot_y_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_rot_y_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.reverse_joint_spring_rot_z_txt = wx.StaticText(self.advance_window, wx.ID_ANY, u"ばね(回転Z)", wx.DefaultPosition, wx.DefaultSize, 0)
        self.reverse_joint_spring_rot_z_txt.SetToolTip(u"末端逆ジョイントのばね(回転Z)")
        self.reverse_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.reverse_joint_spring_rot_z_spin = wx.SpinCtrlDouble(self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1)
        self.reverse_joint_spring_rot_z_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.main_frame.on_wheel_spin_ctrl(event, 0.1))
        self.advance_reverse_joint_grid_sizer.Add(self.reverse_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_reverse_joint_sizer.Add(self.advance_reverse_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_reverse_joint_sizer, 0, wx.ALL, 5)

    def get_param_options(self, pidx: int, is_show_error):
        params = {}

        if self.simple_material_ctrl.GetStringSelection() and self.simple_parent_bone_ctrl.GetStringSelection() and self.simple_group_ctrl.GetStringSelection() \
           and self.simple_abb_ctrl.GetValue():
            if not self.main_frame.is_vroid and self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.material_indices[self.simple_material_ctrl.GetStringSelection()] == 0:
                logger.error("頂点のない材質が指定されています。", decoration=MLogger.DECORATION_BOX)
                return params

            if self.simple_material_ctrl.GetStringSelection() == self.simple_back_material_ctrl.GetStringSelection():
                logger.error("物理材質と同じ材質が裏面に指定されています。", decoration=MLogger.DECORATION_BOX)
                return params

            # 簡易版オプションデータ -------------
            params["material_name"] = self.simple_material_ctrl.GetStringSelection()
            params["back_material_name"] = self.simple_back_material_ctrl.GetStringSelection()
            params["parent_bone_name"] = self.simple_parent_bone_ctrl.GetStringSelection()
            params["abb_name"] = self.simple_abb_ctrl.GetValue()
            params["direction"] = self.simple_direction_ctrl.GetStringSelection()
            params["similarity"] = self.simple_similarity_slider.GetValue()
            params["fineness"] = self.simple_fineness_slider.GetValue()
            params["mass"] = self.simple_mass_slider.GetValue()
            params["air_resistance"] = self.simple_air_resistance_slider.GetValue()
            params["shape_maintenance"] = self.simple_shape_maintenance_slider.GetValue()

            # 詳細版オプションデータ -------------
            params["vertical_bone_density"] = int(self.vertical_bone_density_spin.GetValue())
            params["horizonal_bone_density"] = int(self.horizonal_bone_density_spin.GetValue())
            params["bone_thinning_out"] = self.bone_thinning_out_check.GetValue()
            params["physics_type"] = self.physics_type_ctrl.GetStringSelection()
            
            # 自身を非衝突対象
            no_collision_group = 0
            for nc in range(16):
                if nc not in [int(self.simple_group_ctrl.GetStringSelection()) - 1]:
                    no_collision_group |= 1 << nc

            params["rigidbody"] = RigidBody("", "", self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.bones[self.simple_parent_bone_ctrl.GetStringSelection()].index, \
                                            int(self.simple_group_ctrl.GetStringSelection()) - 1, no_collision_group, self.advance_rigidbody_shape_type_ctrl.GetSelection(), MVector3D(), MVector3D(), \
                                            MVector3D(), self.rigidbody_mass_spin.GetValue(), self.rigidbody_linear_damping_spin.GetValue(), self.rigidbody_angular_damping_spin.GetValue(), \
                                            self.rigidbody_restitution_spin.GetValue(), self.rigidbody_friction_spin.GetValue(), 0)
            params["rigidbody_coefficient"] = self.rigidbody_coefficient_spin.GetValue()

            params["vertical_joint"] = None
            if self.advance_vertical_joint_valid_check.GetValue():
                params["vertical_joint"] = \
                    Joint("", "", -1, -1, -1, MVector3D(), MVector3D(), \
                          MVector3D(self.vertical_joint_mov_x_min_spin.GetValue(), self.vertical_joint_mov_y_min_spin.GetValue(), self.vertical_joint_mov_z_min_spin.GetValue()), \
                          MVector3D(self.vertical_joint_mov_x_max_spin.GetValue(), self.vertical_joint_mov_y_max_spin.GetValue(), self.vertical_joint_mov_z_max_spin.GetValue()), \
                          MVector3D(self.vertical_joint_rot_x_min_spin.GetValue(), self.vertical_joint_rot_y_min_spin.GetValue(), self.vertical_joint_rot_z_min_spin.GetValue()), \
                          MVector3D(self.vertical_joint_rot_x_max_spin.GetValue(), self.vertical_joint_rot_y_max_spin.GetValue(), self.vertical_joint_rot_z_max_spin.GetValue()), \
                          MVector3D(self.vertical_joint_spring_mov_x_spin.GetValue(), self.vertical_joint_spring_mov_y_spin.GetValue(), self.vertical_joint_spring_mov_z_spin.GetValue()), \
                          MVector3D(self.vertical_joint_spring_rot_x_spin.GetValue(), self.vertical_joint_spring_rot_y_spin.GetValue(), self.vertical_joint_spring_rot_z_spin.GetValue())
                          )
            params['vertical_joint_coefficient'] = self.advance_vertical_joint_coefficient_spin.GetValue()

            params["horizonal_joint"] = None
            if self.advance_horizonal_joint_valid_check.GetValue():
                params["horizonal_joint"] = \
                    Joint("", "", -1, -1, -1, MVector3D(), MVector3D(), \
                          MVector3D(self.horizonal_joint_mov_x_min_spin.GetValue(), self.horizonal_joint_mov_y_min_spin.GetValue(), self.horizonal_joint_mov_z_min_spin.GetValue()), \
                          MVector3D(self.horizonal_joint_mov_x_max_spin.GetValue(), self.horizonal_joint_mov_y_max_spin.GetValue(), self.horizonal_joint_mov_z_max_spin.GetValue()), \
                          MVector3D(self.horizonal_joint_rot_x_min_spin.GetValue(), self.horizonal_joint_rot_y_min_spin.GetValue(), self.horizonal_joint_rot_z_min_spin.GetValue()), \
                          MVector3D(self.horizonal_joint_rot_x_max_spin.GetValue(), self.horizonal_joint_rot_y_max_spin.GetValue(), self.horizonal_joint_rot_z_max_spin.GetValue()), \
                          MVector3D(self.horizonal_joint_spring_mov_x_spin.GetValue(), self.horizonal_joint_spring_mov_y_spin.GetValue(), self.horizonal_joint_spring_mov_z_spin.GetValue()), \
                          MVector3D(self.horizonal_joint_spring_rot_x_spin.GetValue(), self.horizonal_joint_spring_rot_y_spin.GetValue(), self.horizonal_joint_spring_rot_z_spin.GetValue())
                          )
            params['horizonal_joint_coefficient'] = self.advance_horizonal_joint_coefficient_spin.GetValue()

            params["diagonal_joint"] = None
            if self.advance_diagonal_joint_valid_check.GetValue():
                params["diagonal_joint"] = \
                    Joint("", "", -1, -1, -1, MVector3D(), MVector3D(), \
                          MVector3D(self.diagonal_joint_mov_x_min_spin.GetValue(), self.diagonal_joint_mov_y_min_spin.GetValue(), self.diagonal_joint_mov_z_min_spin.GetValue()), \
                          MVector3D(self.diagonal_joint_mov_x_max_spin.GetValue(), self.diagonal_joint_mov_y_max_spin.GetValue(), self.diagonal_joint_mov_z_max_spin.GetValue()), \
                          MVector3D(self.diagonal_joint_rot_x_min_spin.GetValue(), self.diagonal_joint_rot_y_min_spin.GetValue(), self.diagonal_joint_rot_z_min_spin.GetValue()), \
                          MVector3D(self.diagonal_joint_rot_x_max_spin.GetValue(), self.diagonal_joint_rot_y_max_spin.GetValue(), self.diagonal_joint_rot_z_max_spin.GetValue()), \
                          MVector3D(self.diagonal_joint_spring_mov_x_spin.GetValue(), self.diagonal_joint_spring_mov_y_spin.GetValue(), self.diagonal_joint_spring_mov_z_spin.GetValue()), \
                          MVector3D(self.diagonal_joint_spring_rot_x_spin.GetValue(), self.diagonal_joint_spring_rot_y_spin.GetValue(), self.diagonal_joint_spring_rot_z_spin.GetValue())
                          )
            params['diagonal_joint_coefficient'] = self.advance_diagonal_joint_coefficient_spin.GetValue()

            params["reverse_joint"] = None
            if self.advance_reverse_joint_valid_check.GetValue():
                params["reverse_joint"] = \
                    Joint("", "", -1, -1, -1, MVector3D(), MVector3D(), \
                          MVector3D(self.reverse_joint_mov_x_min_spin.GetValue(), self.reverse_joint_mov_y_min_spin.GetValue(), self.reverse_joint_mov_z_min_spin.GetValue()), \
                          MVector3D(self.reverse_joint_mov_x_max_spin.GetValue(), self.reverse_joint_mov_y_max_spin.GetValue(), self.reverse_joint_mov_z_max_spin.GetValue()), \
                          MVector3D(self.reverse_joint_rot_x_min_spin.GetValue(), self.reverse_joint_rot_y_min_spin.GetValue(), self.reverse_joint_rot_z_min_spin.GetValue()), \
                          MVector3D(self.reverse_joint_rot_x_max_spin.GetValue(), self.reverse_joint_rot_y_max_spin.GetValue(), self.reverse_joint_rot_z_max_spin.GetValue()), \
                          MVector3D(self.reverse_joint_spring_mov_x_spin.GetValue(), self.reverse_joint_spring_mov_y_spin.GetValue(), self.reverse_joint_spring_mov_z_spin.GetValue()), \
                          MVector3D(self.reverse_joint_spring_rot_x_spin.GetValue(), self.reverse_joint_spring_rot_y_spin.GetValue(), self.reverse_joint_spring_rot_z_spin.GetValue())
                          )
            params['reverse_joint_coefficient'] = self.advance_reverse_joint_coefficient_spin.GetValue()
        elif not self.main_frame.is_vroid:
            if is_show_error:
                empty_param_list = []
                if not self.simple_material_ctrl.GetStringSelection():
                    empty_param_list.append("材質名")
                if not self.simple_parent_bone_ctrl.GetStringSelection():
                    empty_param_list.append("親ボーン名")
                if not self.simple_group_ctrl.GetStringSelection():
                    empty_param_list.append("剛体グループ")
                if not self.simple_abb_ctrl.GetValue():
                    empty_param_list.append("材質略称")

                logger.error(f"No.{pidx + 1}の{'・'.join(empty_param_list)}に値が設定されていません。", decoration=MLogger.DECORATION_BOX)

        return params

    def on_import(self, event: wx.Event):
        wx.MessageBox("未実装")

    def on_export(self, event: wx.Event):
        wx.MessageBox("未実装")

    def on_vertical_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_vertical_joint_coefficient_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.vertical_joint_mov_x_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_mov_x_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_mov_y_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_mov_y_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_mov_z_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_mov_z_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_x_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_x_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_y_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_y_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_z_min_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_rot_z_max_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_mov_x_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_mov_y_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_mov_z_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_rot_x_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_rot_y_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
        self.vertical_joint_spring_rot_z_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
    
    def on_horizonal_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_horizonal_joint_coefficient_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.horizonal_joint_mov_x_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_mov_x_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_mov_y_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_mov_y_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_mov_z_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_mov_z_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_x_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_x_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_y_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_y_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_z_min_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_rot_z_max_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_mov_x_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_mov_y_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_mov_z_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_rot_x_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_rot_y_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.horizonal_joint_spring_rot_z_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        
    def on_diagonal_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_diagonal_joint_coefficient_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.diagonal_joint_mov_x_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_mov_x_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_mov_y_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_mov_y_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_mov_z_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_mov_z_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_x_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_x_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_y_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_y_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_z_min_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_rot_z_max_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_mov_x_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_mov_y_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_mov_z_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_rot_x_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_rot_y_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
        self.diagonal_joint_spring_rot_z_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
    
    def on_reverse_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_reverse_joint_coefficient_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_x_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_x_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_y_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_y_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_z_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_mov_z_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_x_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_x_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_y_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_y_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_z_min_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_rot_z_max_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_mov_x_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_mov_y_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_mov_z_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_rot_x_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_rot_y_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
        self.reverse_joint_spring_rot_z_spin.Enable(self.advance_reverse_joint_valid_check.GetValue())
    
    def set_material_name(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_material_ctrl.SetLabelText(self.simple_material_ctrl.GetStringSelection())
        self.simple_abb_ctrl.SetValue(self.simple_material_ctrl.GetStringSelection()[:5])
    
    def set_fineness(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.vertical_bone_density_spin.SetValue(int(self.simple_fineness_slider.GetValue() // 1.3))
        self.horizonal_bone_density_spin.SetValue(int(self.simple_fineness_slider.GetValue() // 1.5))
        # self.bone_thinning_out_check.SetValue((self.simple_fineness_slider.GetValue() // 1.2) % 2 == 0)

    def set_mass(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.rigidbody_mass_spin.SetValue(self.simple_mass_slider.GetValue())
        self.set_air_resistance(event)
    
    def set_simple_primitive(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        if '布' in self.simple_primitive_ctrl.GetStringSelection():
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection('箱')
        elif '髪' in self.simple_primitive_ctrl.GetStringSelection():
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection('カプセル')
        self.physics_type_ctrl.SetStringSelection(self.simple_primitive_ctrl.GetStringSelection()[0])

    def set_air_resistance(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        # 質量に応じて減衰を設定
        self.rigidbody_linear_damping_spin.SetValue(
            max(0, min(0.9999, 1 - (((1 - self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax()) \
                * (self.simple_mass_slider.GetValue() / self.simple_mass_slider.GetMax())) * 0.8))))
        self.rigidbody_angular_damping_spin.SetValue(
            max(0, min(0.9999, 1 - (((1 - self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax()) \
                * (self.simple_mass_slider.GetValue() / self.simple_mass_slider.GetMax())) * 0.6))))
        # 摩擦力を設定
        self.rigidbody_friction_spin.SetValue(
            max(0, min(0.9999, (self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax() * 0.7))))
        self.set_shape_maintenance(event)

    def set_shape_maintenance(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)

        base_joint_val = ((self.simple_shape_maintenance_slider.GetValue() / self.simple_shape_maintenance_slider.GetMax()) * \
                          (self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax()))

        vertical_joint_rot = max(0, min(180, 180 - base_joint_val * 180 * 1))
        self.vertical_joint_rot_x_min_spin.SetValue(-vertical_joint_rot)
        self.vertical_joint_rot_x_max_spin.SetValue(vertical_joint_rot)
        self.vertical_joint_rot_y_min_spin.SetValue(-vertical_joint_rot / 1.5)
        self.vertical_joint_rot_y_max_spin.SetValue(vertical_joint_rot / 1.5)
        self.vertical_joint_rot_z_min_spin.SetValue(-vertical_joint_rot / 1.5)
        self.vertical_joint_rot_z_max_spin.SetValue(vertical_joint_rot / 1.5)

        spring_rot = max(0, min(180, base_joint_val * 20))
        self.vertical_joint_spring_rot_x_spin.SetValue(spring_rot)
        self.vertical_joint_spring_rot_y_spin.SetValue(spring_rot)
        self.vertical_joint_spring_rot_z_spin.SetValue(spring_rot)

        horizonal_joint_rot = max(0, min(180, 180 - base_joint_val * 180 * 1.5))
        self.horizonal_joint_rot_x_min_spin.SetValue(-horizonal_joint_rot / 1.5)
        self.horizonal_joint_rot_x_max_spin.SetValue(horizonal_joint_rot / 1.5)
        self.horizonal_joint_rot_y_min_spin.SetValue(-horizonal_joint_rot / 1.5)
        self.horizonal_joint_rot_y_max_spin.SetValue(horizonal_joint_rot / 1.5)
        self.horizonal_joint_rot_z_min_spin.SetValue(-horizonal_joint_rot)
        self.horizonal_joint_rot_z_max_spin.SetValue(horizonal_joint_rot)

        spring_rot = max(0, min(180, base_joint_val * 40))
        self.horizonal_joint_spring_rot_x_spin.SetValue(spring_rot)
        self.horizonal_joint_spring_rot_y_spin.SetValue(spring_rot)
        self.horizonal_joint_spring_rot_z_spin.SetValue(spring_rot)

        diagonal_joint_rot = max(0, min(180, 180 - base_joint_val * 180 * 0.75))
        self.diagonal_joint_rot_x_min_spin.SetValue(-diagonal_joint_rot)
        self.diagonal_joint_rot_x_max_spin.SetValue(diagonal_joint_rot)
        self.diagonal_joint_rot_y_min_spin.SetValue(-diagonal_joint_rot)
        self.diagonal_joint_rot_y_max_spin.SetValue(diagonal_joint_rot)
        self.diagonal_joint_rot_z_min_spin.SetValue(-diagonal_joint_rot)
        self.diagonal_joint_rot_z_max_spin.SetValue(diagonal_joint_rot)

        spring_rot = max(0, min(180, base_joint_val * 10))
        self.diagonal_joint_spring_rot_x_spin.SetValue(spring_rot)
        self.diagonal_joint_spring_rot_y_spin.SetValue(spring_rot)
        self.diagonal_joint_spring_rot_z_spin.SetValue(spring_rot)

        reverse_joint_rot = max(0, min(180, 180 - base_joint_val * 180 * 0.5))
        self.reverse_joint_rot_x_min_spin.SetValue(-reverse_joint_rot)
        self.reverse_joint_rot_x_max_spin.SetValue(reverse_joint_rot)
        self.reverse_joint_rot_y_min_spin.SetValue(-reverse_joint_rot)
        self.reverse_joint_rot_y_max_spin.SetValue(reverse_joint_rot)
        self.reverse_joint_rot_z_min_spin.SetValue(-reverse_joint_rot)
        self.reverse_joint_rot_z_max_spin.SetValue(reverse_joint_rot)

        spring_rot = max(0, min(180, base_joint_val * 10))
        self.reverse_joint_spring_rot_x_spin.SetValue(spring_rot)
        self.reverse_joint_spring_rot_y_spin.SetValue(spring_rot)
        self.reverse_joint_spring_rot_z_spin.SetValue(spring_rot)

        if self.simple_shape_maintenance_slider.GetValue() > self.simple_shape_maintenance_slider.GetMax() * 0.6:
            # 一定以上の維持感であれば斜めも張る
            if "布" in self.simple_material_ctrl.GetStringSelection():
                # 斜めは布のみ
                self.advance_diagonal_joint_valid_check.SetValue(1)
        else:
            self.advance_diagonal_joint_valid_check.SetValue(0)

        if self.simple_shape_maintenance_slider.GetValue() > self.simple_shape_maintenance_slider.GetMax() * 0.8:
            # 一定以上の維持感であれば逆も張る
            self.advance_reverse_joint_valid_check.SetValue(1)
        else:
            self.advance_reverse_joint_valid_check.SetValue(0)

        self.on_diagonal_joint(event)
        self.on_reverse_joint(event)
    
    def on_clear(self, event: wx.Event):
        self.simple_similarity_slider.SetValue(0.75)
        self.simple_fineness_slider.SetValue(3)
        self.simple_mass_slider.SetValue(1.5)
        self.simple_air_resistance_slider.SetValue(1.8)
        self.simple_shape_maintenance_slider.SetValue(1.5)
        self.simple_direction_ctrl.SetStringSelection('下')

        self.advance_rigidbody_shape_type_ctrl.SetStringSelection('箱')
        self.physics_type_ctrl.SetStringSelection('布')
        self.bone_thinning_out_check.SetValue(0)

        self.set_material_name(event)
        self.set_fineness(event)
        self.set_mass(event)
        self.set_air_resistance(event)
        self.set_shape_maintenance(event)
        self.on_diagonal_joint(event)
        self.on_reverse_joint(event)

        self.advance_vertical_joint_valid_check.SetValue(1)
        self.advance_horizonal_joint_valid_check.SetValue(1)
        self.advance_diagonal_joint_valid_check.SetValue(0)
        self.advance_reverse_joint_valid_check.SetValue(0)

        self.advance_vertical_joint_coefficient_spin.SetValue(1.2)
        self.advance_horizonal_joint_coefficient_spin.SetValue(2.3)
        self.advance_diagonal_joint_coefficient_spin.SetValue(1)
        self.advance_reverse_joint_coefficient_spin.SetValue(1)


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
