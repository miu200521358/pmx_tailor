# -*- coding: utf-8 -*-
#
import wx
import wx.lib.newevent
from wx.grid import Grid, GridCellChoiceEditor, EVT_GRID_CELL_CHANGING

import glob
import json
import os
import unicodedata
import traceback
import csv

from form.panel.BasePanel import BasePanel
from form.parts.FloatSliderCtrl import FloatSliderCtrl
from form.parts.HistoryFilePickerCtrl import HistoryFilePickerCtrl
from mmd.PmxData import RigidBody, Joint, Bdef1, Bdef2, Bdef4, Sdef
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4  # noqa
from utils import MFileUtils
from utils.MLogger import MLogger  # noqa

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

        self.description_txt = wx.StaticText(
            self,
            wx.ID_ANY,
            logger.transtext(
                "材質を選択して、パラメーターを調整してください。\n"
                + "スライダーパラメーターで調整した設定に基づいて詳細タブ内のMMD物理パラメーターを変更します。\n"
                + "物理を再利用したい場合は、ボーンパネルでボーンの並び順を指定してください。"
            ),
            wx.DefaultPosition,
            wx.DefaultSize,
            0,
        )
        self.header_sizer.Add(self.description_txt, 0, wx.ALL, 5)

        self.static_line01 = wx.StaticLine(
            self.header_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL
        )
        self.header_sizer.Add(self.static_line01, 0, wx.EXPAND | wx.ALL, 5)

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Vroid2Pmxインポートボタン
        self.clear_btn_ctrl = wx.Button(
            self.header_panel,
            wx.ID_ANY,
            logger.transtext("Vroid2Pmx設定インポート ... "),
            wx.DefaultPosition,
            wx.DefaultSize,
            0,
        )
        self.clear_btn_ctrl.SetToolTip(logger.transtext("Vroid2Pmxで出力したPmxTailor用設定データを選択インポートします。"))
        self.clear_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_vroid2pmx_import)
        self.btn_sizer.Add(self.clear_btn_ctrl, 0, wx.ALL, 5)

        # 材質設定クリアボタン
        self.clear_btn_ctrl = wx.Button(
            self.header_panel, wx.ID_ANY, logger.transtext("物理設定クリア"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.clear_btn_ctrl.SetToolTip(logger.transtext("既に入力されたデータをすべて空にします。"))
        self.clear_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_clear_set)
        self.btn_sizer.Add(self.clear_btn_ctrl, 0, wx.ALL, 5)

        # 材質設定追加ボタン
        self.add_btn_ctrl = wx.Button(
            self.header_panel, wx.ID_ANY, logger.transtext("物理設定追加"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.add_btn_ctrl.SetToolTip(logger.transtext("物理設定フォームをパネルに追加します。"))
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

        self.scrolled_window = wx.ScrolledWindow(
            self,
            wx.ID_ANY,
            wx.DefaultPosition,
            wx.DefaultSize,
            wx.FULL_REPAINT_ON_RESIZE | wx.VSCROLL | wx.ALWAYS_SHOW_SB,
        )
        self.scrolled_window.SetScrollRate(5, 5)

        self.scrolled_window.SetSizer(self.simple_sizer)
        self.scrolled_window.Layout()
        self.sizer.Add(self.scrolled_window, 1, wx.ALL | wx.EXPAND | wx.FIXED_MINSIZE, 5)
        self.fit()

    def on_vroid2pmx_import(self, event: wx.Event):
        file_pathes = glob.glob(
            os.path.join(
                os.path.dirname(self.frame.file_panel_ctrl.org_model_file_ctrl.path()), "PmxTailorSetting", "*.json"
            )
        )
        file_names = [os.path.basename(fpath) for fpath in file_pathes]

        with wx.MultiChoiceDialog(
            self.frame,
            logger.transtext("インポートしたい設定を選んで、OKボタンをクリックしてください。複数選択OKです。\n件数が多い場合、読み込み完了まで少し時間がかかります。"),
            caption=logger.transtext("Vroid2Pmx設定インポート選択"),
            choices=file_names,
            style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.OK | wx.CANCEL | wx.CENTRE | wx.RESIZE_BORDER,
        ) as choiceDialog:

            if choiceDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            if not choiceDialog.GetSelections():
                dialog = wx.MessageDialog(
                    self.frame, logger.transtext("設定のチェックボックスが1件も選択されなかった為、登録処理をスキップします。"), style=wx.OK
                )
                dialog.ShowModal()
                dialog.Destroy()
                return

            ng_setting_names = []
            last_pidx = -1
            for cidx in choiceDialog.GetSelections():
                # 選択された数だけ設定追加
                for pidx, physics_param in enumerate(self.physics_list):
                    param, result = physics_param.get_param_options(pidx, False)
                    if param:
                        last_pidx = pidx
                if last_pidx >= 0:
                    # 1番目以降もしくは既に設定されている場合、エリア追加
                    self.on_add(event)

                try:
                    with open(file_pathes[cidx], "r", encoding="utf-8") as f:
                        vroid2pmx_setting = json.load(f)

                        self.physics_list[-1].simple_material_ctrl.SetStringSelection(
                            vroid2pmx_setting["material_name"]
                        )
                        self.physics_list[-1].simple_back_material_ctrl.SetStringSelection(
                            vroid2pmx_setting.get("back_material_name", "")
                        )
                        if vroid2pmx_setting.get("back_extend_material_names", []):
                            selected_idxs = []
                            for n, material_name in enumerate(self.material_list):
                                if material_name and material_name in vroid2pmx_setting.get(
                                    "back_extend_material_names", []
                                ):
                                    selected_idxs.append(n - 1)
                            self.physics_list[-1].simple_extend_back_choice_ctrl.SetSelections(selected_idxs)

                        self.physics_list[-1].simple_parent_bone_ctrl.SetStringSelection(
                            vroid2pmx_setting["parent_bone_name"]
                        )
                        self.physics_list[-1].simple_abb_ctrl.ChangeValue(vroid2pmx_setting["abb_name"])
                        self.physics_list[-1].simple_group_ctrl.SetStringSelection(vroid2pmx_setting["group"])
                        self.physics_list[-1].simple_direction_ctrl.SetStringSelection(vroid2pmx_setting["direction"])
                        self.physics_list[-1].simple_exist_physics_clear_ctrl.SetStringSelection(
                            vroid2pmx_setting["exist_physics_clear"]
                        )
                        self.physics_list[-1].simple_primitive_ctrl.SetStringSelection(vroid2pmx_setting["primitive"])

                        # ラベル設定(テキストボックスイベントではないので、個別設定)
                        label_text = f"{self.physics_list[-1].simple_material_ctrl.GetStringSelection()}:{self.physics_list[-1].simple_abb_ctrl.GetValue()}"
                        self.physics_list[-1].advance_material_ctrl.SetLabelText(label_text)
                        self.physics_list[-1].bone_material_ctrl.SetLabelText(label_text)

                        self.physics_list[-1].set_simple_primitive(event)

                        if "髪H" in vroid2pmx_setting["abb_name"]:
                            self.physics_list[-1].simple_mass_slider.SetValue(1.0)
                            self.physics_list[-1].simple_air_resistance_slider.SetValue(2.7)
                            self.physics_list[-1].simple_shape_maintenance_slider.SetValue(3.8)
                        elif "耳" in vroid2pmx_setting["abb_name"]:
                            self.physics_list[-1].simple_mass_slider.SetValue(0.8)
                            self.physics_list[-1].simple_air_resistance_slider.SetValue(5)
                            self.physics_list[-1].simple_shape_maintenance_slider.SetValue(3)
                        elif (
                            "CS" in vroid2pmx_setting["abb_name"]
                            or "SK" in vroid2pmx_setting["abb_name"]
                            or "CT" in vroid2pmx_setting["abb_name"]
                        ):
                            self.physics_list[-1].advance_horizonal_joint_restruct_check.SetValue(0)
                            self.physics_list[-1].simple_mass_slider.SetValue(4.6)
                            self.physics_list[-1].simple_air_resistance_slider.SetValue(3.6)
                            self.physics_list[-1].simple_shape_maintenance_slider.SetValue(4)

                        self.physics_list[-1].set_mass(event)
                        self.physics_list[-1].initialize_bone_param(event)

                        for c, target_bone_cols in enumerate(vroid2pmx_setting["target_bones"]):
                            for r, target_bone_name in enumerate(target_bone_cols):
                                if (
                                    r < self.physics_list[-1].bone_grid.GetNumberRows()
                                    and c < self.physics_list[-1].bone_grid.GetNumberCols()
                                ):
                                    self.physics_list[-1].bone_grid.GetTable().SetValue(r, c, target_bone_name)

                        if "耳" in vroid2pmx_setting["abb_name"]:
                            # ケモ耳は中心ボーンを作る
                            self.physics_list[-1].parent_type_ctrl.SetStringSelection(logger.transtext("中心"))

                        # 剛体の厚みを設定する
                        self.physics_list[-1].rigidbody_root_thicks_spin.SetValue(
                            vroid2pmx_setting.get("rigidbody_root_thick", 0.07)
                        )
                        self.physics_list[-1].rigidbody_end_thicks_spin.SetValue(
                            vroid2pmx_setting.get("rigidbody_end_thick", 0.3)
                        )
                        self.physics_list[-1].rigidbody_cover_coefficient_spin.SetValue(
                            vroid2pmx_setting.get("rigidbody_cover_coefficient", 1.2)
                        )

                        if (
                            "CS" in vroid2pmx_setting["abb_name"]
                            or "SK" in vroid2pmx_setting["abb_name"]
                            or "CT" in vroid2pmx_setting["abb_name"]
                        ):
                            self.physics_list[-1].advance_horizonal_joint_restruct_check.SetValue(0)
                            # self.physics_list[-1].horizonal_joint_mov_y_min_spin.SetValue(-3)
                            # self.physics_list[-1].horizonal_joint_mov_y_max_spin.SetValue(3)
                            self.physics_list[-1].joint_pos_type_ctrl.SetStringSelection(logger.transtext("ボーン位置"))

                except Exception:
                    ng_setting_names.append(file_names[cidx])

                last_pidx += 1

            if len(ng_setting_names) == 0:
                dialog = wx.MessageDialog(
                    self.frame, logger.transtext("Vroid2Pmxからの設定JSONのインポートに成功しました。"), style=wx.OK
                )
                dialog.ShowModal()
                dialog.Destroy()
            else:
                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("Vroid2Pmxからの設定JSONのインポートに一部失敗しました。\n\n{0}").format(",".join(ng_setting_names)),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()

    def on_add(self, event: wx.Event):
        self.physics_list.append(
            PhysicsParam(
                self.frame,
                self,
                self.scrolled_window,
                self.frame.advance_param_panel_ctrl.scrolled_window,
                self.frame.bone_param_panel_ctrl.scrolled_window,
                len(self.physics_list),
            )
        )

        # 基本
        self.simple_sizer.Add(self.physics_list[-1].simple_sizer, 0, wx.ALL | wx.EXPAND, 5)
        self.simple_sizer.Layout()

        # 詳細
        self.frame.advance_param_panel_ctrl.advance_sizer.Add(
            self.physics_list[-1].advance_sizer, 0, wx.ALL | wx.EXPAND, 5
        )
        self.frame.advance_param_panel_ctrl.advance_sizer.Layout()

        # ボーン
        self.frame.bone_param_panel_ctrl.bone_sizer.Add(self.physics_list[-1].bone_sizer, 0, wx.ALL | wx.EXPAND, 5)
        self.frame.bone_param_panel_ctrl.bone_sizer.Layout()

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
        if (
            self.frame.file_panel_ctrl.org_model_file_ctrl.data
            and self.org_model_digest != self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest
        ):
            for physics_param in self.physics_list:
                self.simple_sizer.Hide(physics_param.simple_sizer, recursive=True)
                self.frame.advance_param_panel_ctrl.advance_sizer.Hide(physics_param.advance_sizer, recursive=True)
                self.frame.bone_param_panel_ctrl.bone_sizer.Hide(physics_param.bone_sizer, recursive=True)

            # ハッシュ
            self.org_model_digest = self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest
            # 物理リストクリア
            self.physics_list = []
            # プルダウン用材質名リスト
            self.material_list = [""]
            for material_name in self.frame.file_panel_ctrl.org_model_file_ctrl.data.materials.keys():
                self.material_list.append(material_name)
            # ボーンリストクリア
            self.bone_list = [""]
            for bone in self.frame.file_panel_ctrl.org_model_file_ctrl.data.bones.values():
                if bone.getVisibleFlag():
                    self.bone_list.append(bone.name)
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
        param_material_names = []
        param_abb_names = []

        if (
            self.frame.file_panel_ctrl.org_model_file_ctrl.data
            and self.org_model_digest == self.frame.file_panel_ctrl.org_model_file_ctrl.data.digest
        ):
            for pidx, physics_param in enumerate(self.physics_list):
                param, result = physics_param.get_param_options(pidx, is_show_error)
                if param and result:
                    material_key = f"{param['material_name']}:{param['abb_name']}:{param['vertices_csv']}"
                    if material_key in param_material_names:
                        logger.error(
                            "同じ材質・同じ略称に対して複数の物理設定が割り当てられています\n異なる箇所に物理を割り当てたい場合、略称を変えているか確認してください。",
                            decoration=MLogger.DECORATION_BOX,
                        )
                        return []

                    if param["abb_name"] in param_abb_names:
                        logger.error("同じ略称が複数の物理設定が割り当てられています", decoration=MLogger.DECORATION_BOX)
                        return []

                    params.append(param)
                    param_material_names.append(material_key)
                    param_abb_names.append(param["abb_name"])
                else:
                    if not result:
                        logger.error("一部無効な物理設定が設定されています。\n物理設定を見直してください。", decoration=MLogger.DECORATION_BOX)
                        return []

        if len(params) == 0:
            logger.error("有効な物理設定が1件も設定されていません。\nモデルを選択しなおした場合、物理設定は初期化されます。", decoration=MLogger.DECORATION_BOX)
            return []

        for pidx, param in enumerate(params):
            physics_parent = param.get("physics_parent", 0)
            if physics_parent and physics_parent == pidx + 1:
                logger.error("%s番目の物理親のインデックスに自分自身が指定されています。", pidx + 1, decoration=MLogger.DECORATION_BOX)
                return []
            elif physics_parent and physics_parent > len(params) - 1:
                logger.error("%s番目の物理親のインデックスが物理設定個数を超えて指定されています。", pidx + 1, decoration=MLogger.DECORATION_BOX)
                return []

        return params


class PhysicsParam:
    def __init__(
        self,
        main_frame: wx.Frame,
        frame: wx.Frame,
        simple_window: wx.Panel,
        advance_window: wx.Panel,
        bone_window: wx.Panel,
        param_no: int,
    ):
        self.main_frame = main_frame
        self.frame = frame
        self.simple_window = simple_window
        self.advance_window = advance_window
        self.bone_window = bone_window
        self.weighted_bone_names = {}

        # 簡易版 ------------------
        self.simple_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.simple_window, wx.ID_ANY, "【No.{0}】".format(param_no + 1)), orient=wx.VERTICAL
        )
        self.simple_param_sizer = wx.BoxSizer(wx.VERTICAL)

        self.simple_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # インポートボタン
        self.import_btn_ctrl = wx.Button(
            self.simple_window, wx.ID_ANY, logger.transtext("インポート ..."), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.import_btn_ctrl.SetToolTip(logger.transtext("物理設定データをjsonファイルから読み込みます。\nファイル選択ダイアログが開きます。"))
        self.import_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_param_import)
        self.simple_btn_sizer.Add(self.import_btn_ctrl, 0, wx.ALL, 5)

        # エクスポートボタン
        self.export_btn_ctrl = wx.Button(
            self.simple_window, wx.ID_ANY, logger.transtext("エクスポート ..."), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.export_btn_ctrl.SetToolTip(
            logger.transtext("物理設定データをjsonファイルに出力します。\n（ボーン等モデルに対する情報はエクスポートできません）\n出力先を指定できます。")
        )
        self.export_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_param_export)
        self.simple_btn_sizer.Add(self.export_btn_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 0)

        self.simple_material_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_material_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("物理材質 *"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_material_txt.SetToolTip(
            logger.transtext("物理を設定する材質を選択してください。\n裾など一部にのみ物理を設定したい場合、頂点データCSVを指定してください。")
        )
        self.simple_material_txt.Wrap(-1)
        self.simple_material_sizer.Add(self.simple_material_txt, 0, wx.ALL, 5)

        self.simple_material_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.material_list)
        self.simple_material_ctrl.SetToolTip(self.simple_material_txt.GetToolTipText())
        self.simple_material_ctrl.Bind(wx.EVT_CHOICE, self.set_material_name)
        self.simple_material_sizer.Add(self.simple_material_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_material_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_parent_bone_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_parent_bone_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("親ボーン *　"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_parent_bone_txt.SetToolTip(
            logger.transtext("材質物理の起点となる親ボーン\n（指定された親ボーンの子に「○○中心」ボーンを追加して、それを起点に物理を設定します。切替オプションあり）")
        )
        self.simple_parent_bone_txt.Wrap(-1)
        self.simple_parent_bone_sizer.Add(self.simple_parent_bone_txt, 0, wx.ALL, 5)

        self.simple_parent_bone_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.bone_list)
        self.simple_parent_bone_ctrl.SetToolTip(self.simple_parent_bone_txt.GetToolTipText())
        self.simple_parent_bone_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_parent_bone_sizer.Add(self.simple_parent_bone_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_parent_bone_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_header_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        self.simple_abb_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("材質略称 *"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_abb_txt.SetToolTip(
            logger.transtext(
                "ボーン名などに使用する材質略称を半角6文字 or 全角3文字以内で入力してください。（任意変更可能。その場合は3文字まで）\nENTERキーを押すと過去に設定した物理設定を再設定します。(ボーンタブの値は再設定しません)"
            )
        )
        self.simple_abb_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_abb_txt, 0, wx.ALL, 5)

        self.simple_abb_ctrl = wx.TextCtrl(
            self.simple_window, id=wx.ID_ANY, size=wx.Size(70, -1), style=wx.TE_PROCESS_ENTER
        )
        self.simple_abb_ctrl.SetToolTip(self.simple_abb_txt.GetToolTipText())
        self.simple_abb_ctrl.SetMaxLength(6)
        self.simple_abb_ctrl.Bind(wx.EVT_TEXT, self.set_abb_name)
        self.simple_abb_ctrl.Bind(wx.EVT_TEXT_ENTER, self.set_abb_setting)
        self.simple_header_grid_sizer.Add(self.simple_abb_ctrl, 0, wx.ALL, 5)

        self.simple_group_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("剛体グループ *"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_group_txt.SetToolTip(logger.transtext("剛体のグループ。初期設定では、自分自身のグループのみ非衝突として設定します。"))
        self.simple_group_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_group_txt, 0, wx.ALL, 5)

        self.simple_group_ctrl = wx.Choice(
            self.simple_window,
            id=wx.ID_ANY,
            choices=["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
        )
        self.simple_group_ctrl.SetToolTip(self.simple_group_txt.GetToolTipText())
        self.simple_group_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_header_grid_sizer.Add(self.simple_group_ctrl, 0, wx.ALL, 5)

        self.simple_direction_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("物理方向"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_direction_txt.SetToolTip(logger.transtext("物理材質の向き(例：左腕側の物理を設定したい場合に「左」を設定して、物理が流れる方向を左方向に伸ばす)"))
        self.simple_direction_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_direction_txt, 0, wx.ALL, 5)

        self.simple_direction_ctrl = wx.Choice(
            self.simple_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("下"), logger.transtext("右"), logger.transtext("左")],
        )
        self.simple_direction_ctrl.SetToolTip(self.simple_direction_txt.GetToolTipText())
        self.simple_direction_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_header_grid_sizer.Add(self.simple_direction_ctrl, 0, wx.ALL, 5)

        self.simple_exist_physics_clear_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("既存設定"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_exist_physics_clear_txt.SetToolTip(
            logger.transtext("指定された材質に割り当てられている既存物理（ボーン・剛体・ジョイント）がある場合の挙動\nそのまま：処理しない\n")
            + logger.transtext("再利用：ボーンとウェイトは既存のものを利用し、剛体とジョイントだけ作り直す\n上書き：ボーン・剛体・ジョイントを削除して作り直す")
        )
        self.simple_exist_physics_clear_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_exist_physics_clear_txt, 0, wx.ALL, 5)

        self.simple_exist_physics_clear_ctrl = wx.Choice(
            self.simple_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("そのまま"), logger.transtext("再利用"), logger.transtext("上書き")],
        )
        self.simple_exist_physics_clear_ctrl.SetToolTip(self.simple_exist_physics_clear_txt.GetToolTipText())
        self.simple_exist_physics_clear_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_header_grid_sizer.Add(self.simple_exist_physics_clear_ctrl, 0, wx.ALL, 5)

        self.simple_primitive_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("プリセット"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_primitive_txt.SetToolTip(
            logger.transtext(
                "物理の参考値プリセット\nコットン：軽くてサラッとした質感\nシルク：とても軽くて身体にまとわりくつ質感\nウール：重めでしっかりした質感\nベルベッド：重めで翻りやすい質感\nレザー：重めで弾力がある質感\nデニム：どっしりとした質感\n単一揺れ物：縦ジョイントのみで繋ぐ汎用プリセット"
            )
        )
        self.simple_primitive_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_primitive_txt, 0, wx.ALL, 5)

        self.simple_primitive_ctrl = wx.Choice(
            self.simple_window,
            id=wx.ID_ANY,
            choices=[
                logger.transtext("布(コットン)"),
                logger.transtext("布(シルク)"),
                logger.transtext("布(ウール)"),
                logger.transtext("布(ベルベッド)"),
                logger.transtext("布(レザー)"),
                logger.transtext("布(デニム)"),
                logger.transtext("髪(ショート)"),
                logger.transtext("髪(ロング)"),
                logger.transtext("髪(アホ毛)"),
                logger.transtext("胸(小)"),
                logger.transtext("胸(大)"),
                logger.transtext("単一揺れ物"),
            ],
        )
        self.simple_primitive_ctrl.SetToolTip(self.simple_primitive_txt.GetToolTipText())
        self.simple_primitive_ctrl.Bind(wx.EVT_CHOICE, self.set_simple_primitive)
        self.simple_header_grid_sizer.Add(self.simple_primitive_ctrl, 0, wx.ALL, 5)

        self.simple_special_shape_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("特殊形状"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_special_shape_txt.SetToolTip(
            logger.transtext(
                "スカート等で特殊な処理が必要な形状\n"
                + "全て表面: プリーツ（ポリ割に折り返しがある）などの形状で裏面判定が誤検知をする場合、強制的に全ての面を表面として扱います（厚みは裏面材質に分けてください）\n"
                + "面欠け: VRoid Studioで「透明メッシュの削除」（デフォルトON）の状態でpmxに出力するなどして、エッジではなくでこぼこしている場合、仮想面を貼って安定した物理を設定します\n"
            )
        )
        self.simple_special_shape_txt.Wrap(-1)
        self.simple_header_grid_sizer.Add(self.simple_special_shape_txt, 0, wx.ALL, 5)

        self.simple_special_shape_ctrl = wx.Choice(
            self.simple_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("なし"), logger.transtext("全て表面"), logger.transtext("面欠け")],
        )
        self.simple_special_shape_ctrl.SetToolTip(self.simple_special_shape_txt.GetToolTipText())
        self.simple_special_shape_ctrl.Bind(wx.EVT_CHOICE, self.on_special_shape)
        self.simple_header_grid_sizer.Add(self.simple_special_shape_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_header_grid_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_edge_material_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_edge_material_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("裾材質　　"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_edge_material_txt.SetToolTip(
            logger.transtext("物理材質の裾（など別材質）にあたる材質がある場合、選択してください。\n物理材質のボーン割りに応じてウェイトを割り当てます")
        )
        self.simple_edge_material_txt.Wrap(-1)
        self.simple_edge_material_sizer.Add(self.simple_edge_material_txt, 0, wx.ALL, 5)

        self.simple_edge_material_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.material_list)
        self.simple_edge_material_ctrl.SetToolTip(self.simple_edge_material_txt.GetToolTipText())
        self.simple_edge_material_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_edge_material_sizer.Add(self.simple_edge_material_ctrl, 0, wx.ALL, 5)

        # 追加裾材質ダイアログ
        self.simple_extend_edge_choice_ctrl = wx.MultiChoiceDialog(
            self.frame,
            logger.transtext("追加で裾材質を指定したい場合に選択してください。\n裾の裏は裏材質に割り当ててください"),
            caption="追加裾材質選択",
            choices=self.frame.material_list[1:],
            style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.OK | wx.CANCEL | wx.CENTRE,
        )
        self.simple_extend_edge_choice_ctrl.Hide()

        # 追加裾材質ボタン
        self.simple_extend_edge_btn_ctrl = wx.Button(
            self.simple_window, wx.ID_ANY, logger.transtext("追加裾材質"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_extend_edge_btn_ctrl.SetToolTip(logger.transtext("追加で裾材質を指定したい場合に選択してください。\n裾の裏は裏材質に割り当ててください"))
        self.simple_extend_edge_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_click_extend_edge)
        self.simple_edge_material_sizer.Add(self.simple_extend_edge_btn_ctrl, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_edge_material_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_back_material_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.simple_back_material_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("裏面材質　"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_back_material_txt.SetToolTip(
            logger.transtext("物理材質の裏面にあたる材質がある場合、選択してください。\n物理材質の最も近い頂点ウェイトを転写します")
        )
        self.simple_back_material_txt.Wrap(-1)
        self.simple_back_material_sizer.Add(self.simple_back_material_txt, 0, wx.ALL, 5)

        self.simple_back_material_ctrl = wx.Choice(self.simple_window, id=wx.ID_ANY, choices=self.frame.material_list)
        self.simple_back_material_ctrl.SetToolTip(self.simple_back_material_txt.GetToolTipText())
        self.simple_back_material_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.simple_back_material_sizer.Add(self.simple_back_material_ctrl, 0, wx.ALL, 5)

        # 追加裏材質ダイアログ
        self.simple_extend_back_choice_ctrl = wx.MultiChoiceDialog(
            self.frame,
            logger.transtext("追加で裏材質を指定したい場合に選択してください。"),
            caption="追加裏材質選択",
            choices=self.frame.material_list[1:],
            style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.OK | wx.CANCEL | wx.CENTRE,
        )
        self.simple_extend_back_choice_ctrl.Hide()

        # 追加裏材質ボタン
        self.simple_extend_back_btn_ctrl = wx.Button(
            self.simple_window, wx.ID_ANY, logger.transtext("追加裏材質"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_extend_back_btn_ctrl.SetToolTip(logger.transtext("追加で裏材質を指定したい場合に選択してください。"))
        self.simple_extend_back_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_click_extend_back)
        self.simple_back_material_sizer.Add(self.simple_extend_back_btn_ctrl, 0, wx.ALL, 5)
        self.simple_param_sizer.Add(self.simple_back_material_sizer, 0, wx.ALL | wx.EXPAND, 0)

        self.simple_grid_sizer = wx.FlexGridSizer(0, 5, 0, 0)

        # 剛体の質量スライダー
        self.simple_mass_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("質量"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_mass_txt.SetToolTip(logger.transtext("材質の質量。剛体の質量・減衰等に影響します。"))
        self.simple_mass_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_txt, 0, wx.ALL, 5)

        self.simple_mass_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("（0.5）"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_mass_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_label, 0, wx.ALL, 5)

        self.simple_mass_min_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("軽"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_mass_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_min_label, 0, wx.ALL, 5)

        self.simple_mass_slider = FloatSliderCtrl(
            self.simple_window,
            wx.ID_ANY,
            1.5,
            0.01,
            10,
            0.01,
            self.simple_mass_label,
            wx.DefaultPosition,
            (350, 30),
            wx.SL_HORIZONTAL,
        )
        self.simple_mass_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_mass)
        self.simple_grid_sizer.Add(self.simple_mass_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_mass_max_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("重"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_mass_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_mass_max_label, 0, wx.ALL, 5)

        # 空気抵抗スライダー
        self.simple_air_resistance_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("柔らかさ"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_air_resistance_txt.SetToolTip(
            logger.transtext("材質の柔らかさ。剛体の減衰・ジョイントの移動回転制限に影響します。\n小さくなるほどよく変形し、すぐに元の形状に戻ります。（回転制限が大きく、減衰が高い）")
        )
        self.simple_air_resistance_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_txt, 0, wx.ALL, 5)

        self.simple_air_resistance_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("（1.8）"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_air_resistance_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_label, 0, wx.ALL, 5)

        self.simple_air_resistance_min_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("柔"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_air_resistance_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_min_label, 0, wx.ALL, 5)

        self.simple_air_resistance_slider = FloatSliderCtrl(
            self.simple_window,
            wx.ID_ANY,
            1.8,
            0.01,
            5,
            0.01,
            self.simple_air_resistance_label,
            wx.DefaultPosition,
            (350, 30),
            wx.SL_HORIZONTAL,
        )
        self.simple_air_resistance_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_air_resistance)
        self.simple_grid_sizer.Add(self.simple_air_resistance_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_air_resistance_max_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("硬"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_air_resistance_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_air_resistance_max_label, 0, wx.ALL, 5)

        # 形状維持スライダー
        self.simple_shape_maintenance_txt = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("張り"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_shape_maintenance_txt.SetToolTip(
            logger.transtext("材質の形状維持強度。ジョイントのバネの強さに影響します。\n大きいほど変形の振り幅が大きくなります（あんまり変形しないが、変形するときは大きく変形する）")
        )
        self.simple_shape_maintenance_txt.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_txt, 0, wx.ALL, 5)

        self.simple_shape_maintenance_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("（1.5）"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_shape_maintenance_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_label, 0, wx.ALL, 5)

        self.simple_shape_maintenance_min_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("弱"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_shape_maintenance_min_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_min_label, 0, wx.ALL, 5)

        self.simple_shape_maintenance_slider = FloatSliderCtrl(
            self.simple_window,
            wx.ID_ANY,
            1.5,
            0.01,
            5,
            0.01,
            self.simple_shape_maintenance_label,
            wx.DefaultPosition,
            (350, 30),
            wx.SL_HORIZONTAL,
        )
        self.simple_shape_maintenance_slider.Bind(wx.EVT_SCROLL_CHANGED, self.set_shape_maintenance)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_slider, 1, wx.ALL | wx.EXPAND, 5)

        self.simple_shape_maintenance_max_label = wx.StaticText(
            self.simple_window, wx.ID_ANY, logger.transtext("強"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.simple_shape_maintenance_max_label.Wrap(-1)
        self.simple_grid_sizer.Add(self.simple_shape_maintenance_max_label, 0, wx.ALL, 5)

        self.simple_param_sizer.Add(self.simple_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)

        # 頂点CSVファイルコントロール
        self.vertices_csv_file_ctrl = HistoryFilePickerCtrl(
            self.main_frame,
            self.simple_window,
            logger.transtext("物理対象頂点CSV"),
            logger.transtext("物理対象頂点CSVファイルを開く"),
            ("csv"),
            wx.FLP_DEFAULT_STYLE,
            logger.transtext(
                "材質の中で物理を割り当てたい頂点を絞り込みたい場合、PmxEditorで頂点リストを選択できるようにして保存した頂点CSVファイルを指定してください。\n"
                + "根元頂点CSVが指定されている場合、根元より上（親ボーンに近い）の頂点はウェイトを物理ボーンとのグラデーションで塗り直します。\n"
                + "D&Dでの指定、開くボタンからの指定、履歴からの選択ができます。"
            ),
            file_model_spacer=0,
            title_parts_ctrl=None,
            title_parts2_ctrl=None,
            file_histories_key="vertices_csv",
            is_change_output=False,
            is_aster=False,
            is_save=False,
            set_no=0,
        )
        self.vertices_csv_file_ctrl.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_change_vertices_csv)
        self.simple_param_sizer.Add(self.vertices_csv_file_ctrl.sizer, 0, wx.EXPAND, 0)

        # 根元頂点CSVファイルコントロール
        self.top_vertices_csv_file_ctrl = HistoryFilePickerCtrl(
            self.main_frame,
            self.simple_window,
            logger.transtext("根元頂点CSV"),
            logger.transtext("根元頂点CSVファイルを開く"),
            ("csv"),
            wx.FLP_DEFAULT_STYLE,
            logger.transtext(
                "（自動判定で根元がうまく抽出できなかったなどで）根元頂点を指定したい場合、PmxEditorで頂点リストを選択できるようにして保存した頂点CSVファイルを指定してください。\n根元頂点は物理対象頂点にも含まれている必要があります。\nD&Dでの指定、開くボタンからの指定、履歴からの選択ができます。"
            ),
            file_model_spacer=0,
            title_parts_ctrl=None,
            title_parts2_ctrl=None,
            file_histories_key="top_vertices_csv",
            is_change_output=False,
            is_aster=False,
            is_save=False,
            set_no=0,
        )
        self.top_vertices_csv_file_ctrl.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_change_vertices_csv)
        self.simple_param_sizer.Add(self.top_vertices_csv_file_ctrl.sizer, 0, wx.EXPAND, 0)

        # 裾対象頂点CSVファイルコントロール
        self.vertices_edge_csv_file_ctrl = HistoryFilePickerCtrl(
            self.main_frame,
            self.simple_window,
            logger.transtext("裾対象頂点CSV"),
            logger.transtext("裾対象頂点CSVファイルを開く"),
            ("csv"),
            wx.FLP_DEFAULT_STYLE,
            logger.transtext(
                "材質ではなく特定の頂点だけ裾対象としてウェイトを塗り直したい場合、PmxEditorで頂点リストを選択できるようにして保存した頂点CSVファイルを指定してください。\n"
                + "ファイル名を「*.csv」にすることで、フォルダ内の全CSVファイルを対象とすることができます。\n"
                + "D&Dでの指定、開くボタンからの指定、履歴からの選択ができます。"
            ),
            file_model_spacer=0,
            title_parts_ctrl=None,
            title_parts2_ctrl=None,
            file_histories_key="vertices_edge_csv",
            is_change_output=False,
            is_aster=True,
            is_save=False,
            set_no=0,
        )
        self.vertices_edge_csv_file_ctrl.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_change_vertices_csv)
        self.simple_param_sizer.Add(self.vertices_edge_csv_file_ctrl.sizer, 0, wx.EXPAND, 0)

        # 裏面対象頂点CSVファイルコントロール
        self.vertices_back_csv_file_ctrl = HistoryFilePickerCtrl(
            self.main_frame,
            self.simple_window,
            logger.transtext("裏面対象頂点CSV"),
            logger.transtext("裏面対象頂点CSVファイルを開く"),
            ("csv"),
            wx.FLP_DEFAULT_STYLE,
            logger.transtext(
                "材質ではなく特定の頂点だけ裏面対象としてウェイトを塗り直したい場合、PmxEditorで頂点リストを選択できるようにして保存した頂点CSVファイルを指定してください。\n"
                + "ファイル名を「*.csv」にすることで、フォルダ内の全CSVファイルを対象とすることができます。\n"
                + "D&Dでの指定、開くボタンからの指定、履歴からの選択ができます。"
            ),
            file_model_spacer=0,
            title_parts_ctrl=None,
            title_parts2_ctrl=None,
            file_histories_key="vertices_back_csv",
            is_change_output=False,
            is_aster=True,
            is_save=False,
            set_no=0,
        )
        self.vertices_back_csv_file_ctrl.file_ctrl.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_change_vertices_csv)
        self.simple_param_sizer.Add(self.vertices_back_csv_file_ctrl.sizer, 0, wx.EXPAND, 0)

        self.simple_sizer.Add(self.simple_param_sizer, 1, wx.ALL | wx.EXPAND, 5)

        # 詳細版 ------------------
        self.advance_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, "【No.{0}】".format(param_no + 1)), orient=wx.VERTICAL
        )
        self.advance_param_sizer = wx.BoxSizer(wx.VERTICAL)

        self.advance_material_ctrl = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("（材質未選択）"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_param_sizer.Add(self.advance_material_ctrl, 0, wx.ALL, 5)

        self.advance_sizer.Add(self.advance_param_sizer, 1, wx.ALL | wx.EXPAND, 0)

        # ボーン密度ブロック -------------------------------
        self.advance_bone_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("ボーン密度")), orient=wx.VERTICAL
        )
        self.advance_bone_grid_sizer = wx.FlexGridSizer(0, 8, 0, 0)

        # 縦密度
        self.vertical_bone_density_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("縦密度"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_bone_density_txt.SetToolTip(logger.transtext("ボーンの縦方向のメッシュに対する密度"))
        self.vertical_bone_density_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.vertical_bone_density_txt, 0, wx.ALL, 5)

        self.vertical_bone_density_spin = wx.SpinCtrl(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(50, -1), value="2", min=1, max=20, initial=2
        )
        self.vertical_bone_density_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.vertical_bone_density_spin, 0, wx.ALL, 5)

        # 横密度
        self.horizonal_bone_density_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("横密度"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_bone_density_txt.SetToolTip(
            logger.transtext("ボーンの横方向の裾メッシュに対する密度\n根元と裾でメッシュの細かさが異なる場合、裾側のメッシュで数えてください。")
        )
        self.horizonal_bone_density_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_density_txt, 0, wx.ALL, 5)

        self.horizonal_bone_density_spin = wx.SpinCtrl(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(50, -1), value="2", min=1, max=20, initial=2
        )
        self.horizonal_bone_density_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_density_spin, 0, wx.ALL, 5)

        # 横オフセット
        self.horizonal_bone_offset_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("横オフセット"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_bone_offset_txt.SetToolTip(
            logger.transtext("横密度を2以上にした場合のボーンの配置をずらすオフセット値。\n＋の場合…右方向にずらす。－の場合…左方向にずらす。")
        )
        self.horizonal_bone_offset_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_offset_txt, 0, wx.ALL, 5)

        self.horizonal_bone_offset_spin = wx.SpinCtrl(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(50, -1), value="0", min=-100, max=100, initial=0
        )
        self.horizonal_bone_offset_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_bone_grid_sizer.Add(self.horizonal_bone_offset_spin, 0, wx.ALL, 5)

        # 密度計算タイプ
        self.density_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("密度基準"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.density_type_txt.SetToolTip(
            logger.transtext(
                "頂点：実際の頂点の密度で計算する（頂点スキップ可能性なし）\n距離：頂点の距離を等間隔に繋いだ密度で計算する（頂点スキップ可能性あり）\n中央：メッシュ縦方向の真ん中のみ計算対象とする"
            )
        )
        self.density_type_txt.Wrap(-1)
        self.advance_bone_grid_sizer.Add(self.density_type_txt, 0, wx.ALL, 5)

        self.density_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("頂点"), logger.transtext("距離"), logger.transtext("中央")],
        )
        self.density_type_ctrl.SetToolTip(self.density_type_txt.GetToolTipText())
        self.density_type_ctrl.Bind(wx.EVT_CHOICE, self.set_density_type)
        self.advance_bone_grid_sizer.Add(self.density_type_ctrl, 0, wx.ALL, 5)

        # # 間引きオプション
        # self.bone_thinning_out_check = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("間引き")
        # self.bone_thinning_out_check.SetToolTip("ボーン密度が均一になるよう間引きするか否か")
        # self.advance_bone_grid_sizer.Add(self.bone_thinning_out_check, 0, wx.ALL, 5)

        self.advance_bone_sizer.Add(self.advance_bone_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_bone_sizer, 0, wx.ALL, 5)

        # 根元剛体ブロック -------------------------------
        self.advance_rigidbody_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("根元剛体")), orient=wx.VERTICAL
        )
        self.advance_rigidbody_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 質量
        self.rigidbody_mass_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("質量"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_mass_txt.SetToolTip(logger.transtext("根元剛体の質量"))
        self.rigidbody_mass_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_mass_txt, 0, wx.ALL, 5)

        self.rigidbody_mass_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.5",
            min=0.000001,
            max=1000,
            initial=0.5,
            inc=0.01,
        )
        self.rigidbody_mass_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_mass_spin, 0, wx.ALL, 5)

        # 移動減衰
        self.rigidbody_linear_damping_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動減衰"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_linear_damping_txt.SetToolTip(logger.transtext("根元剛体の移動減衰"))
        self.rigidbody_linear_damping_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_linear_damping_txt, 0, wx.ALL, 5)

        self.rigidbody_linear_damping_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.999",
            min=0.000001,
            max=0.9999999,
            initial=0.999,
            inc=0.01,
        )
        self.rigidbody_linear_damping_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_linear_damping_spin, 0, wx.ALL, 5)

        # 回転減衰
        self.rigidbody_angular_damping_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転減衰"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_angular_damping_txt.SetToolTip(logger.transtext("根元剛体の回転減衰"))
        self.rigidbody_angular_damping_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_angular_damping_txt, 0, wx.ALL, 5)

        self.rigidbody_angular_damping_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.999",
            min=0.000001,
            max=0.9999999,
            initial=0.999,
            inc=0.01,
        )
        self.rigidbody_angular_damping_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_angular_damping_spin, 0, wx.ALL, 5)

        # 反発力
        self.rigidbody_restitution_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("反発力"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_restitution_txt.SetToolTip(logger.transtext("根元剛体の反発力"))
        self.rigidbody_restitution_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_restitution_txt, 0, wx.ALL, 5)

        self.rigidbody_restitution_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=10, initial=0, inc=0.01
        )
        self.rigidbody_restitution_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_restitution_spin, 0, wx.ALL, 5)

        # 摩擦力
        self.rigidbody_friction_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("摩擦力"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_friction_txt.SetToolTip(logger.transtext("根元剛体の摩擦力"))
        self.rigidbody_friction_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_friction_txt, 0, wx.ALL, 5)

        self.rigidbody_friction_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.2",
            min=0,
            max=1000,
            initial=0.2,
            inc=0.01,
        )
        self.rigidbody_friction_spin.Bind(wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1))
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_friction_spin, 0, wx.ALL, 5)

        # 係数
        self.rigidbody_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_coefficient_txt.SetToolTip(logger.transtext("根元剛体から下の剛体にかけての加算係数"))
        self.rigidbody_coefficient_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_coefficient_txt, 0, wx.ALL, 5)

        self.rigidbody_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1.2", min=1, max=20, initial=1.2, inc=0.1
        )
        self.rigidbody_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_grid_sizer.Add(self.rigidbody_coefficient_spin, 0, wx.ALL, 5)

        # 剛体形状
        self.advance_rigidbody_shape_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("剛体形状"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_rigidbody_shape_type_txt.SetToolTip(logger.transtext("剛体の形状"))
        self.advance_rigidbody_shape_type_txt.Wrap(-1)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_shape_type_txt, 0, wx.ALL, 5)

        self.advance_rigidbody_shape_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("球"), logger.transtext("箱"), logger.transtext("カプセル")],
        )
        self.advance_rigidbody_shape_type_ctrl.SetToolTip(logger.transtext("剛体の形状"))
        self.advance_rigidbody_shape_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_shape_type_ctrl, 0, wx.ALL, 5)

        self.advance_rigidbody_balancer_ctrl = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("バランサー剛体"))
        self.advance_rigidbody_balancer_ctrl.SetToolTip(logger.transtext("バランサー剛体を作成するか否か"))
        self.advance_rigidbody_balancer_ctrl.Bind(wx.EVT_CHECKBOX, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_balancer_ctrl, 0, wx.ALL, 5)

        self.advance_rigidbody_leg_ctrl = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("足に繋げる"))
        self.advance_rigidbody_leg_ctrl.SetToolTip(
            logger.transtext("物理中心剛体を足剛体の子どもとするか否か\n下半身と足の半分より下から物理を始めたい場合（足の動きに物理を合わせたい場合）にチェックしてください")
        )
        self.advance_rigidbody_leg_ctrl.Bind(wx.EVT_CHECKBOX, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_rigidbody_grid_sizer.Add(self.advance_rigidbody_leg_ctrl, 0, wx.ALL, 5)

        self.advance_rigidbody_sizer.Add(self.advance_rigidbody_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_rigidbody_sizer, 0, wx.ALL, 5)

        # 剛体サイズブロック -------------------------------
        self.advance_rigidbody_thicks_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("剛体サイズ")), orient=wx.VERTICAL
        )
        self.advance_rigidbody_thicks_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 根元厚み
        self.rigidbody_root_thicks_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("根元厚み"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_root_thicks_txt.SetToolTip(logger.transtext("根元剛体の厚み"))
        self.rigidbody_root_thicks_txt.Wrap(-1)
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_root_thicks_txt, 0, wx.ALL, 5)

        self.rigidbody_root_thicks_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.07",
            min=0.01,
            max=10,
            initial=0.07,
            inc=0.01,
        )
        self.rigidbody_root_thicks_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_root_thicks_spin, 0, wx.ALL, 5)

        # 末端厚み
        self.rigidbody_end_thicks_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("末端厚み"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_end_thicks_txt.SetToolTip(logger.transtext("末端剛体の厚み"))
        self.rigidbody_end_thicks_txt.Wrap(-1)
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_end_thicks_txt, 0, wx.ALL, 5)

        self.rigidbody_end_thicks_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="0.3",
            min=0.01,
            max=10,
            initial=0.3,
            inc=0.01,
        )
        self.rigidbody_end_thicks_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_end_thicks_spin, 0, wx.ALL, 5)

        # カバー率
        self.rigidbody_cover_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("カバー率"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.rigidbody_cover_coefficient_txt.SetToolTip(
            logger.transtext("剛体のはみ出し具合。値が大きいほど剛体の縦横が大きくなり、足が貫通しにくくなります。\n大きすぎるとちょっと凸凹します。")
        )
        self.rigidbody_cover_coefficient_txt.Wrap(-1)
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_cover_coefficient_txt, 0, wx.ALL, 5)

        self.rigidbody_cover_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window,
            id=wx.ID_ANY,
            size=wx.Size(90, -1),
            value="1.2",
            min=1,
            max=2,
            initial=1.2,
            inc=0.01,
        )
        self.rigidbody_cover_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_rigidbody_thicks_grid_sizer.Add(self.rigidbody_cover_coefficient_spin, 0, wx.ALL, 5)

        self.advance_rigidbody_thicks_sizer.Add(self.advance_rigidbody_thicks_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_rigidbody_thicks_sizer, 0, wx.ALL, 5)

        # 縦ジョイントブロック -------------------------------
        self.advance_vertical_joint_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("縦ジョイント")), orient=wx.VERTICAL
        )

        self.advance_vertical_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_vertical_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("有効"))
        self.advance_vertical_joint_valid_check.SetToolTip(logger.transtext("縦ジョイントを有効にするか否か"))
        self.advance_vertical_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_vertical_joint)
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_vertical_joint_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("制限係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_vertical_joint_coefficient_txt.SetToolTip(
            logger.transtext("根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        )
        self.advance_vertical_joint_coefficient_txt.Wrap(-1)
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_vertical_joint_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="2.8", min=1, max=50, initial=1, inc=0.1
        )
        self.advance_vertical_joint_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_head_sizer.Add(self.advance_vertical_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_vertical_joint_sizer.Add(self.advance_vertical_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_vertical_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.vertical_joint_mov_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_x_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動X(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_x_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_mov_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.vertical_joint_mov_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_y_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動Y(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_y_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_mov_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.vertical_joint_mov_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_z_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動Z(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_z_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_mov_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.vertical_joint_mov_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_x_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動X(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_x_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_mov_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.vertical_joint_mov_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_y_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動Y(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_y_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_mov_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.vertical_joint_mov_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_mov_z_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの移動Z(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_joint_mov_z_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_mov_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_mov_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.vertical_joint_rot_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_x_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転X(最小)"))
        self.vertical_joint_rot_x_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_rot_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.vertical_joint_rot_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_y_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転Y(最小)"))
        self.vertical_joint_rot_y_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_rot_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.vertical_joint_rot_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_z_min_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転Z(最小)"))
        self.vertical_joint_rot_z_min_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_joint_rot_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.vertical_joint_rot_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_x_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転X(最大)"))
        self.vertical_joint_rot_x_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_rot_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.vertical_joint_rot_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_y_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転Y(最大)"))
        self.vertical_joint_rot_y_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_rot_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.vertical_joint_rot_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_rot_z_max_txt.SetToolTip(logger.transtext("末端縦ジョイントの回転Z(最大)"))
        self.vertical_joint_rot_z_max_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.vertical_joint_rot_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_rot_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.vertical_joint_spring_mov_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_mov_x_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(移動X)"))
        self.vertical_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_mov_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.vertical_joint_spring_mov_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_mov_y_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(移動Y)"))
        self.vertical_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_mov_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.vertical_joint_spring_mov_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_mov_z_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(移動Z)"))
        self.vertical_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_mov_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_mov_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.vertical_joint_spring_rot_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_rot_x_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(回転X)"))
        self.vertical_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_rot_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.vertical_joint_spring_rot_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_rot_y_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(回転Y)"))
        self.vertical_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_rot_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.vertical_joint_spring_rot_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_joint_spring_rot_z_txt.SetToolTip(logger.transtext("末端縦ジョイントのばね(回転Z)"))
        self.vertical_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.vertical_joint_spring_rot_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_joint_spring_rot_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_joint_grid_sizer.Add(self.vertical_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_vertical_joint_sizer.Add(self.advance_vertical_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_vertical_joint_sizer, 0, wx.ALL, 5)

        # 横ジョイントブロック -------------------------------
        self.advance_horizonal_joint_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("横ジョイント")), orient=wx.VERTICAL
        )

        self.advance_horizonal_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_horizonal_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("有効"))
        self.advance_horizonal_joint_valid_check.SetToolTip(logger.transtext("横ジョイントを有効にするか否か"))
        self.advance_horizonal_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_horizonal_joint)
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_horizonal_joint_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("制限係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_horizonal_joint_coefficient_txt.SetToolTip(
            logger.transtext("根元ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        )
        self.advance_horizonal_joint_coefficient_txt.Wrap(-1)
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_horizonal_joint_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="4.2", min=1, max=50, initial=4.2, inc=0.1
        )
        self.advance_horizonal_joint_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_horizonal_joint_sizer.Add(self.advance_horizonal_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_horizonal_joint_restruct_check = wx.CheckBox(
            self.advance_window, wx.ID_ANY, logger.transtext("親剛体距離制限")
        )
        self.advance_horizonal_joint_restruct_check.SetToolTip(logger.transtext("親剛体との距離が近い場合にジョイントの可動域を制限するか否か"))
        self.advance_horizonal_joint_head_sizer.Add(self.advance_horizonal_joint_restruct_check, 0, wx.ALL, 5)

        self.advance_horizonal_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.horizonal_joint_mov_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_x_min_txt.SetToolTip(logger.transtext("末端横ジョイントの移動X(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_x_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.horizonal_joint_mov_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_y_min_txt.SetToolTip(logger.transtext("末端横ジョイントの移動Y(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_y_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.horizonal_joint_mov_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_z_min_txt.SetToolTip(logger.transtext("末端横ジョイントの移動Z(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_z_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.horizonal_joint_mov_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_x_max_txt.SetToolTip(logger.transtext("末端横ジョイントの移動X(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_x_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.horizonal_joint_mov_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_y_max_txt.SetToolTip(logger.transtext("末端横ジョイントの移動Y(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_y_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.horizonal_joint_mov_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_mov_z_max_txt.SetToolTip(logger.transtext("末端横ジョイントの移動Z(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_joint_mov_z_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_mov_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_mov_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.horizonal_joint_rot_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_x_min_txt.SetToolTip(logger.transtext("末端横ジョイントの回転X(最小)"))
        self.horizonal_joint_rot_x_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.horizonal_joint_rot_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_y_min_txt.SetToolTip(logger.transtext("末端横ジョイントの回転Y(最小)"))
        self.horizonal_joint_rot_y_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.horizonal_joint_rot_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_z_min_txt.SetToolTip(logger.transtext("末端横ジョイントの回転Z(最小)"))
        self.horizonal_joint_rot_z_min_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.horizonal_joint_rot_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_x_max_txt.SetToolTip(logger.transtext("末端横ジョイントの回転X(最大)"))
        self.horizonal_joint_rot_x_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.horizonal_joint_rot_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_y_max_txt.SetToolTip(logger.transtext("末端横ジョイントの回転Y(最大)"))
        self.horizonal_joint_rot_y_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.horizonal_joint_rot_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_rot_z_max_txt.SetToolTip(logger.transtext("末端横ジョイントの回転Z(最大)"))
        self.horizonal_joint_rot_z_max_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_joint_rot_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_rot_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.horizonal_joint_spring_mov_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_mov_x_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(移動X)"))
        self.horizonal_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_mov_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.horizonal_joint_spring_mov_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_mov_y_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(移動Y)"))
        self.horizonal_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_mov_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.horizonal_joint_spring_mov_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_mov_z_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(移動Z)"))
        self.horizonal_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_mov_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_mov_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.horizonal_joint_spring_rot_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_rot_x_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(回転X)"))
        self.horizonal_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_rot_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.horizonal_joint_spring_rot_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_rot_y_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(回転Y)"))
        self.horizonal_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_rot_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.horizonal_joint_spring_rot_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_joint_spring_rot_z_txt.SetToolTip(logger.transtext("末端横ジョイントのばね(回転Z)"))
        self.horizonal_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.horizonal_joint_spring_rot_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_joint_spring_rot_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_joint_grid_sizer.Add(self.horizonal_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_horizonal_joint_sizer.Add(self.advance_horizonal_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_horizonal_joint_sizer, 0, wx.ALL, 5)

        # 斜めジョイントブロック -------------------------------
        self.advance_diagonal_joint_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("斜めジョイント")), orient=wx.VERTICAL
        )

        self.advance_diagonal_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_diagonal_joint_valid_check = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("有効"))
        self.advance_diagonal_joint_valid_check.SetToolTip(logger.transtext("斜めジョイントを有効にするか否か"))
        self.advance_diagonal_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_diagonal_joint)
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_valid_check, 0, wx.ALL, 5)

        # 係数
        self.advance_diagonal_joint_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("制限係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_diagonal_joint_coefficient_txt.SetToolTip(
            logger.transtext("末端ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        )
        self.advance_diagonal_joint_coefficient_txt.Wrap(-1)
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_coefficient_txt, 0, wx.ALL, 5)

        self.advance_diagonal_joint_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=50, initial=1, inc=0.1
        )
        self.advance_diagonal_joint_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_head_sizer.Add(self.advance_diagonal_joint_coefficient_spin, 0, wx.ALL, 5)

        self.advance_diagonal_joint_sizer.Add(self.advance_diagonal_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_diagonal_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.diagonal_joint_mov_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_x_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動X(最小)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_x_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.diagonal_joint_mov_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_y_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動Y(最小)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_y_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.diagonal_joint_mov_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_z_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動Z(最小)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_z_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.diagonal_joint_mov_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_x_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動X(最大)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_x_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.diagonal_joint_mov_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_y_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動Y(最大)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_y_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.diagonal_joint_mov_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_mov_z_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの移動Z(最大)\n1ミクセルあたりの許容移動量"))
        self.diagonal_joint_mov_z_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_mov_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_mov_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.diagonal_joint_rot_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_x_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転X(最小)"))
        self.diagonal_joint_rot_x_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.diagonal_joint_rot_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_y_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転Y(最小)"))
        self.diagonal_joint_rot_y_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.diagonal_joint_rot_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_z_min_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転Z(最小)"))
        self.diagonal_joint_rot_z_min_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.diagonal_joint_rot_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_x_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転X(最大)"))
        self.diagonal_joint_rot_x_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.diagonal_joint_rot_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_y_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転Y(最大)"))
        self.diagonal_joint_rot_y_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.diagonal_joint_rot_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_rot_z_max_txt.SetToolTip(logger.transtext("末端斜めジョイントの回転Z(最大)"))
        self.diagonal_joint_rot_z_max_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.diagonal_joint_rot_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_rot_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.diagonal_joint_spring_mov_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_mov_x_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(移動X)"))
        self.diagonal_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_mov_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.diagonal_joint_spring_mov_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_mov_y_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(移動Y)"))
        self.diagonal_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_mov_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.diagonal_joint_spring_mov_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_mov_z_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(移動Z)"))
        self.diagonal_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_mov_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_mov_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.diagonal_joint_spring_rot_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_rot_x_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(回転X)"))
        self.diagonal_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_rot_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.diagonal_joint_spring_rot_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_rot_y_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(回転Y)"))
        self.diagonal_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_rot_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.diagonal_joint_spring_rot_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.diagonal_joint_spring_rot_z_txt.SetToolTip(logger.transtext("末端斜めジョイントのばね(回転Z)"))
        self.diagonal_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.diagonal_joint_spring_rot_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.diagonal_joint_spring_rot_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_diagonal_joint_grid_sizer.Add(self.diagonal_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_diagonal_joint_sizer.Add(self.advance_diagonal_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_diagonal_joint_sizer, 0, wx.ALL, 5)

        # 縦逆ジョイントブロック -------------------------------
        self.advance_vertical_reverse_joint_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("縦逆ジョイント")), orient=wx.VERTICAL
        )

        self.advance_vertical_reverse_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_vertical_reverse_joint_valid_check = wx.CheckBox(
            self.advance_window, wx.ID_ANY, logger.transtext("有効")
        )
        self.advance_vertical_reverse_joint_valid_check.SetToolTip(logger.transtext("縦逆ジョイントを有効にするか否か"))
        self.advance_vertical_reverse_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_vertical_reverse_joint)
        self.advance_vertical_reverse_joint_head_sizer.Add(
            self.advance_vertical_reverse_joint_valid_check, 0, wx.ALL, 5
        )

        # 係数
        self.advance_vertical_reverse_joint_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("制限係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_vertical_reverse_joint_coefficient_txt.SetToolTip(
            logger.transtext("末端ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        )
        self.advance_vertical_reverse_joint_coefficient_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_head_sizer.Add(
            self.advance_vertical_reverse_joint_coefficient_txt, 0, wx.ALL, 5
        )

        self.advance_vertical_reverse_joint_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=50, initial=1, inc=0.1
        )
        self.advance_vertical_reverse_joint_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_head_sizer.Add(
            self.advance_vertical_reverse_joint_coefficient_spin, 0, wx.ALL, 5
        )

        self.advance_vertical_reverse_joint_sizer.Add(self.advance_vertical_reverse_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_vertical_reverse_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.vertical_reverse_joint_mov_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_x_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動X(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_x_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.vertical_reverse_joint_mov_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_y_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動Y(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_y_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.vertical_reverse_joint_mov_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_z_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動Z(最小)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_z_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.vertical_reverse_joint_mov_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_x_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動X(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_x_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.vertical_reverse_joint_mov_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_y_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動Y(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_y_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.vertical_reverse_joint_mov_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_mov_z_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの移動Z(最大)\n1ミクセルあたりの許容移動量"))
        self.vertical_reverse_joint_mov_z_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_mov_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_mov_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.vertical_reverse_joint_rot_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_x_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転X(最小)"))
        self.vertical_reverse_joint_rot_x_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.vertical_reverse_joint_rot_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_y_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転Y(最小)"))
        self.vertical_reverse_joint_rot_y_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.vertical_reverse_joint_rot_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_z_min_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転Z(最小)"))
        self.vertical_reverse_joint_rot_z_min_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.vertical_reverse_joint_rot_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_x_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転X(最大)"))
        self.vertical_reverse_joint_rot_x_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.vertical_reverse_joint_rot_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_y_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転Y(最大)"))
        self.vertical_reverse_joint_rot_y_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.vertical_reverse_joint_rot_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_rot_z_max_txt.SetToolTip(logger.transtext("末端縦逆ジョイントの回転Z(最大)"))
        self.vertical_reverse_joint_rot_z_max_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_rot_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_rot_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.vertical_reverse_joint_spring_mov_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_mov_x_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(移動X)"))
        self.vertical_reverse_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_x_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_mov_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_mov_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_x_spin, 0, wx.ALL, 5)

        # ばね(移動Y)
        self.vertical_reverse_joint_spring_mov_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_mov_y_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(移動Y)"))
        self.vertical_reverse_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_y_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_mov_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_mov_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_y_spin, 0, wx.ALL, 5)

        # ばね(移動Z)
        self.vertical_reverse_joint_spring_mov_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_mov_z_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(移動Z)"))
        self.vertical_reverse_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_z_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_mov_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_mov_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_mov_z_spin, 0, wx.ALL, 5)

        # ばね(回転X)
        self.vertical_reverse_joint_spring_rot_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_rot_x_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(回転X)"))
        self.vertical_reverse_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_x_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_rot_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_rot_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_x_spin, 0, wx.ALL, 5)

        # ばね(回転Y)
        self.vertical_reverse_joint_spring_rot_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_rot_y_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(回転Y)"))
        self.vertical_reverse_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_y_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_rot_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_rot_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_y_spin, 0, wx.ALL, 5)

        # ばね(回転Z)
        self.vertical_reverse_joint_spring_rot_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.vertical_reverse_joint_spring_rot_z_txt.SetToolTip(logger.transtext("末端縦逆ジョイントのばね(回転Z)"))
        self.vertical_reverse_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_z_txt, 0, wx.ALL, 5)

        self.vertical_reverse_joint_spring_rot_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.vertical_reverse_joint_spring_rot_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_vertical_reverse_joint_grid_sizer.Add(self.vertical_reverse_joint_spring_rot_z_spin, 0, wx.ALL, 5)

        self.advance_vertical_reverse_joint_sizer.Add(
            self.advance_vertical_reverse_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5
        )
        self.advance_param_sizer.Add(self.advance_vertical_reverse_joint_sizer, 0, wx.ALL, 5)

        # 横逆ジョイントブロック -------------------------------
        self.advance_horizonal_reverse_joint_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("横逆ジョイント")), orient=wx.VERTICAL
        )

        self.advance_horizonal_reverse_joint_head_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.advance_horizonal_reverse_joint_valid_check = wx.CheckBox(
            self.advance_window, wx.ID_ANY, logger.transtext("有効")
        )
        self.advance_horizonal_reverse_joint_valid_check.SetToolTip(logger.transtext("横逆ジョイントを有効にするか否か"))
        self.advance_horizonal_reverse_joint_valid_check.Bind(wx.EVT_CHECKBOX, self.on_horizonal_reverse_joint)
        self.advance_horizonal_reverse_joint_head_sizer.Add(
            self.advance_horizonal_reverse_joint_valid_check, 0, wx.ALL, 5
        )

        # 係数
        self.advance_horizonal_reverse_joint_coefficient_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("制限係数"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.advance_horizonal_reverse_joint_coefficient_txt.SetToolTip(
            logger.transtext("末端ジョイントが末端ジョイントよりどれくらい制限を強くするか。1の場合、全ての段の制限が均一になります。")
        )
        self.advance_horizonal_reverse_joint_coefficient_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_head_sizer.Add(
            self.advance_horizonal_reverse_joint_coefficient_txt, 0, wx.ALL, 5
        )

        self.advance_horizonal_reverse_joint_coefficient_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="1", min=1, max=50, initial=1, inc=0.1
        )
        self.advance_horizonal_reverse_joint_coefficient_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_head_sizer.Add(
            self.advance_horizonal_reverse_joint_coefficient_spin, 0, wx.ALL, 5
        )

        self.advance_horizonal_reverse_joint_sizer.Add(self.advance_horizonal_reverse_joint_head_sizer, 0, wx.ALL, 5)

        self.advance_horizonal_reverse_joint_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 移動X(最小)
        self.horizonal_reverse_joint_mov_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_x_min_txt.SetToolTip(logger.transtext("横逆ジョイントの移動X(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_x_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_x_min_spin, 0, wx.ALL, 5)

        # 移動Y(最小)
        self.horizonal_reverse_joint_mov_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_y_min_txt.SetToolTip(logger.transtext("横逆ジョイントの移動Y(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_y_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_y_min_spin, 0, wx.ALL, 5)

        # 移動Z(最小)
        self.horizonal_reverse_joint_mov_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_z_min_txt.SetToolTip(logger.transtext("横逆ジョイントの移動Z(最小)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_z_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_z_min_spin, 0, wx.ALL, 5)

        # 移動X(最大)
        self.horizonal_reverse_joint_mov_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_x_max_txt.SetToolTip(logger.transtext("横逆ジョイントの移動X(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_x_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_x_max_spin, 0, wx.ALL, 5)

        # 移動Y(最大)
        self.horizonal_reverse_joint_mov_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_y_max_txt.SetToolTip(logger.transtext("横逆ジョイントの移動Y(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_y_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_y_max_spin, 0, wx.ALL, 5)

        # 移動Z(最大)
        self.horizonal_reverse_joint_mov_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("移動Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_mov_z_max_txt.SetToolTip(logger.transtext("横逆ジョイントの移動Z(最大)\n1ミクセルあたりの許容移動量"))
        self.horizonal_reverse_joint_mov_z_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_mov_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_mov_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_mov_z_max_spin, 0, wx.ALL, 5)

        # 回転X(最小)
        self.horizonal_reverse_joint_rot_x_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_x_min_txt.SetToolTip(logger.transtext("横逆ジョイントの回転X(最小)"))
        self.horizonal_reverse_joint_rot_x_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_x_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_x_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_x_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_x_min_spin, 0, wx.ALL, 5)

        # 回転Y(最小)
        self.horizonal_reverse_joint_rot_y_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_y_min_txt.SetToolTip(logger.transtext("横逆ジョイントの回転Y(最小)"))
        self.horizonal_reverse_joint_rot_y_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_y_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_y_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_y_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_y_min_spin, 0, wx.ALL, 5)

        # 回転Z(最小)
        self.horizonal_reverse_joint_rot_z_min_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最小)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_z_min_txt.SetToolTip(logger.transtext("横逆ジョイントの回転Z(最小)"))
        self.horizonal_reverse_joint_rot_z_min_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_z_min_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_z_min_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1000, max=1, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_z_min_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_z_min_spin, 0, wx.ALL, 5)

        # 回転X(最大)
        self.horizonal_reverse_joint_rot_x_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転X(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_x_max_txt.SetToolTip(logger.transtext("横逆ジョイントの回転X(最大)"))
        self.horizonal_reverse_joint_rot_x_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_x_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_x_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_x_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_x_max_spin, 0, wx.ALL, 5)

        # 回転Y(最大)
        self.horizonal_reverse_joint_rot_y_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Y(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_y_max_txt.SetToolTip(logger.transtext("横逆ジョイントの回転Y(最大)"))
        self.horizonal_reverse_joint_rot_y_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_y_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_y_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_y_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_y_max_spin, 0, wx.ALL, 5)

        # 回転Z(最大)
        self.horizonal_reverse_joint_rot_z_max_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("回転Z(最大)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_rot_z_max_txt.SetToolTip(logger.transtext("横逆ジョイントの回転Z(最大)"))
        self.horizonal_reverse_joint_rot_z_max_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_z_max_txt, 0, wx.ALL, 5)

        self.horizonal_reverse_joint_rot_z_max_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=-1, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_rot_z_max_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(self.horizonal_reverse_joint_rot_z_max_spin, 0, wx.ALL, 5)

        # ばね(移動X)
        self.horizonal_reverse_joint_spring_mov_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_mov_x_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(移動X)"))
        self.horizonal_reverse_joint_spring_mov_x_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_x_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_mov_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_mov_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_x_spin, 0, wx.ALL, 5
        )

        # ばね(移動Y)
        self.horizonal_reverse_joint_spring_mov_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_mov_y_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(移動Y)"))
        self.horizonal_reverse_joint_spring_mov_y_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_y_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_mov_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_mov_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_y_spin, 0, wx.ALL, 5
        )

        # ばね(移動Z)
        self.horizonal_reverse_joint_spring_mov_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(移動Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_mov_z_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(移動Z)"))
        self.horizonal_reverse_joint_spring_mov_z_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_z_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_mov_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_mov_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_mov_z_spin, 0, wx.ALL, 5
        )

        # ばね(回転X)
        self.horizonal_reverse_joint_spring_rot_x_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転X)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_rot_x_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(回転X)"))
        self.horizonal_reverse_joint_spring_rot_x_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_x_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_rot_x_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_rot_x_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_x_spin, 0, wx.ALL, 5
        )

        # ばね(回転Y)
        self.horizonal_reverse_joint_spring_rot_y_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Y)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_rot_y_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(回転Y)"))
        self.horizonal_reverse_joint_spring_rot_y_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_y_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_rot_y_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_rot_y_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_y_spin, 0, wx.ALL, 5
        )

        # ばね(回転Z)
        self.horizonal_reverse_joint_spring_rot_z_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ばね(回転Z)"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.horizonal_reverse_joint_spring_rot_z_txt.SetToolTip(logger.transtext("横逆ジョイントのばね(回転Z)"))
        self.horizonal_reverse_joint_spring_rot_z_txt.Wrap(-1)
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_z_txt, 0, wx.ALL, 5
        )

        self.horizonal_reverse_joint_spring_rot_z_spin = wx.SpinCtrlDouble(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(90, -1), value="0", min=0, max=1000, initial=0, inc=0.1
        )
        self.horizonal_reverse_joint_spring_rot_z_spin.Bind(
            wx.EVT_MOUSEWHEEL, lambda event: self.on_wheel_double_spin_ctrl(event, 0.1)
        )
        self.advance_horizonal_reverse_joint_grid_sizer.Add(
            self.horizonal_reverse_joint_spring_rot_z_spin, 0, wx.ALL, 5
        )

        self.advance_horizonal_reverse_joint_sizer.Add(
            self.advance_horizonal_reverse_joint_grid_sizer, 1, wx.ALL | wx.EXPAND, 5
        )
        self.advance_param_sizer.Add(self.advance_horizonal_reverse_joint_sizer, 0, wx.ALL, 5)

        # 詳細オプションブロック -------------------------------
        self.advance_option_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.advance_window, wx.ID_ANY, logger.transtext("詳細オプション")), orient=wx.VERTICAL
        )
        self.advance_option_grid_sizer = wx.FlexGridSizer(0, 6, 0, 0)

        # 物理接続タイプ
        self.parent_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("物理接続"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.parent_type_txt.SetToolTip(logger.transtext("中心: 中心ボーンを経由して繋ぐ\n親: 親ボーンに直接繋ぐ（※要親ボーンのボーン追従剛体）"))
        self.parent_type_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.parent_type_txt, 0, wx.ALL, 5)

        self.parent_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("中心"), logger.transtext("親")],
        )
        self.parent_type_ctrl.SetToolTip(self.parent_type_txt.GetToolTipText())
        self.parent_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.parent_type_ctrl, 0, wx.ALL, 5)

        # 物理タイプ
        self.physics_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("物理タイプ"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.physics_type_txt.SetToolTip(
            logger.transtext("布: 板剛体で縦横を繋ぐ\n髪: カプセル剛体で縦を繋ぐ(※要ボーン定義)\n袖: カプセル剛体で縦を繋ぐ(※要ボーン定義)")
        )
        self.physics_type_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.physics_type_txt, 0, wx.ALL, 5)

        self.physics_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("布"), logger.transtext("髪"), logger.transtext("胸"), logger.transtext("単一揺")],
        )
        self.physics_type_ctrl.SetToolTip(self.physics_type_txt.GetToolTipText())
        self.physics_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.physics_type_ctrl, 0, wx.ALL, 5)

        # ジョイント位置タイプ
        self.joint_pos_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ジョイント位置"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.joint_pos_type_txt.SetToolTip(logger.transtext("ボーン位置: ボーンの位置にジョイントが入る\nボーン間: ボーンとボーンの間にジョイントが入る"))
        self.joint_pos_type_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.joint_pos_type_txt, 0, wx.ALL, 5)

        self.joint_pos_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("ボーン位置"), logger.transtext("ボーン間")],
        )
        self.joint_pos_type_ctrl.SetToolTip(self.joint_pos_type_txt.GetToolTipText())
        self.joint_pos_type_ctrl.Bind(wx.EVT_CHOICE, self.on_change_joint_pos_type)
        self.advance_option_grid_sizer.Add(self.joint_pos_type_ctrl, 0, wx.ALL, 5)

        # 頂点ルート探索手法タイプ
        self.route_search_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("ルート探索"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.route_search_type_txt.SetToolTip(
            logger.transtext("末端頂点から根元頂点を走査する際の手法\n前頂点優先: 前頂点との角度差を優先\n根元頂点優先: 推定根元頂点への向きを優先")
        )
        self.route_search_type_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.route_search_type_txt, 0, wx.ALL, 5)

        self.route_search_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("前頂点優先"), logger.transtext("根元頂点優先")],
        )
        self.route_search_type_ctrl.SetToolTip(self.route_search_type_txt.GetToolTipText())
        self.route_search_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.route_search_type_ctrl, 0, wx.ALL, 5)

        # 根元頂点推定手法
        self.route_estimate_type_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("根元頂点推定"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.route_estimate_type_txt.SetToolTip(
            logger.transtext("根元頂点を推定する際の手法\n角度: 末端頂点角度に類似した根元頂点を推定\n縮尺: 末端頂点円周と根元頂点円周の縮尺を推定\nリング: 筒状メッシュの横方向の繋がりから推定")
        )
        self.route_estimate_type_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.route_estimate_type_txt, 0, wx.ALL, 5)

        self.route_estimate_type_ctrl = wx.Choice(
            self.advance_window,
            id=wx.ID_ANY,
            choices=[logger.transtext("角度"), logger.transtext("縮尺"), logger.transtext("リング")],
        )
        self.route_estimate_type_ctrl.SetToolTip(self.route_estimate_type_txt.GetToolTipText())
        self.route_estimate_type_ctrl.Bind(wx.EVT_CHOICE, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.route_estimate_type_ctrl, 0, wx.ALL, 5)

        # 物理親
        physics_parent_tooltip = logger.transtext("物理的に親となる物理設定番号。物理親を設定すると設定した物理の上端が物理親の下端に紐付けられます。")
        self.physics_parent_txt = wx.StaticText(
            self.advance_window, wx.ID_ANY, logger.transtext("物理親"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.physics_parent_txt.SetToolTip(physics_parent_tooltip)
        self.physics_parent_txt.Wrap(-1)
        self.advance_option_grid_sizer.Add(self.physics_parent_txt, 0, wx.ALL, 5)

        self.physics_parent_spin = wx.SpinCtrl(
            self.advance_window, id=wx.ID_ANY, size=wx.Size(50, -1), value="0", min=0, max=20, initial=0
        )
        self.physics_parent_spin.SetToolTip(physics_parent_tooltip)
        self.physics_parent_spin.Bind(wx.EVT_SPINCTRL, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.physics_parent_spin, 0, wx.ALL, 5)

        # 詳細コメント
        self.advance_comment_ctrl = wx.CheckBox(self.advance_window, wx.ID_ANY, logger.transtext("詳細コメント"))
        self.advance_comment_ctrl.SetToolTip(logger.transtext("物理の詳細な挙動をコメントに出力したい場合にチェックしてください"))
        self.advance_comment_ctrl.Bind(wx.EVT_CHECKBOX, self.main_frame.file_panel_ctrl.on_change_file)
        self.advance_option_grid_sizer.Add(self.advance_comment_ctrl, 0, wx.ALL, 5)

        # グリッドを登録
        self.advance_option_sizer.Add(self.advance_option_grid_sizer, 1, wx.ALL | wx.EXPAND, 5)
        self.advance_param_sizer.Add(self.advance_option_sizer, 0, wx.ALL, 5)

        # ボーン版 ------------------
        self.bone_sizer = wx.StaticBoxSizer(
            wx.StaticBox(self.bone_window, wx.ID_ANY, "【No.{0}】".format(param_no + 1)), orient=wx.VERTICAL
        )
        self.bone_param_sizer = wx.BoxSizer(wx.VERTICAL)

        self.bone_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bone_btn_sizer.AddSpacer(250)

        # インポートボタン
        self.bone_import_btn_ctrl = wx.Button(
            self.bone_window, wx.ID_ANY, logger.transtext("設定クリア"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.bone_import_btn_ctrl.SetToolTip(logger.transtext("ボーン設定データ全てクリアします。"))
        self.bone_import_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_bone_clear)
        self.bone_btn_sizer.Add(self.bone_import_btn_ctrl, 0, wx.ALL, 5)

        # インポートボタン
        self.bone_import_btn_ctrl = wx.Button(
            self.bone_window, wx.ID_ANY, logger.transtext("インポート ..."), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.bone_import_btn_ctrl.SetToolTip(logger.transtext("ボーン設定データをcsvファイルから読み込みます。\nファイル選択ダイアログが開きます。"))
        self.bone_import_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_bone_import)
        self.bone_btn_sizer.Add(self.bone_import_btn_ctrl, 0, wx.ALL, 5)

        # エクスポートボタン
        self.bone_export_btn_ctrl = wx.Button(
            self.bone_window, wx.ID_ANY, logger.transtext("エクスポート ..."), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.bone_export_btn_ctrl.SetToolTip(logger.transtext("ボーン設定データをcsvファイルに出力します。\n出力先を指定できます。"))
        self.bone_export_btn_ctrl.Bind(wx.EVT_BUTTON, self.on_bone_export)
        self.bone_btn_sizer.Add(self.bone_export_btn_ctrl, 0, wx.ALL, 5)

        self.bone_param_sizer.Add(self.bone_btn_sizer, 0, wx.ALL, 0)

        self.bone_material_ctrl = wx.StaticText(
            self.bone_window, wx.ID_ANY, logger.transtext("（材質未選択）"), wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.bone_param_sizer.Add(self.bone_material_ctrl, 0, wx.ALL, 5)

        self.bone_sizer.Add(self.bone_param_sizer, 1, wx.ALL | wx.EXPAND, 0)

        self.bone_grid_sizer = wx.BoxSizer(wx.VERTICAL)

        self.bone_sizer.Add(self.bone_grid_sizer, 1, wx.ALL | wx.EXPAND, 0)

    def get_param_options(self, pidx: int, is_show_error):
        params = {}

        if (
            self.simple_material_ctrl.GetStringSelection()
            and self.simple_parent_bone_ctrl.GetStringSelection()
            and self.simple_group_ctrl.GetStringSelection()
            and self.simple_abb_ctrl.GetValue()
        ):
            if (
                self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.material_indices[
                    self.simple_material_ctrl.GetStringSelection()
                ]
                == 0
            ):
                logger.error("頂点のない材質が指定されています。", decoration=MLogger.DECORATION_BOX)
                return params, False

            if (
                not self.vertices_csv_file_ctrl.path()
                and self.simple_material_ctrl.GetStringSelection()
                == self.simple_back_material_ctrl.GetStringSelection()
            ):
                logger.error(
                    "物理材質と同じ材質が裏面に指定されています。\n物理頂点CSVが指定されていない場合、物理材質と裏面材質は同じ材質を指定する事ができません。",
                    decoration=MLogger.DECORATION_BOX,
                )
                return params, False

            if self.physics_type_ctrl.GetStringSelection() == logger.transtext(
                "髪"
            ) and self.simple_exist_physics_clear_ctrl.GetStringSelection() != logger.transtext("再利用"):
                logger.error("髪物理を設定する時には、既存設定は「再利用」を指定してください。", decoration=MLogger.DECORATION_BOX)
                return params, False

            if self.physics_type_ctrl.GetStringSelection() == logger.transtext(
                "単一揺"
            ) and self.simple_exist_physics_clear_ctrl.GetStringSelection() != logger.transtext("再利用"):
                logger.error("単一揺れ物物理を設定する時には、既存設定は「再利用」を指定してください。", decoration=MLogger.DECORATION_BOX)
                return params, False

            bone_grid, bone_grid_rows, bone_grid_cols, is_boned = self.get_bone_grid()
            if self.simple_exist_physics_clear_ctrl.GetStringSelection() == logger.transtext("再利用"):
                if not is_boned:
                    logger.error("既存設定を再利用する場合、「パラ調整(ボーン)」画面で有効なボーン並び順を指定してください。", decoration=MLogger.DECORATION_BOX)
                    return params, False
                params["bone_grid"] = bone_grid
                params["bone_grid_rows"] = bone_grid_rows
                params["bone_grid_cols"] = bone_grid_cols
            else:
                if self.simple_exist_physics_clear_ctrl.GetStringSelection() != logger.transtext("再利用") and is_boned:
                    logger.error("「パラ調整(ボーン)」画面でボーン並び順を指定した場合、既存設定は「再利用」を指定してください。", decoration=MLogger.DECORATION_BOX)
                    return params, False
                params["bone_grid"] = {}
                params["bone_grid_rows"] = 0
                params["bone_grid_cols"] = 0

            if (
                self.simple_exist_physics_clear_ctrl.GetStringSelection() == logger.transtext("再利用")
                and bone_grid
                and not is_boned
            ):
                return params, False

            if self.vertices_csv_file_ctrl.path() and not os.path.exists(self.vertices_csv_file_ctrl.path()):
                logger.error("物理対象頂点CSVファイルが存在しません", decoration=MLogger.DECORATION_BOX)
                return params, False

            if (
                self.vertices_back_csv_file_ctrl.path()
                and "*" not in self.vertices_back_csv_file_ctrl.path()
                and not os.path.exists(self.vertices_back_csv_file_ctrl.path())
            ):
                # *がついてない場合だけチェック
                logger.error("裏面対象頂点CSVファイルが存在しません", decoration=MLogger.DECORATION_BOX)
                return params, False

            if self.top_vertices_csv_file_ctrl.path() and not os.path.exists(self.top_vertices_csv_file_ctrl.path()):
                logger.error("根元頂点CSVファイルが存在しません", decoration=MLogger.DECORATION_BOX)
                return params, False

            self.vertices_csv_file_ctrl.save()
            self.vertices_edge_csv_file_ctrl.save()
            self.vertices_back_csv_file_ctrl.save()
            self.top_vertices_csv_file_ctrl.save()

            if self.simple_abb_ctrl.GetValue() not in self.main_frame.file_hitories["abb_setting"]:
                self.main_frame.file_hitories["abb_setting"][self.simple_abb_ctrl.GetValue()] = {}
            self.main_frame.file_hitories["abb_setting"][self.simple_abb_ctrl.GetValue()] = {
                "material_name": self.simple_material_ctrl.GetStringSelection(),
                "parent_bone_name": self.simple_parent_bone_ctrl.GetStringSelection(),
                "group": self.simple_group_ctrl.GetStringSelection(),
                "direction": self.simple_direction_ctrl.GetStringSelection(),
                "exist_physics_clear": self.simple_exist_physics_clear_ctrl.GetStringSelection(),
                "primitive": self.simple_primitive_ctrl.GetStringSelection(),
                "special_shape": self.simple_special_shape_ctrl.GetStringSelection(),
                "back_material_name": self.simple_back_material_ctrl.GetStringSelection(),
                "back_extend_material_names": ",".join(
                    self.frame.material_list[cidx + 1] for cidx in self.simple_extend_back_choice_ctrl.GetSelections()
                ),
                "edge_material_name": self.simple_edge_material_ctrl.GetStringSelection(),
                "edge_extend_material_names": ",".join(
                    self.frame.material_list[cidx + 1] for cidx in self.simple_extend_edge_choice_ctrl.GetSelections()
                ),
                "vertices_csv": self.vertices_csv_file_ctrl.path(),
                "vertices_edge_csv": self.vertices_edge_csv_file_ctrl.path(),
                "vertices_back_csv": self.vertices_back_csv_file_ctrl.path(),
                "top_vertices_csv": self.top_vertices_csv_file_ctrl.path(),
                "physics_parent": self.physics_parent_spin.GetValue(),
                "parent_type": self.parent_type_ctrl.GetStringSelection(),
                "physics_type": self.physics_type_ctrl.GetStringSelection(),
                "density_type": self.density_type_ctrl.GetStringSelection(),
                "joint_pos_type": self.joint_pos_type_ctrl.GetStringSelection(),
                "route_search_type": self.route_search_type_ctrl.GetStringSelection(),
                "route_estimate_type": self.route_estimate_type_ctrl.GetStringSelection(),
                "params": self.get_param_export_data(),
            }
            self.main_frame.file_hitories["advance_comment"] = self.advance_comment_ctrl.GetValue()
            MFileUtils.save_history(self.main_frame.mydir_path, self.main_frame.file_hitories)

            # 簡易版オプションデータ -------------
            params["material_name"] = self.simple_material_ctrl.GetStringSelection()
            params["back_material_name"] = self.simple_back_material_ctrl.GetStringSelection()
            params["back_extend_material_names"] = [
                self.frame.material_list[cidx + 1] for cidx in self.simple_extend_back_choice_ctrl.GetSelections()
            ]
            params["edge_material_name"] = self.simple_edge_material_ctrl.GetStringSelection()
            params["edge_extend_material_names"] = [
                self.frame.material_list[cidx + 1] for cidx in self.simple_extend_edge_choice_ctrl.GetSelections()
            ]
            params["parent_bone_name"] = self.simple_parent_bone_ctrl.GetStringSelection()
            params["abb_name"] = self.simple_abb_ctrl.GetValue()
            params["direction"] = self.simple_direction_ctrl.GetStringSelection()
            params["exist_physics_clear"] = self.simple_exist_physics_clear_ctrl.GetStringSelection()
            params["special_shape"] = self.simple_special_shape_ctrl.GetStringSelection()
            params["vertices_csv"] = self.vertices_csv_file_ctrl.path()
            params["vertices_edge_csv"] = self.vertices_edge_csv_file_ctrl.path()
            params["vertices_back_csv"] = self.vertices_back_csv_file_ctrl.path()
            params["top_vertices_csv"] = self.top_vertices_csv_file_ctrl.path()
            # params["threshold"] = self.simple_threshold_slider.GetValue()
            # params["fineness"] = self.simple_fineness_slider.GetValue()
            params["mass"] = self.simple_mass_slider.GetValue()
            params["air_resistance"] = self.simple_air_resistance_slider.GetValue()
            params["shape_maintenance"] = self.simple_shape_maintenance_slider.GetValue()

            # 詳細版オプションデータ -------------
            params["vertical_bone_density"] = int(self.vertical_bone_density_spin.GetValue())
            params["horizonal_bone_density"] = int(self.horizonal_bone_density_spin.GetValue())
            params["horizonal_bone_offset"] = int(self.horizonal_bone_offset_spin.GetValue())
            # params["bone_thinning_out"] = self.bone_thinning_out_check.GetValue()
            params["bone_thinning_out"] = False
            params["physics_type"] = self.physics_type_ctrl.GetStringSelection()
            params["density_type"] = self.density_type_ctrl.GetStringSelection()
            params["physics_parent"] = int(self.physics_parent_spin.GetValue())
            params["parent_type"] = self.parent_type_ctrl.GetStringSelection()
            params["joint_pos_type"] = self.joint_pos_type_ctrl.GetStringSelection()
            params["route_search_type"] = self.route_search_type_ctrl.GetStringSelection()
            params["route_estimate_type"] = self.route_estimate_type_ctrl.GetStringSelection()
            params["advance_comment"] = self.advance_comment_ctrl.GetValue()

            # 自身を非衝突対象
            no_collision_group = 0
            for nc in range(16):
                if nc not in [int(self.simple_group_ctrl.GetStringSelection()) - 1]:
                    no_collision_group |= 1 << nc

            params["rigidbody"] = RigidBody(
                "",
                "",
                self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.bones[
                    self.simple_parent_bone_ctrl.GetStringSelection()
                ].index,
                int(self.simple_group_ctrl.GetStringSelection()) - 1,
                no_collision_group,
                self.advance_rigidbody_shape_type_ctrl.GetSelection(),
                MVector3D(),
                MVector3D(),
                MVector3D(),
                self.rigidbody_mass_spin.GetValue(),
                self.rigidbody_linear_damping_spin.GetValue(),
                self.rigidbody_angular_damping_spin.GetValue(),
                self.rigidbody_restitution_spin.GetValue(),
                self.rigidbody_friction_spin.GetValue(),
                0,
            )
            params["rigidbody_coefficient"] = self.rigidbody_coefficient_spin.GetValue()
            params["rigidbody_shape_type"] = self.advance_rigidbody_shape_type_ctrl.GetSelection()
            params["rigidbody_balancer"] = self.advance_rigidbody_balancer_ctrl.GetValue()
            params["rigidbody_leg"] = self.advance_rigidbody_leg_ctrl.GetValue()

            params["rigidbody_root_thicks"] = self.rigidbody_root_thicks_spin.GetValue()
            params["rigidbody_end_thicks"] = self.rigidbody_end_thicks_spin.GetValue()
            params["rigidbody_cover_coefficient"] = self.rigidbody_cover_coefficient_spin.GetValue()

            params["vertical_joint"] = None
            if self.advance_vertical_joint_valid_check.GetValue():
                params["vertical_joint"] = Joint(
                    "",
                    "",
                    -1,
                    -1,
                    -1,
                    MVector3D(),
                    MVector3D(),
                    MVector3D(
                        self.vertical_joint_mov_x_min_spin.GetValue(),
                        self.vertical_joint_mov_y_min_spin.GetValue(),
                        self.vertical_joint_mov_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_joint_mov_x_max_spin.GetValue(),
                        self.vertical_joint_mov_y_max_spin.GetValue(),
                        self.vertical_joint_mov_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_joint_rot_x_min_spin.GetValue(),
                        self.vertical_joint_rot_y_min_spin.GetValue(),
                        self.vertical_joint_rot_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_joint_rot_x_max_spin.GetValue(),
                        self.vertical_joint_rot_y_max_spin.GetValue(),
                        self.vertical_joint_rot_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_joint_spring_mov_x_spin.GetValue(),
                        self.vertical_joint_spring_mov_y_spin.GetValue(),
                        self.vertical_joint_spring_mov_z_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_joint_spring_rot_x_spin.GetValue(),
                        self.vertical_joint_spring_rot_y_spin.GetValue(),
                        self.vertical_joint_spring_rot_z_spin.GetValue(),
                    ),
                )
            params["vertical_joint_coefficient"] = self.advance_vertical_joint_coefficient_spin.GetValue()

            params["horizonal_joint"] = None
            if self.advance_horizonal_joint_valid_check.GetValue():
                params["horizonal_joint"] = Joint(
                    "",
                    "",
                    -1,
                    -1,
                    -1,
                    MVector3D(),
                    MVector3D(),
                    MVector3D(
                        self.horizonal_joint_mov_x_min_spin.GetValue(),
                        self.horizonal_joint_mov_y_min_spin.GetValue(),
                        self.horizonal_joint_mov_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_joint_mov_x_max_spin.GetValue(),
                        self.horizonal_joint_mov_y_max_spin.GetValue(),
                        self.horizonal_joint_mov_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_joint_rot_x_min_spin.GetValue(),
                        self.horizonal_joint_rot_y_min_spin.GetValue(),
                        self.horizonal_joint_rot_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_joint_rot_x_max_spin.GetValue(),
                        self.horizonal_joint_rot_y_max_spin.GetValue(),
                        self.horizonal_joint_rot_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_joint_spring_mov_x_spin.GetValue(),
                        self.horizonal_joint_spring_mov_y_spin.GetValue(),
                        self.horizonal_joint_spring_mov_z_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_joint_spring_rot_x_spin.GetValue(),
                        self.horizonal_joint_spring_rot_y_spin.GetValue(),
                        self.horizonal_joint_spring_rot_z_spin.GetValue(),
                    ),
                )
            params["horizonal_joint_restruct"] = self.advance_horizonal_joint_restruct_check.GetValue()
            params["horizonal_joint_coefficient"] = self.advance_horizonal_joint_coefficient_spin.GetValue()

            params["diagonal_joint"] = None
            if self.advance_diagonal_joint_valid_check.GetValue():
                params["diagonal_joint"] = Joint(
                    "",
                    "",
                    -1,
                    -1,
                    -1,
                    MVector3D(),
                    MVector3D(),
                    MVector3D(
                        self.diagonal_joint_mov_x_min_spin.GetValue(),
                        self.diagonal_joint_mov_y_min_spin.GetValue(),
                        self.diagonal_joint_mov_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.diagonal_joint_mov_x_max_spin.GetValue(),
                        self.diagonal_joint_mov_y_max_spin.GetValue(),
                        self.diagonal_joint_mov_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.diagonal_joint_rot_x_min_spin.GetValue(),
                        self.diagonal_joint_rot_y_min_spin.GetValue(),
                        self.diagonal_joint_rot_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.diagonal_joint_rot_x_max_spin.GetValue(),
                        self.diagonal_joint_rot_y_max_spin.GetValue(),
                        self.diagonal_joint_rot_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.diagonal_joint_spring_mov_x_spin.GetValue(),
                        self.diagonal_joint_spring_mov_y_spin.GetValue(),
                        self.diagonal_joint_spring_mov_z_spin.GetValue(),
                    ),
                    MVector3D(
                        self.diagonal_joint_spring_rot_x_spin.GetValue(),
                        self.diagonal_joint_spring_rot_y_spin.GetValue(),
                        self.diagonal_joint_spring_rot_z_spin.GetValue(),
                    ),
                )
            params["diagonal_joint_coefficient"] = self.advance_diagonal_joint_coefficient_spin.GetValue()

            params["vertical_reverse_joint"] = None
            if self.advance_vertical_reverse_joint_valid_check.GetValue():
                params["vertical_reverse_joint"] = Joint(
                    "",
                    "",
                    -1,
                    -1,
                    -1,
                    MVector3D(),
                    MVector3D(),
                    MVector3D(
                        self.vertical_reverse_joint_mov_x_min_spin.GetValue(),
                        self.vertical_reverse_joint_mov_y_min_spin.GetValue(),
                        self.vertical_reverse_joint_mov_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_reverse_joint_mov_x_max_spin.GetValue(),
                        self.vertical_reverse_joint_mov_y_max_spin.GetValue(),
                        self.vertical_reverse_joint_mov_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_reverse_joint_rot_x_min_spin.GetValue(),
                        self.vertical_reverse_joint_rot_y_min_spin.GetValue(),
                        self.vertical_reverse_joint_rot_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_reverse_joint_rot_x_max_spin.GetValue(),
                        self.vertical_reverse_joint_rot_y_max_spin.GetValue(),
                        self.vertical_reverse_joint_rot_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_reverse_joint_spring_mov_x_spin.GetValue(),
                        self.vertical_reverse_joint_spring_mov_y_spin.GetValue(),
                        self.vertical_reverse_joint_spring_mov_z_spin.GetValue(),
                    ),
                    MVector3D(
                        self.vertical_reverse_joint_spring_rot_x_spin.GetValue(),
                        self.vertical_reverse_joint_spring_rot_y_spin.GetValue(),
                        self.vertical_reverse_joint_spring_rot_z_spin.GetValue(),
                    ),
                )
            params[
                "vertical_reverse_joint_coefficient"
            ] = self.advance_vertical_reverse_joint_coefficient_spin.GetValue()

            params["horizonal_reverse_joint"] = None
            if self.advance_horizonal_reverse_joint_valid_check.GetValue():
                params["horizonal_reverse_joint"] = Joint(
                    "",
                    "",
                    -1,
                    -1,
                    -1,
                    MVector3D(),
                    MVector3D(),
                    MVector3D(
                        self.horizonal_reverse_joint_mov_x_min_spin.GetValue(),
                        self.horizonal_reverse_joint_mov_y_min_spin.GetValue(),
                        self.horizonal_reverse_joint_mov_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_reverse_joint_mov_x_max_spin.GetValue(),
                        self.horizonal_reverse_joint_mov_y_max_spin.GetValue(),
                        self.horizonal_reverse_joint_mov_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_reverse_joint_rot_x_min_spin.GetValue(),
                        self.horizonal_reverse_joint_rot_y_min_spin.GetValue(),
                        self.horizonal_reverse_joint_rot_z_min_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_reverse_joint_rot_x_max_spin.GetValue(),
                        self.horizonal_reverse_joint_rot_y_max_spin.GetValue(),
                        self.horizonal_reverse_joint_rot_z_max_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_reverse_joint_spring_mov_x_spin.GetValue(),
                        self.horizonal_reverse_joint_spring_mov_y_spin.GetValue(),
                        self.horizonal_reverse_joint_spring_mov_z_spin.GetValue(),
                    ),
                    MVector3D(
                        self.horizonal_reverse_joint_spring_rot_x_spin.GetValue(),
                        self.horizonal_reverse_joint_spring_rot_y_spin.GetValue(),
                        self.horizonal_reverse_joint_spring_rot_z_spin.GetValue(),
                    ),
                )
            params[
                "horizonal_reverse_joint_coefficient"
            ] = self.advance_horizonal_reverse_joint_coefficient_spin.GetValue()
        else:
            if is_show_error:
                empty_param_list = []
                if not self.simple_material_ctrl.GetStringSelection():
                    empty_param_list.append(logger.transtext("材質名"))
                if not self.simple_parent_bone_ctrl.GetStringSelection():
                    empty_param_list.append(logger.transtext("親ボーン名"))
                if not self.simple_group_ctrl.GetStringSelection():
                    empty_param_list.append(logger.transtext("剛体グループ"))
                if not self.simple_abb_ctrl.GetValue():
                    empty_param_list.append(logger.transtext("材質略称"))

                if len(empty_param_list) < 4:
                    logger.error(
                        "No.%sの%sに値が設定されていません。",
                        pidx + 1,
                        "・".join(empty_param_list),
                        decoration=MLogger.DECORATION_BOX,
                    )
                    return params, False

        return params, True

    def on_change_vertices_csv(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)

    def on_param_import(self, event: wx.Event):
        with wx.FileDialog(
            self.frame,
            logger.transtext("材質物理設定JSONを読み込む"),
            wildcard="JSONファイル (*.json)|*.json|すべてのファイル (*.*)|*.*",
            defaultDir=os.path.dirname(self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.path),
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # Proceed loading the file chosen by the user
            target_physics_path = fileDialog.GetPath()
            try:
                with open(target_physics_path, "r") as f:
                    params = json.load(f)
                    self.set_param_import_data(params)

                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("材質物理設定JSONのインポートに成功しました \n{0}").format(target_physics_path),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()
            except Exception:
                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("材質物理設定JSONが読み込めませんでした '{0}'\n\n{1}").format(
                        target_physics_path, traceback.format_exc()
                    ),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()

    def set_param_import_data(self, params: dict):
        # 簡易版オプションデータ -------------
        # self.simple_threshold_slider.SetValue(params.get("threshold", 0.07))
        # self.simple_fineness_slider.SetValue(params.get("fineness", 3.4))
        self.simple_mass_slider.SetValue(params.get("mass", 0.5))
        self.simple_air_resistance_slider.SetValue(params.get("air_resistance", 1.8))
        self.simple_shape_maintenance_slider.SetValue(params.get("shape_maintenance", 1.5))

        # 詳細版オプションデータ -------------
        self.vertical_bone_density_spin.SetValue(params["vertical_bone_density"])
        self.horizonal_bone_density_spin.SetValue(params["horizonal_bone_density"])
        self.horizonal_bone_offset_spin.SetValue(params.get("horizonal_bone_offset", 0))

        # 剛体 ---------------
        self.advance_rigidbody_shape_type_ctrl.SetSelection(params["rigidbody_shape_type"])
        self.rigidbody_mass_spin.SetValue(params["rigidbody_mass"])
        self.rigidbody_linear_damping_spin.SetValue(params["rigidbody_linear_damping"])
        self.rigidbody_angular_damping_spin.SetValue(params["rigidbody_angular_damping"])
        self.rigidbody_restitution_spin.SetValue(params["rigidbody_restitution"])
        self.rigidbody_friction_spin.SetValue(params["rigidbody_friction"])
        self.rigidbody_coefficient_spin.SetValue(params["rigidbody_coefficient"])
        self.advance_rigidbody_balancer_ctrl.SetValue(params.get("rigidbody_balancer", 0))
        self.advance_rigidbody_leg_ctrl.SetValue(params.get("rigidbody_leg", 0))

        self.rigidbody_root_thicks_spin.SetValue(params.get("rigidbody_root_thicks", 0.07))
        self.rigidbody_end_thicks_spin.SetValue(params.get("rigidbody_end_thicks", 0.3))
        self.rigidbody_cover_coefficient_spin.SetValue(params.get("rigidbody_cover_coefficient", 1.2))

        # 縦ジョイント -----------
        self.advance_vertical_joint_valid_check.SetValue(params["vertical_joint_valid"])
        self.vertical_joint_mov_x_min_spin.SetValue(params["vertical_joint_mov_x_min"])
        self.vertical_joint_mov_y_min_spin.SetValue(params["vertical_joint_mov_y_min"])
        self.vertical_joint_mov_z_min_spin.SetValue(params["vertical_joint_mov_z_min"])
        self.vertical_joint_mov_x_max_spin.SetValue(params["vertical_joint_mov_x_max"])
        self.vertical_joint_mov_y_max_spin.SetValue(params["vertical_joint_mov_y_max"])
        self.vertical_joint_mov_z_max_spin.SetValue(params["vertical_joint_mov_z_max"])
        self.vertical_joint_rot_x_min_spin.SetValue(params["vertical_joint_rot_x_min"])
        self.vertical_joint_rot_y_min_spin.SetValue(params["vertical_joint_rot_y_min"])
        self.vertical_joint_rot_z_min_spin.SetValue(params["vertical_joint_rot_z_min"])
        self.vertical_joint_rot_x_max_spin.SetValue(params["vertical_joint_rot_x_max"])
        self.vertical_joint_rot_y_max_spin.SetValue(params["vertical_joint_rot_y_max"])
        self.vertical_joint_rot_z_max_spin.SetValue(params["vertical_joint_rot_z_max"])
        self.vertical_joint_spring_mov_x_spin.SetValue(params["vertical_joint_spring_mov_x"])
        self.vertical_joint_spring_mov_y_spin.SetValue(params["vertical_joint_spring_mov_y"])
        self.vertical_joint_spring_mov_z_spin.SetValue(params["vertical_joint_spring_mov_z"])
        self.vertical_joint_spring_rot_x_spin.SetValue(params["vertical_joint_spring_rot_x"])
        self.vertical_joint_spring_rot_y_spin.SetValue(params["vertical_joint_spring_rot_y"])
        self.vertical_joint_spring_rot_z_spin.SetValue(params["vertical_joint_spring_rot_z"])
        self.advance_vertical_joint_coefficient_spin.SetValue(params["vertical_joint_coefficient"])
        self.on_vertical_joint(wx.EVT_CHECKBOX)

        # 横ジョイント -----------
        self.advance_horizonal_joint_valid_check.SetValue(params["horizonal_joint_valid"])
        self.horizonal_joint_mov_x_min_spin.SetValue(params["horizonal_joint_mov_x_min"])
        self.horizonal_joint_mov_y_min_spin.SetValue(params["horizonal_joint_mov_y_min"])
        self.horizonal_joint_mov_z_min_spin.SetValue(params["horizonal_joint_mov_z_min"])
        self.horizonal_joint_mov_x_max_spin.SetValue(params["horizonal_joint_mov_x_max"])
        self.horizonal_joint_mov_y_max_spin.SetValue(params["horizonal_joint_mov_y_max"])
        self.horizonal_joint_mov_z_max_spin.SetValue(params["horizonal_joint_mov_z_max"])
        self.horizonal_joint_rot_x_min_spin.SetValue(params["horizonal_joint_rot_x_min"])
        self.horizonal_joint_rot_y_min_spin.SetValue(params["horizonal_joint_rot_y_min"])
        self.horizonal_joint_rot_z_min_spin.SetValue(params["horizonal_joint_rot_z_min"])
        self.horizonal_joint_rot_x_max_spin.SetValue(params["horizonal_joint_rot_x_max"])
        self.horizonal_joint_rot_y_max_spin.SetValue(params["horizonal_joint_rot_y_max"])
        self.horizonal_joint_rot_z_max_spin.SetValue(params["horizonal_joint_rot_z_max"])
        self.horizonal_joint_spring_mov_x_spin.SetValue(params["horizonal_joint_spring_mov_x"])
        self.horizonal_joint_spring_mov_y_spin.SetValue(params["horizonal_joint_spring_mov_y"])
        self.horizonal_joint_spring_mov_z_spin.SetValue(params["horizonal_joint_spring_mov_z"])
        self.horizonal_joint_spring_rot_x_spin.SetValue(params["horizonal_joint_spring_rot_x"])
        self.horizonal_joint_spring_rot_y_spin.SetValue(params["horizonal_joint_spring_rot_y"])
        self.horizonal_joint_spring_rot_z_spin.SetValue(params["horizonal_joint_spring_rot_z"])
        self.advance_horizonal_joint_restruct_check.SetValue(params.get("horizonal_joint_restruct", 0))
        self.advance_horizonal_joint_coefficient_spin.SetValue(params["horizonal_joint_coefficient"])
        self.on_horizonal_joint(wx.EVT_CHECKBOX)

        # 斜めジョイント -----------
        self.advance_diagonal_joint_valid_check.SetValue(params["diagonal_joint_valid"])
        self.diagonal_joint_mov_x_min_spin.SetValue(params["diagonal_joint_mov_x_min"])
        self.diagonal_joint_mov_y_min_spin.SetValue(params["diagonal_joint_mov_y_min"])
        self.diagonal_joint_mov_z_min_spin.SetValue(params["diagonal_joint_mov_z_min"])
        self.diagonal_joint_mov_x_max_spin.SetValue(params["diagonal_joint_mov_x_max"])
        self.diagonal_joint_mov_y_max_spin.SetValue(params["diagonal_joint_mov_y_max"])
        self.diagonal_joint_mov_z_max_spin.SetValue(params["diagonal_joint_mov_z_max"])
        self.diagonal_joint_rot_x_min_spin.SetValue(params["diagonal_joint_rot_x_min"])
        self.diagonal_joint_rot_y_min_spin.SetValue(params["diagonal_joint_rot_y_min"])
        self.diagonal_joint_rot_z_min_spin.SetValue(params["diagonal_joint_rot_z_min"])
        self.diagonal_joint_rot_x_max_spin.SetValue(params["diagonal_joint_rot_x_max"])
        self.diagonal_joint_rot_y_max_spin.SetValue(params["diagonal_joint_rot_y_max"])
        self.diagonal_joint_rot_z_max_spin.SetValue(params["diagonal_joint_rot_z_max"])
        self.diagonal_joint_spring_mov_x_spin.SetValue(params["diagonal_joint_spring_mov_x"])
        self.diagonal_joint_spring_mov_y_spin.SetValue(params["diagonal_joint_spring_mov_y"])
        self.diagonal_joint_spring_mov_z_spin.SetValue(params["diagonal_joint_spring_mov_z"])
        self.diagonal_joint_spring_rot_x_spin.SetValue(params["diagonal_joint_spring_rot_x"])
        self.diagonal_joint_spring_rot_y_spin.SetValue(params["diagonal_joint_spring_rot_y"])
        self.diagonal_joint_spring_rot_z_spin.SetValue(params["diagonal_joint_spring_rot_z"])
        self.advance_diagonal_joint_coefficient_spin.SetValue(params["diagonal_joint_coefficient"])
        self.on_diagonal_joint(wx.EVT_CHECKBOX)

        # 縦逆ジョイント -----------
        self.advance_vertical_reverse_joint_valid_check.SetValue(params.get("vertical_reverse_joint_valid", 0))
        self.vertical_reverse_joint_mov_x_min_spin.SetValue(params.get("vertical_reverse_joint_mov_x_min", 0))
        self.vertical_reverse_joint_mov_y_min_spin.SetValue(params.get("vertical_reverse_joint_mov_y_min", 0))
        self.vertical_reverse_joint_mov_z_min_spin.SetValue(params.get("vertical_reverse_joint_mov_z_min", 0))
        self.vertical_reverse_joint_mov_x_max_spin.SetValue(params.get("vertical_reverse_joint_mov_x_max", 0))
        self.vertical_reverse_joint_mov_y_max_spin.SetValue(params.get("vertical_reverse_joint_mov_y_max", 0))
        self.vertical_reverse_joint_mov_z_max_spin.SetValue(params.get("vertical_reverse_joint_mov_z_max", 0))
        self.vertical_reverse_joint_rot_x_min_spin.SetValue(params.get("vertical_reverse_joint_rot_x_min", 0))
        self.vertical_reverse_joint_rot_y_min_spin.SetValue(params.get("vertical_reverse_joint_rot_y_min", 0))
        self.vertical_reverse_joint_rot_z_min_spin.SetValue(params.get("vertical_reverse_joint_rot_z_min", 0))
        self.vertical_reverse_joint_rot_x_max_spin.SetValue(params.get("vertical_reverse_joint_rot_x_max", 0))
        self.vertical_reverse_joint_rot_y_max_spin.SetValue(params.get("vertical_reverse_joint_rot_y_max", 0))
        self.vertical_reverse_joint_rot_z_max_spin.SetValue(params.get("vertical_reverse_joint_rot_z_max", 0))
        self.vertical_reverse_joint_spring_mov_x_spin.SetValue(params.get("vertical_reverse_joint_spring_mov_x", 0))
        self.vertical_reverse_joint_spring_mov_y_spin.SetValue(params.get("vertical_reverse_joint_spring_mov_y", 0))
        self.vertical_reverse_joint_spring_mov_z_spin.SetValue(params.get("vertical_reverse_joint_spring_mov_z", 0))
        self.vertical_reverse_joint_spring_rot_x_spin.SetValue(params.get("vertical_reverse_joint_spring_rot_x", 0))
        self.vertical_reverse_joint_spring_rot_y_spin.SetValue(params.get("vertical_reverse_joint_spring_rot_y", 0))
        self.vertical_reverse_joint_spring_rot_z_spin.SetValue(params.get("vertical_reverse_joint_spring_rot_z", 0))
        self.advance_vertical_reverse_joint_coefficient_spin.SetValue(
            params.get("vertical_reverse_joint_coefficient", 0)
        )
        self.on_vertical_reverse_joint(wx.EVT_CHECKBOX)

        # 横逆ジョイント -----------
        self.advance_horizonal_reverse_joint_valid_check.SetValue(params.get("horizonal_reverse_joint_valid", 0))
        self.horizonal_reverse_joint_mov_x_min_spin.SetValue(params.get("horizonal_reverse_joint_mov_x_min", 0))
        self.horizonal_reverse_joint_mov_y_min_spin.SetValue(params.get("horizonal_reverse_joint_mov_y_min", 0))
        self.horizonal_reverse_joint_mov_z_min_spin.SetValue(params.get("horizonal_reverse_joint_mov_z_min", 0))
        self.horizonal_reverse_joint_mov_x_max_spin.SetValue(params.get("horizonal_reverse_joint_mov_x_max", 0))
        self.horizonal_reverse_joint_mov_y_max_spin.SetValue(params.get("horizonal_reverse_joint_mov_y_max", 0))
        self.horizonal_reverse_joint_mov_z_max_spin.SetValue(params.get("horizonal_reverse_joint_mov_z_max", 0))
        self.horizonal_reverse_joint_rot_x_min_spin.SetValue(params.get("horizonal_reverse_joint_rot_x_min", 0))
        self.horizonal_reverse_joint_rot_y_min_spin.SetValue(params.get("horizonal_reverse_joint_rot_y_min", 0))
        self.horizonal_reverse_joint_rot_z_min_spin.SetValue(params.get("horizonal_reverse_joint_rot_z_min", 0))
        self.horizonal_reverse_joint_rot_x_max_spin.SetValue(params.get("horizonal_reverse_joint_rot_x_max", 0))
        self.horizonal_reverse_joint_rot_y_max_spin.SetValue(params.get("horizonal_reverse_joint_rot_y_max", 0))
        self.horizonal_reverse_joint_rot_z_max_spin.SetValue(params.get("horizonal_reverse_joint_rot_z_max", 0))
        self.horizonal_reverse_joint_spring_mov_x_spin.SetValue(params.get("horizonal_reverse_joint_spring_mov_x", 0))
        self.horizonal_reverse_joint_spring_mov_y_spin.SetValue(params.get("horizonal_reverse_joint_spring_mov_y", 0))
        self.horizonal_reverse_joint_spring_mov_z_spin.SetValue(params.get("horizonal_reverse_joint_spring_mov_z", 0))
        self.horizonal_reverse_joint_spring_rot_x_spin.SetValue(params.get("horizonal_reverse_joint_spring_rot_x", 0))
        self.horizonal_reverse_joint_spring_rot_y_spin.SetValue(params.get("horizonal_reverse_joint_spring_rot_y", 0))
        self.horizonal_reverse_joint_spring_rot_z_spin.SetValue(params.get("horizonal_reverse_joint_spring_rot_z", 0))
        self.advance_horizonal_reverse_joint_coefficient_spin.SetValue(
            params.get("horizonal_reverse_joint_coefficient", 0)
        )
        self.on_horizonal_reverse_joint(wx.EVT_CHECKBOX)

    def on_param_export(self, event: wx.Event):
        params = self.get_param_export_data()

        with wx.FileDialog(
            self.frame,
            logger.transtext("材質物理設定JSONを保存する"),
            wildcard="JSONファイル (*.json)|*.json|すべてのファイル (*.*)|*.*",
            defaultDir=os.path.dirname(self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.path),
            style=wx.FD_SAVE,
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # Proceed loading the file chosen by the user
            target_physics_path = fileDialog.GetPath()
            try:
                with open(target_physics_path, "w") as f:
                    json.dump(params, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": "))

                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("材質物理設定JSONのエクスポートに成功しました \n{0}").format(target_physics_path),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()
            except Exception:
                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("材質物理設定JSONが保存できませんでした '{0}'\n\n{1}").format(
                        target_physics_path, traceback.format_exc()
                    ),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()

    def get_param_export_data(self):
        params = {}

        # 簡易版オプションデータ -------------
        params["mass"] = self.simple_mass_slider.GetValue()
        params["air_resistance"] = self.simple_air_resistance_slider.GetValue()
        params["shape_maintenance"] = self.simple_shape_maintenance_slider.GetValue()

        # 詳細版オプションデータ -------------
        params["vertical_bone_density"] = int(self.vertical_bone_density_spin.GetValue())
        params["horizonal_bone_density"] = int(self.horizonal_bone_density_spin.GetValue())
        params["horizonal_bone_offset"] = int(self.horizonal_bone_offset_spin.GetValue())

        # 剛体 ---------------
        params["rigidbody_shape_type"] = self.advance_rigidbody_shape_type_ctrl.GetSelection()
        params["rigidbody_mass"] = self.rigidbody_mass_spin.GetValue()
        params["rigidbody_linear_damping"] = self.rigidbody_linear_damping_spin.GetValue()
        params["rigidbody_angular_damping"] = self.rigidbody_angular_damping_spin.GetValue()
        params["rigidbody_restitution"] = self.rigidbody_restitution_spin.GetValue()
        params["rigidbody_friction"] = self.rigidbody_friction_spin.GetValue()
        params["rigidbody_coefficient"] = self.rigidbody_coefficient_spin.GetValue()
        params["rigidbody_balancer"] = self.advance_rigidbody_balancer_ctrl.GetValue()
        params["rigidbody_leg"] = self.advance_rigidbody_leg_ctrl.GetValue()

        params["rigidbody_root_thicks"] = self.rigidbody_root_thicks_spin.GetValue()
        params["rigidbody_end_thicks"] = self.rigidbody_end_thicks_spin.GetValue()
        params["rigidbody_cover_coefficient"] = self.rigidbody_cover_coefficient_spin.GetValue()

        # 縦ジョイント -----------
        params["vertical_joint_valid"] = self.advance_vertical_joint_valid_check.GetValue()
        params["vertical_joint_mov_x_min"] = self.vertical_joint_mov_x_min_spin.GetValue()
        params["vertical_joint_mov_y_min"] = self.vertical_joint_mov_y_min_spin.GetValue()
        params["vertical_joint_mov_z_min"] = self.vertical_joint_mov_z_min_spin.GetValue()
        params["vertical_joint_mov_x_max"] = self.vertical_joint_mov_x_max_spin.GetValue()
        params["vertical_joint_mov_y_max"] = self.vertical_joint_mov_y_max_spin.GetValue()
        params["vertical_joint_mov_z_max"] = self.vertical_joint_mov_z_max_spin.GetValue()
        params["vertical_joint_rot_x_min"] = self.vertical_joint_rot_x_min_spin.GetValue()
        params["vertical_joint_rot_y_min"] = self.vertical_joint_rot_y_min_spin.GetValue()
        params["vertical_joint_rot_z_min"] = self.vertical_joint_rot_z_min_spin.GetValue()
        params["vertical_joint_rot_x_max"] = self.vertical_joint_rot_x_max_spin.GetValue()
        params["vertical_joint_rot_y_max"] = self.vertical_joint_rot_y_max_spin.GetValue()
        params["vertical_joint_rot_z_max"] = self.vertical_joint_rot_z_max_spin.GetValue()
        params["vertical_joint_spring_mov_x"] = self.vertical_joint_spring_mov_x_spin.GetValue()
        params["vertical_joint_spring_mov_y"] = self.vertical_joint_spring_mov_y_spin.GetValue()
        params["vertical_joint_spring_mov_z"] = self.vertical_joint_spring_mov_z_spin.GetValue()
        params["vertical_joint_spring_rot_x"] = self.vertical_joint_spring_rot_x_spin.GetValue()
        params["vertical_joint_spring_rot_y"] = self.vertical_joint_spring_rot_y_spin.GetValue()
        params["vertical_joint_spring_rot_z"] = self.vertical_joint_spring_rot_z_spin.GetValue()
        params["vertical_joint_coefficient"] = self.advance_vertical_joint_coefficient_spin.GetValue()

        # 横ジョイント -----------
        params["horizonal_joint_valid"] = self.advance_horizonal_joint_valid_check.GetValue()
        params["horizonal_joint_mov_x_min"] = self.horizonal_joint_mov_x_min_spin.GetValue()
        params["horizonal_joint_mov_y_min"] = self.horizonal_joint_mov_y_min_spin.GetValue()
        params["horizonal_joint_mov_z_min"] = self.horizonal_joint_mov_z_min_spin.GetValue()
        params["horizonal_joint_mov_x_max"] = self.horizonal_joint_mov_x_max_spin.GetValue()
        params["horizonal_joint_mov_y_max"] = self.horizonal_joint_mov_y_max_spin.GetValue()
        params["horizonal_joint_mov_z_max"] = self.horizonal_joint_mov_z_max_spin.GetValue()
        params["horizonal_joint_rot_x_min"] = self.horizonal_joint_rot_x_min_spin.GetValue()
        params["horizonal_joint_rot_y_min"] = self.horizonal_joint_rot_y_min_spin.GetValue()
        params["horizonal_joint_rot_z_min"] = self.horizonal_joint_rot_z_min_spin.GetValue()
        params["horizonal_joint_rot_x_max"] = self.horizonal_joint_rot_x_max_spin.GetValue()
        params["horizonal_joint_rot_y_max"] = self.horizonal_joint_rot_y_max_spin.GetValue()
        params["horizonal_joint_rot_z_max"] = self.horizonal_joint_rot_z_max_spin.GetValue()
        params["horizonal_joint_spring_mov_x"] = self.horizonal_joint_spring_mov_x_spin.GetValue()
        params["horizonal_joint_spring_mov_y"] = self.horizonal_joint_spring_mov_y_spin.GetValue()
        params["horizonal_joint_spring_mov_z"] = self.horizonal_joint_spring_mov_z_spin.GetValue()
        params["horizonal_joint_spring_rot_x"] = self.horizonal_joint_spring_rot_x_spin.GetValue()
        params["horizonal_joint_spring_rot_y"] = self.horizonal_joint_spring_rot_y_spin.GetValue()
        params["horizonal_joint_spring_rot_z"] = self.horizonal_joint_spring_rot_z_spin.GetValue()
        params["horizonal_joint_restruct"] = self.advance_horizonal_joint_restruct_check.GetValue()
        params["horizonal_joint_coefficient"] = self.advance_horizonal_joint_coefficient_spin.GetValue()

        # 斜めジョイント -----------
        params["diagonal_joint_valid"] = self.advance_diagonal_joint_valid_check.GetValue()
        params["diagonal_joint_mov_x_min"] = self.diagonal_joint_mov_x_min_spin.GetValue()
        params["diagonal_joint_mov_y_min"] = self.diagonal_joint_mov_y_min_spin.GetValue()
        params["diagonal_joint_mov_z_min"] = self.diagonal_joint_mov_z_min_spin.GetValue()
        params["diagonal_joint_mov_x_max"] = self.diagonal_joint_mov_x_max_spin.GetValue()
        params["diagonal_joint_mov_y_max"] = self.diagonal_joint_mov_y_max_spin.GetValue()
        params["diagonal_joint_mov_z_max"] = self.diagonal_joint_mov_z_max_spin.GetValue()
        params["diagonal_joint_rot_x_min"] = self.diagonal_joint_rot_x_min_spin.GetValue()
        params["diagonal_joint_rot_y_min"] = self.diagonal_joint_rot_y_min_spin.GetValue()
        params["diagonal_joint_rot_z_min"] = self.diagonal_joint_rot_z_min_spin.GetValue()
        params["diagonal_joint_rot_x_max"] = self.diagonal_joint_rot_x_max_spin.GetValue()
        params["diagonal_joint_rot_y_max"] = self.diagonal_joint_rot_y_max_spin.GetValue()
        params["diagonal_joint_rot_z_max"] = self.diagonal_joint_rot_z_max_spin.GetValue()
        params["diagonal_joint_spring_mov_x"] = self.diagonal_joint_spring_mov_x_spin.GetValue()
        params["diagonal_joint_spring_mov_y"] = self.diagonal_joint_spring_mov_y_spin.GetValue()
        params["diagonal_joint_spring_mov_z"] = self.diagonal_joint_spring_mov_z_spin.GetValue()
        params["diagonal_joint_spring_rot_x"] = self.diagonal_joint_spring_rot_x_spin.GetValue()
        params["diagonal_joint_spring_rot_y"] = self.diagonal_joint_spring_rot_y_spin.GetValue()
        params["diagonal_joint_spring_rot_z"] = self.diagonal_joint_spring_rot_z_spin.GetValue()
        params["diagonal_joint_coefficient"] = self.advance_diagonal_joint_coefficient_spin.GetValue()

        # 縦逆ジョイント -----------
        params["vertical_reverse_joint_valid"] = self.advance_vertical_reverse_joint_valid_check.GetValue()
        params["vertical_reverse_joint_mov_x_min"] = self.vertical_reverse_joint_mov_x_min_spin.GetValue()
        params["vertical_reverse_joint_mov_y_min"] = self.vertical_reverse_joint_mov_y_min_spin.GetValue()
        params["vertical_reverse_joint_mov_z_min"] = self.vertical_reverse_joint_mov_z_min_spin.GetValue()
        params["vertical_reverse_joint_mov_x_max"] = self.vertical_reverse_joint_mov_x_max_spin.GetValue()
        params["vertical_reverse_joint_mov_y_max"] = self.vertical_reverse_joint_mov_y_max_spin.GetValue()
        params["vertical_reverse_joint_mov_z_max"] = self.vertical_reverse_joint_mov_z_max_spin.GetValue()
        params["vertical_reverse_joint_rot_x_min"] = self.vertical_reverse_joint_rot_x_min_spin.GetValue()
        params["vertical_reverse_joint_rot_y_min"] = self.vertical_reverse_joint_rot_y_min_spin.GetValue()
        params["vertical_reverse_joint_rot_z_min"] = self.vertical_reverse_joint_rot_z_min_spin.GetValue()
        params["vertical_reverse_joint_rot_x_max"] = self.vertical_reverse_joint_rot_x_max_spin.GetValue()
        params["vertical_reverse_joint_rot_y_max"] = self.vertical_reverse_joint_rot_y_max_spin.GetValue()
        params["vertical_reverse_joint_rot_z_max"] = self.vertical_reverse_joint_rot_z_max_spin.GetValue()
        params["vertical_reverse_joint_spring_mov_x"] = self.vertical_reverse_joint_spring_mov_x_spin.GetValue()
        params["vertical_reverse_joint_spring_mov_y"] = self.vertical_reverse_joint_spring_mov_y_spin.GetValue()
        params["vertical_reverse_joint_spring_mov_z"] = self.vertical_reverse_joint_spring_mov_z_spin.GetValue()
        params["vertical_reverse_joint_spring_rot_x"] = self.vertical_reverse_joint_spring_rot_x_spin.GetValue()
        params["vertical_reverse_joint_spring_rot_y"] = self.vertical_reverse_joint_spring_rot_y_spin.GetValue()
        params["vertical_reverse_joint_spring_rot_z"] = self.vertical_reverse_joint_spring_rot_z_spin.GetValue()
        params["vertical_reverse_joint_coefficient"] = self.advance_vertical_reverse_joint_coefficient_spin.GetValue()

        # 横逆ジョイント -----------
        params["horizonal_reverse_joint_valid"] = self.advance_horizonal_reverse_joint_valid_check.GetValue()
        params["horizonal_reverse_joint_mov_x_min"] = self.horizonal_reverse_joint_mov_x_min_spin.GetValue()
        params["horizonal_reverse_joint_mov_y_min"] = self.horizonal_reverse_joint_mov_y_min_spin.GetValue()
        params["horizonal_reverse_joint_mov_z_min"] = self.horizonal_reverse_joint_mov_z_min_spin.GetValue()
        params["horizonal_reverse_joint_mov_x_max"] = self.horizonal_reverse_joint_mov_x_max_spin.GetValue()
        params["horizonal_reverse_joint_mov_y_max"] = self.horizonal_reverse_joint_mov_y_max_spin.GetValue()
        params["horizonal_reverse_joint_mov_z_max"] = self.horizonal_reverse_joint_mov_z_max_spin.GetValue()
        params["horizonal_reverse_joint_rot_x_min"] = self.horizonal_reverse_joint_rot_x_min_spin.GetValue()
        params["horizonal_reverse_joint_rot_y_min"] = self.horizonal_reverse_joint_rot_y_min_spin.GetValue()
        params["horizonal_reverse_joint_rot_z_min"] = self.horizonal_reverse_joint_rot_z_min_spin.GetValue()
        params["horizonal_reverse_joint_rot_x_max"] = self.horizonal_reverse_joint_rot_x_max_spin.GetValue()
        params["horizonal_reverse_joint_rot_y_max"] = self.horizonal_reverse_joint_rot_y_max_spin.GetValue()
        params["horizonal_reverse_joint_rot_z_max"] = self.horizonal_reverse_joint_rot_z_max_spin.GetValue()
        params["horizonal_reverse_joint_spring_mov_x"] = self.horizonal_reverse_joint_spring_mov_x_spin.GetValue()
        params["horizonal_reverse_joint_spring_mov_y"] = self.horizonal_reverse_joint_spring_mov_y_spin.GetValue()
        params["horizonal_reverse_joint_spring_mov_z"] = self.horizonal_reverse_joint_spring_mov_z_spin.GetValue()
        params["horizonal_reverse_joint_spring_rot_x"] = self.horizonal_reverse_joint_spring_rot_x_spin.GetValue()
        params["horizonal_reverse_joint_spring_rot_y"] = self.horizonal_reverse_joint_spring_rot_y_spin.GetValue()
        params["horizonal_reverse_joint_spring_rot_z"] = self.horizonal_reverse_joint_spring_rot_z_spin.GetValue()
        params[
            "horizonal_reverse_joint_coefficient"
        ] = self.advance_horizonal_reverse_joint_coefficient_spin.GetValue()

        return params

    def on_bone_clear(self, event: wx.Event):
        for r in range(self.bone_grid.GetNumberRows()):
            for c in range(self.bone_grid.GetNumberCols()):
                self.bone_grid.GetTable().SetValue(r, c, "")

    def on_bone_import(self, event: wx.Event):
        with wx.FileDialog(
            self.frame,
            logger.transtext("ボーン設定CSVを読み込む"),
            wildcard="CSVファイル (*.csv)|*.csv|すべてのファイル (*.*)|*.*",
            defaultDir=os.path.dirname(self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.path),
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # Proceed loading the file chosen by the user
            target_physics_path = fileDialog.GetPath()
            try:
                with open(target_physics_path, encoding="cp932", mode="r") as f:
                    cr = csv.reader(f, delimiter=",", quotechar='"')

                    for r, row in enumerate(cr):
                        for c, val in enumerate(row):
                            if r < self.bone_grid.GetNumberRows() and c < self.bone_grid.GetNumberCols():
                                self.bone_grid.GetTable().SetValue(r, c, val)

                self.bone_grid.ForceRefresh()

                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("ボーン設定CSVのインポートに成功しました \n{0}").format(target_physics_path),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()
            except Exception:
                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("ボーン設定CSVが読み込めませんでした '{0}'\n\n{1}").format(
                        target_physics_path, traceback.format_exc()
                    ),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()

    def on_bone_export(self, event: wx.Event):
        bone_grid, _, _, _ = self.get_bone_grid()

        with wx.FileDialog(
            self.frame,
            logger.transtext("ボーン設定CSVを保存する"),
            wildcard="CSVファイル (*.csv)|*.csv|すべてのファイル (*.*)|*.*",
            defaultDir=os.path.dirname(self.main_frame.file_panel_ctrl.org_model_file_ctrl.data.path),
            style=wx.FD_SAVE,
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # Proceed loading the file chosen by the user
            target_physics_path = fileDialog.GetPath()
            try:
                with open(target_physics_path, encoding="cp932", mode="w", newline="") as f:
                    cw = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

                    for bone_cols in bone_grid.values():
                        cw.writerow(list(bone_cols.values()))

                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("ボーン設定CSVのエクスポートに成功しました \n{0}").format(target_physics_path),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()
            except Exception:
                dialog = wx.MessageDialog(
                    self.frame,
                    logger.transtext("ボーン設定CSVが保存できませんでした '{0}'\n\n{1}").format(
                        target_physics_path, traceback.format_exc()
                    ),
                    style=wx.OK,
                )
                dialog.ShowModal()
                dialog.Destroy()

    def on_vertical_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_vertical_joint_coefficient_spin.Enable(self.advance_vertical_joint_valid_check.GetValue())
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
        self.advance_horizonal_joint_restruct_check.Enable(self.advance_horizonal_joint_valid_check.GetValue())
        self.advance_horizonal_joint_coefficient_spin.Enable(self.advance_horizonal_joint_valid_check.GetValue())
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
        self.advance_diagonal_joint_coefficient_spin.Enable(self.advance_diagonal_joint_valid_check.GetValue())
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

    def on_vertical_reverse_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_vertical_reverse_joint_coefficient_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_mov_x_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_mov_x_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_mov_y_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_mov_y_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_mov_z_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_mov_z_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_x_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_x_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_y_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_y_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_z_min_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_rot_z_max_spin.Enable(self.advance_vertical_reverse_joint_valid_check.GetValue())
        self.vertical_reverse_joint_spring_mov_x_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_spring_mov_y_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_spring_mov_z_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_spring_rot_x_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_spring_rot_y_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )
        self.vertical_reverse_joint_spring_rot_z_spin.Enable(
            self.advance_vertical_reverse_joint_valid_check.GetValue()
        )

    def on_horizonal_reverse_joint(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_horizonal_reverse_joint_coefficient_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_mov_x_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_mov_x_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_mov_y_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_mov_y_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_mov_z_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_mov_z_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_x_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_x_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_y_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_y_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_z_min_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_rot_z_max_spin.Enable(self.advance_horizonal_reverse_joint_valid_check.GetValue())
        self.horizonal_reverse_joint_spring_mov_x_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_spring_mov_y_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_spring_mov_z_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_spring_rot_x_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_spring_rot_y_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )
        self.horizonal_reverse_joint_spring_rot_z_spin.Enable(
            self.advance_horizonal_reverse_joint_valid_check.GetValue()
        )

    def on_wheel_double_spin_ctrl(self, event: wx.Event, inc=0.1):
        self.main_frame.on_wheel_spin_ctrl(event, inc)
        self.main_frame.file_panel_ctrl.on_change_file(event)

    def on_special_shape(self, event: wx.Event):
        if self.simple_special_shape_ctrl.GetStringSelection() in [logger.transtext("全て表面")]:
            # すべて表面の場合、推定を角度に再設定する（デフォルトで入っているはずだが、念のため）
            self.route_estimate_type_ctrl.SetStringSelection(logger.transtext("角度"))
            self.route_search_type_ctrl.SetStringSelection(logger.transtext("前頂点優先"))
        self.main_frame.file_panel_ctrl.on_change_file(event)

    def set_density_type(self, event: wx.Event):
        if self.density_type_ctrl.GetStringSelection() in [logger.transtext("中央")]:
            # 中央の場合、ボーン間（余計な剛体を作る必要はない）
            self.joint_pos_type_ctrl.SetStringSelection(logger.transtext("ボーン間"))
            self.advance_horizonal_joint_valid_check.SetValue(0)
        self.main_frame.file_panel_ctrl.on_change_file(event)

    def set_material_name(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        # setValueでそのままset_abb_nameを起動する
        self.simple_abb_ctrl.SetValue(self.simple_material_ctrl.GetStringSelection())

        # 設定ボーンパネル初期化
        self.initialize_bone_param(event)

    def set_abb_name(self, event: wx.Event):
        # バイト長を加味してスライス
        self.simple_abb_ctrl.ChangeValue(truncate_double_byte_str(event.GetEventObject().GetValue(), 6))

        # ラベル設定
        label_text = f"{self.simple_material_ctrl.GetStringSelection()}:{self.simple_abb_ctrl.GetValue()}"
        self.advance_material_ctrl.SetLabelText(label_text)
        self.bone_material_ctrl.SetLabelText(label_text)

    def set_abb_setting(self, event: wx.Event):
        # 既存の設定があったら再設定
        if self.simple_abb_ctrl.GetValue() in self.main_frame.file_hitories["abb_setting"]:
            abb_setting = self.main_frame.file_hitories["abb_setting"][self.simple_abb_ctrl.GetValue()]
            if not self.simple_material_ctrl.GetStringSelection():
                self.simple_material_ctrl.SetStringSelection(abb_setting["material_name"])
            if not self.simple_parent_bone_ctrl.GetStringSelection():
                self.simple_parent_bone_ctrl.SetStringSelection(abb_setting["parent_bone_name"])
            if not self.simple_group_ctrl.GetStringSelection():
                self.simple_group_ctrl.SetStringSelection(abb_setting["group"])
            if self.simple_direction_ctrl.GetStringSelection() == logger.transtext("下"):
                self.simple_direction_ctrl.SetStringSelection(abb_setting["direction"])
            if self.simple_exist_physics_clear_ctrl.GetStringSelection() == logger.transtext("そのまま"):
                self.simple_exist_physics_clear_ctrl.SetStringSelection(abb_setting["exist_physics_clear"])
            if not self.simple_primitive_ctrl.GetStringSelection():
                self.simple_primitive_ctrl.SetStringSelection(abb_setting["primitive"])
            if self.simple_special_shape_ctrl.GetStringSelection() == logger.transtext("なし"):
                self.simple_special_shape_ctrl.SetStringSelection(
                    abb_setting.get("special_shape", logger.transtext("なし"))
                )
            if not self.simple_back_material_ctrl.GetStringSelection():
                self.simple_back_material_ctrl.SetStringSelection(abb_setting.get("back_material_name", ""))
            if not self.simple_extend_back_choice_ctrl.GetSelections() and abb_setting.get(
                "back_extend_material_names", ""
            ):
                selected_idxs = []
                for n, material_name in enumerate(self.frame.material_list):
                    if material_name and material_name in abb_setting.get("back_extend_material_names", "").split(","):
                        selected_idxs.append(n - 1)
                self.simple_extend_back_choice_ctrl.SetSelections(selected_idxs)
            if not self.simple_edge_material_ctrl.GetStringSelection():
                self.simple_edge_material_ctrl.SetStringSelection(abb_setting.get("edge_material_name", ""))
            if not self.simple_extend_edge_choice_ctrl.GetSelections() and abb_setting.get(
                "edge_extend_material_names", ""
            ):
                selected_idxs = []
                for n, material_name in enumerate(self.frame.material_list):
                    if material_name and material_name in abb_setting.get("edge_extend_material_names", "").split(","):
                        selected_idxs.append(n - 1)
                self.simple_extend_edge_choice_ctrl.SetSelections(selected_idxs)
            if not self.vertices_csv_file_ctrl.path():
                self.vertices_csv_file_ctrl.file_ctrl.SetPath(abb_setting.get("vertices_csv", ""))
            if not self.vertices_edge_csv_file_ctrl.path():
                self.vertices_edge_csv_file_ctrl.file_ctrl.SetPath(abb_setting.get("vertices_edge_csv", ""))
            if not self.vertices_back_csv_file_ctrl.path():
                self.vertices_back_csv_file_ctrl.file_ctrl.SetPath(abb_setting.get("vertices_back_csv", ""))
            if not self.top_vertices_csv_file_ctrl.path():
                self.top_vertices_csv_file_ctrl.file_ctrl.SetPath(abb_setting.get("top_vertices_csv", ""))

            # 詳細画面
            if self.physics_type_ctrl.GetStringSelection() == logger.transtext("布"):
                self.physics_type_ctrl.SetStringSelection(abb_setting.get("physics_type", logger.transtext("布")))
            if self.density_type_ctrl.GetStringSelection() == logger.transtext("頂点"):
                self.density_type_ctrl.SetStringSelection(abb_setting.get("density_type", logger.transtext("頂点")))
            if self.joint_pos_type_ctrl.GetStringSelection() == logger.transtext("ボーン位置"):
                self.joint_pos_type_ctrl.SetStringSelection(
                    abb_setting.get("joint_pos_type", logger.transtext("ボーン位置"))
                )
            if self.route_search_type_ctrl.GetStringSelection() == logger.transtext("前頂点優先"):
                self.route_search_type_ctrl.SetStringSelection(
                    abb_setting.get("route_search_type", logger.transtext("前頂点優先"))
                )
            if self.route_estimate_type_ctrl.GetStringSelection() == logger.transtext("角度"):
                self.route_estimate_type_ctrl.SetStringSelection(
                    abb_setting.get("route_estimate_type", logger.transtext("角度"))
                )
            if not self.physics_parent_spin.GetValue():
                self.physics_parent_spin.SetValue(int(abb_setting.get("physics_parent", 0)))
            if self.parent_type_ctrl.GetStringSelection() == logger.transtext("中心"):
                self.parent_type_ctrl.SetStringSelection(abb_setting.get("parent_type", logger.transtext("中心")))

            # 物理パラメーターも設定する
            self.set_param_import_data(abb_setting["params"])

    def initialize_bone_param(self, event: wx.Event):
        # ウェイトボーンリンク生成
        self.create_weighted_bone_names()
        max_link_num = 0
        for bone_names in self.weighted_bone_names.values():
            max_link_num = len(bone_names) if max_link_num < len(bone_names) else max_link_num

        self.bone_sizer.Hide(self.bone_grid_sizer, recursive=True)
        self.bone_grid_sizer.Clear()

        # ボーングリッド再生成
        self.bone_grid_sizer = wx.BoxSizer(wx.VERTICAL)
        self.bone_grid = Grid(self.bone_window)
        self.bone_grid.CreateGrid(max_link_num, len(self.weighted_bone_names.keys()))

        max_bone_name_cnt = 0
        all_bone_names = [""]
        for bone_names in self.weighted_bone_names.values():
            for bone_name in bone_names:
                all_bone_names.append(bone_name)
                max_bone_name_cnt = len(bone_name) if max_bone_name_cnt < len(bone_name) else max_bone_name_cnt

        for c in range(len(self.weighted_bone_names.keys())):
            for r in range(max_link_num):
                self.bone_grid.SetCellEditor(r, c, GridCellChoiceEditor(choices=all_bone_names))
            self.bone_grid.SetColLabelValue(c, str(c + 1))
            self.bone_grid.SetColSize(c, int((self.bone_grid.GetCellFont(r, c).PointSize * 1.2) * max_bone_name_cnt))

        self.bone_grid.Bind(EVT_GRID_CELL_CHANGING, self.change_bone_grid)

        self.bone_grid_sizer.Add(self.bone_grid, 0, wx.ALL | wx.EXPAND, 0)
        self.bone_sizer.Add(self.bone_grid_sizer, 0, wx.ALL, 0)

        self.bone_sizer.Layout()

        self.main_frame.bone_param_panel_ctrl.bone_sizer.Layout()
        self.main_frame.bone_param_panel_ctrl.bone_sizer.FitInside(self.bone_window)

    def change_bone_grid(self, event: wx.Event):
        if event.GetString():
            # 文字列が選択されている場合、その子どもを下位層に設定する
            is_set = False
            for tail_bone_name, bone_names in self.weighted_bone_names.items():
                for bi, bone_name in enumerate(bone_names):
                    if bone_name == event.GetString():
                        is_set = True
                        break
                if is_set:
                    break

            for r in range(event.GetRow() + 1, self.bone_grid.GetNumberRows()):
                bi += 1
                if bi < len(bone_names):
                    self.bone_grid.GetTable().SetValue(r, event.GetCol(), bone_names[bi])

            self.bone_grid.ForceRefresh()
        else:
            # 空欄が設定された場合、下も全部空欄
            for r in range(event.GetRow() + 1, self.bone_grid.GetNumberRows()):
                self.bone_grid.GetTable().SetValue(r, event.GetCol(), "")

            self.bone_grid.ForceRefresh()

    def get_bone_grid(self):
        bone_grid = {}
        is_boned = False
        max_r = 0
        max_c = 0
        for r in range(self.bone_grid.GetNumberRows()):
            bone_grid[r] = {}
            for c in range(self.bone_grid.GetNumberCols()):
                bone_name = self.bone_grid.GetTable().GetValue(r, c)
                bone_grid[r][c] = bone_name
                if bone_name:
                    is_boned = True
                if r > 0 and bone_grid[r][c] and not bone_grid[r - 1][c]:
                    logger.error(
                        "行[%s], 列[%s]にボーン名が指定されておらず、次の行にボーン名が指定されています。\n空欄の後にボーン名を指定しないでください。",
                        r,
                        c + 1,
                        decoration=MLogger.DECORATION_BOX,
                    )
                    return bone_grid, max_r + 1, max_c + 1, False

                max_r = r if r > max_r and bone_name else max_r
                max_c = c if c > max_c and bone_name else max_c

        return bone_grid, max_r + 1, max_c + 1, is_boned

    def set_mass(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.rigidbody_mass_spin.SetValue(self.simple_mass_slider.GetValue())

        self.rigidbody_coefficient_spin.SetValue(self.simple_mass_slider.GetMax() / self.simple_mass_slider.GetValue())

        self.set_air_resistance(event)

    def set_simple_primitive(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.advance_rigidbody_balancer_ctrl.SetValue(0)
        self.advance_rigidbody_leg_ctrl.SetValue(0)
        self.advance_horizonal_joint_restruct_check.SetValue(0)
        self.rigidbody_root_thicks_spin.SetValue(0.07)
        self.rigidbody_end_thicks_spin.SetValue(0.3)
        self.rigidbody_cover_coefficient_spin.SetValue(1.2)
        # self.joint_pos_type_ctrl.SetStringSelection(logger.transtext("ボーン位置"))
        # self.route_search_type_ctrl.SetStringSelection(logger.transtext("前頂点優先"))
        # self.route_estimate_type_ctrl.SetStringSelection(logger.transtext("角度"))

        self.advance_vertical_joint_valid_check.SetValue(1)
        self.advance_horizonal_joint_valid_check.SetValue(0)
        self.advance_diagonal_joint_valid_check.SetValue(0)
        self.advance_vertical_reverse_joint_valid_check.SetValue(0)
        self.advance_horizonal_reverse_joint_valid_check.SetValue(0)

        self.advance_vertical_joint_coefficient_spin.SetValue(2.8)
        self.advance_horizonal_joint_coefficient_spin.SetValue(4.2)
        self.advance_diagonal_joint_coefficient_spin.SetValue(1)
        self.advance_vertical_reverse_joint_coefficient_spin.SetValue(1)
        self.advance_horizonal_reverse_joint_coefficient_spin.SetValue(1)

        if logger.transtext("単一揺れ物") in self.simple_primitive_ctrl.GetStringSelection():
            self.physics_type_ctrl.SetStringSelection(logger.transtext("単一揺"))
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection(logger.transtext("カプセル"))
            self.simple_exist_physics_clear_ctrl.SetStringSelection(logger.transtext("再利用"))
            self.parent_type_ctrl.SetStringSelection(logger.transtext("親"))
        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext(
            "胸(小)"
        ) or self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("胸(大)"):
            self.physics_type_ctrl.SetStringSelection(logger.transtext("胸"))
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection(logger.transtext("球"))
            self.simple_exist_physics_clear_ctrl.SetStringSelection(logger.transtext("再利用"))
            self.parent_type_ctrl.SetStringSelection(logger.transtext("親"))
        elif logger.transtext("髪") in self.simple_primitive_ctrl.GetStringSelection():
            self.physics_type_ctrl.SetStringSelection(logger.transtext("髪"))
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection(logger.transtext("カプセル"))
            self.simple_exist_physics_clear_ctrl.SetStringSelection(logger.transtext("再利用"))
            if self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("髪(アホ毛)"):
                self.parent_type_ctrl.SetStringSelection(logger.transtext("中心"))
            else:
                self.parent_type_ctrl.SetStringSelection(logger.transtext("親"))
        else:
            self.physics_type_ctrl.SetStringSelection(logger.transtext("布"))
            self.advance_rigidbody_shape_type_ctrl.SetStringSelection(logger.transtext("箱"))
            # self.parent_type_ctrl.SetStringSelection(logger.transtext("中心"))

        if self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(コットン)"):
            self.simple_mass_slider.SetValue(3.2)
            self.simple_air_resistance_slider.SetValue(1.5)
            self.simple_shape_maintenance_slider.SetValue(1.5)

            self.advance_horizonal_joint_valid_check.SetValue(1)

        if self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(ウール)"):
            self.simple_mass_slider.SetValue(6.2)
            self.simple_air_resistance_slider.SetValue(3.5)
            self.simple_shape_maintenance_slider.SetValue(0.6)

            self.advance_horizonal_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(シルク)"):
            self.simple_mass_slider.SetValue(1.8)
            self.simple_air_resistance_slider.SetValue(0.9)
            self.simple_shape_maintenance_slider.SetValue(0.2)

            self.advance_horizonal_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(ベルベッド)"):
            self.simple_mass_slider.SetValue(5.5)
            self.simple_air_resistance_slider.SetValue(2.1)
            self.simple_shape_maintenance_slider.SetValue(1.8)

            self.advance_horizonal_joint_valid_check.SetValue(1)
            self.advance_diagonal_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(レザー)"):
            self.simple_mass_slider.SetValue(6)
            self.simple_air_resistance_slider.SetValue(1.5)
            self.simple_shape_maintenance_slider.SetValue(2.5)

            self.advance_horizonal_joint_valid_check.SetValue(1)
            self.advance_diagonal_joint_valid_check.SetValue(1)
            # self.advance_vertical_reverse_joint_valid_check.SetValue(1)
            # self.advance_horizonal_reverse_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("布(デニム)"):
            self.simple_mass_slider.SetValue(7)
            self.simple_air_resistance_slider.SetValue(3.5)
            self.simple_shape_maintenance_slider.SetValue(2.0)

            self.advance_horizonal_joint_valid_check.SetValue(1)
            self.advance_diagonal_joint_valid_check.SetValue(1)
            # self.advance_vertical_reverse_joint_valid_check.SetValue(1)
            # self.advance_horizonal_reverse_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("単一揺れ物"):
            self.simple_mass_slider.SetValue(3.3)
            self.simple_air_resistance_slider.SetValue(2.5)
            self.simple_shape_maintenance_slider.SetValue(2.8)
            self.joint_pos_type_ctrl.SetStringSelection(logger.transtext("ボーン位置"))

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("髪(ショート)"):

            self.simple_mass_slider.SetValue(2)
            self.simple_air_resistance_slider.SetValue(4.5)
            self.simple_shape_maintenance_slider.SetValue(1.6)

            self.advance_vertical_joint_coefficient_spin.SetValue(1.0)

            self.advance_vertical_reverse_joint_coefficient_spin.SetValue(1.0)
            self.advance_vertical_reverse_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("髪(ロング)"):
            self.simple_mass_slider.SetValue(1.2)
            self.simple_air_resistance_slider.SetValue(3.8)
            self.simple_shape_maintenance_slider.SetValue(1.6)

            self.advance_vertical_joint_coefficient_spin.SetValue(1.0)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("髪(アホ毛)"):
            self.simple_mass_slider.SetValue(2.2)
            self.simple_air_resistance_slider.SetValue(0.2)
            self.simple_shape_maintenance_slider.SetValue(1.5)

            self.advance_vertical_joint_coefficient_spin.SetValue(1.0)
            self.advance_rigidbody_balancer_ctrl.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("胸(小)"):
            self.simple_mass_slider.SetValue(4.0)
            self.simple_air_resistance_slider.SetValue(1.5)
            self.simple_shape_maintenance_slider.SetValue(3.0)

            self.advance_vertical_joint_coefficient_spin.SetValue(1.0)
            self.advance_horizonal_joint_coefficient_spin.SetValue(1.0)
            self.advance_diagonal_joint_coefficient_spin.SetValue(1.0)

            self.advance_horizonal_joint_valid_check.SetValue(1)
            self.advance_diagonal_joint_valid_check.SetValue(1)

        elif self.simple_primitive_ctrl.GetStringSelection() == logger.transtext("胸(大)"):
            self.simple_mass_slider.SetValue(6.2)
            self.simple_air_resistance_slider.SetValue(1.2)
            self.simple_shape_maintenance_slider.SetValue(2.6)

            self.advance_vertical_joint_coefficient_spin.SetValue(1.0)
            self.advance_horizonal_joint_coefficient_spin.SetValue(1.0)
            self.advance_diagonal_joint_coefficient_spin.SetValue(1.0)

            self.advance_horizonal_joint_valid_check.SetValue(1)
            self.advance_diagonal_joint_valid_check.SetValue(1)

        self.set_mass(event)
        self.on_vertical_joint(event)
        self.on_horizonal_joint(event)
        self.on_diagonal_joint(event)
        self.on_vertical_reverse_joint(event)
        self.on_horizonal_reverse_joint(event)

    def on_change_joint_pos_type(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        # ジョイント接続方法が変わったらジョイント値も変更する
        self.set_mass(event)

    def set_air_resistance(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        # 質量に応じて減衰を設定
        self.rigidbody_linear_damping_spin.SetValue(
            max(
                0,
                min(
                    0.9999,
                    1
                    - (
                        (
                            (
                                1
                                - self.simple_air_resistance_slider.GetValue()
                                / self.simple_air_resistance_slider.GetMax()
                            )
                            * (self.simple_mass_slider.GetValue() / self.simple_mass_slider.GetMax())
                        )
                        * 1.2
                    ),
                ),
            )
        )
        self.rigidbody_angular_damping_spin.SetValue(
            max(
                0,
                min(
                    0.9999,
                    1
                    - (
                        (
                            (
                                1
                                - self.simple_air_resistance_slider.GetValue()
                                / self.simple_air_resistance_slider.GetMax()
                            )
                            * (self.simple_mass_slider.GetValue() / self.simple_mass_slider.GetMax())
                        )
                        * 1.2
                    ),
                ),
            )
        )

        if self.physics_type_ctrl.GetStringSelection() == logger.transtext("髪"):
            # 髪の毛の場合、減衰をちょっと大きく（元に戻りやすく）
            self.rigidbody_linear_damping_spin.SetValue(
                max(0, min(0.9999, self.rigidbody_linear_damping_spin.GetValue() * 1.2))
            )
            self.rigidbody_angular_damping_spin.SetValue(
                max(0, min(0.9999, self.rigidbody_angular_damping_spin.GetValue() * 1.2))
            )

        elif self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸"):
            # 胸の場合、減衰を大きく（元に戻りやすく）
            self.rigidbody_linear_damping_spin.SetValue(
                max(0, min(0.9999, self.rigidbody_linear_damping_spin.GetValue() * 3))
            )
            self.rigidbody_angular_damping_spin.SetValue(
                max(0, min(0.9999, self.rigidbody_angular_damping_spin.GetValue() * 3))
            )

        # 摩擦力を設定
        self.rigidbody_friction_spin.SetValue(
            max(
                0,
                min(
                    0.9999,
                    (self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax() * 0.7),
                ),
            )
        )
        self.set_shape_maintenance(event)

    def set_shape_maintenance(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        mass_coefficient = self.simple_mass_slider.GetValue()

        # 柔らかさ
        air_resistance_ratio = (
            self.simple_air_resistance_slider.GetValue() / self.simple_air_resistance_slider.GetMax()
        )
        # 張り
        shape_maintenance_ratio = (
            self.simple_shape_maintenance_slider.GetValue() / self.simple_shape_maintenance_slider.GetMax()
        )

        base_joint_y_rot = (
            max(
                0,
                min(
                    180,
                    (180 - air_resistance_ratio * 180),
                ),
            )
            if air_resistance_ratio < 1
            else 0
        )
        base_joint_rot = (
            max(
                0,
                min(
                    89,
                    (89 - air_resistance_ratio * 89),
                ),
            )
            if air_resistance_ratio < 1
            else 0
        )

        # 縦ジョイント
        self.advance_vertical_joint_coefficient_spin.SetValue(2)

        vertical_joint_ratio = 0.2 if logger.transtext("胸") in self.physics_type_ctrl.GetStringSelection() else 0.9

        vertical_joint_rot = base_joint_rot * vertical_joint_ratio
        vertical_joint_y_rot = base_joint_y_rot * vertical_joint_ratio

        # 制限角度が0の場合、ちょっとだけ動かす
        vertical_joint_rot = 2 if not vertical_joint_rot else vertical_joint_rot
        vertical_joint_y_rot = 2 if not vertical_joint_y_rot else vertical_joint_y_rot

        self.vertical_joint_mov_x_min_spin.SetValue(0)
        self.vertical_joint_mov_x_max_spin.SetValue(0)
        self.vertical_joint_mov_y_min_spin.SetValue(0)
        self.vertical_joint_mov_y_max_spin.SetValue(0)
        self.vertical_joint_mov_z_min_spin.SetValue(0)
        self.vertical_joint_mov_z_max_spin.SetValue(0)

        vertical_spring_rot_ratio = 60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 40
        vertical_spring_y_rot_ratio = (
            60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 30
        )

        vertical_spring_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(vertical_spring_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )
        vertical_spring_y_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(vertical_spring_y_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )

        if self.physics_type_ctrl.GetStringSelection() == logger.transtext("髪"):
            # 髪の毛の場合、ジョイントの制限はきつめに・ばね値を小さめにしておく
            vertical_joint_rot /= 1.2
            vertical_spring_rot /= 2.5
            vertical_spring_y_rot /= 2.5

        self.vertical_joint_rot_x_min_spin.SetValue(-vertical_joint_rot)
        self.vertical_joint_rot_x_max_spin.SetValue(vertical_joint_rot)
        self.vertical_joint_rot_y_min_spin.SetValue(-vertical_joint_y_rot)
        self.vertical_joint_rot_y_max_spin.SetValue(vertical_joint_y_rot)
        self.vertical_joint_rot_z_min_spin.SetValue(-vertical_joint_rot)
        self.vertical_joint_rot_z_max_spin.SetValue(vertical_joint_rot)

        self.vertical_joint_spring_rot_x_spin.SetValue(vertical_spring_rot)
        self.vertical_joint_spring_rot_y_spin.SetValue(vertical_spring_y_rot)
        self.vertical_joint_spring_rot_z_spin.SetValue(vertical_spring_rot)

        # 横ジョイント
        if logger.transtext("胸") in self.physics_type_ctrl.GetStringSelection():
            self.horizonal_joint_mov_y_min_spin.SetValue(0)
            self.horizonal_joint_mov_y_max_spin.SetValue(0)
        else:
            horizonal_joint_mov = max(0, (1 - air_resistance_ratio) * 0.5)
            self.horizonal_joint_mov_y_min_spin.SetValue(-horizonal_joint_mov)
            self.horizonal_joint_mov_y_max_spin.SetValue(horizonal_joint_mov / 2)
        self.horizonal_joint_mov_x_min_spin.SetValue(0)
        self.horizonal_joint_mov_x_max_spin.SetValue(0)
        self.horizonal_joint_mov_z_min_spin.SetValue(0)
        self.horizonal_joint_mov_z_max_spin.SetValue(0)

        self.advance_horizonal_joint_coefficient_spin.SetValue(3)
        horizonal_joint_ratio = 0.1 if logger.transtext("胸") in self.physics_type_ctrl.GetStringSelection() else 0.8

        horizonal_joint_rot = base_joint_rot * horizonal_joint_ratio
        horizonal_joint_y_rot = base_joint_y_rot * horizonal_joint_ratio

        # 制限角度が0の場合、ちょっとだけ動かす
        horizonal_joint_rot = 2 if not horizonal_joint_rot else horizonal_joint_rot
        horizonal_joint_y_rot = 2 if not horizonal_joint_y_rot else horizonal_joint_y_rot

        horizonal_spring_rot_ratio = 60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 30
        horizonal_spring_y_rot_ratio = (
            60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 20
        )

        horizonal_spring_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(horizonal_spring_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )
        horizonal_spring_y_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(horizonal_spring_y_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )

        self.horizonal_joint_rot_x_min_spin.SetValue(-horizonal_joint_rot)
        self.horizonal_joint_rot_x_max_spin.SetValue(horizonal_joint_rot)
        self.horizonal_joint_rot_y_min_spin.SetValue(-horizonal_joint_y_rot)
        self.horizonal_joint_rot_y_max_spin.SetValue(horizonal_joint_y_rot)
        self.horizonal_joint_rot_z_min_spin.SetValue(-horizonal_joint_rot)
        self.horizonal_joint_rot_z_max_spin.SetValue(horizonal_joint_rot)

        self.horizonal_joint_spring_rot_x_spin.SetValue(horizonal_spring_rot)
        self.horizonal_joint_spring_rot_y_spin.SetValue(horizonal_spring_y_rot)
        self.horizonal_joint_spring_rot_z_spin.SetValue(horizonal_spring_rot)

        # 斜めジョイント
        if logger.transtext("胸") in self.physics_type_ctrl.GetStringSelection():
            self.diagonal_joint_mov_y_min_spin.SetValue(0)
            self.diagonal_joint_mov_y_max_spin.SetValue(0)
        else:
            diagonal_joint_mov = max(0, (1 - air_resistance_ratio) * 0.7)
            self.diagonal_joint_mov_y_min_spin.SetValue(-diagonal_joint_mov)
            self.diagonal_joint_mov_y_max_spin.SetValue(diagonal_joint_mov / 2)
        self.diagonal_joint_mov_x_min_spin.SetValue(0)
        self.diagonal_joint_mov_x_max_spin.SetValue(0)
        self.diagonal_joint_mov_z_min_spin.SetValue(0)
        self.diagonal_joint_mov_z_max_spin.SetValue(0)

        self.advance_diagonal_joint_coefficient_spin.SetValue(1.5)

        diagonal_joint_ratio = 0.05 if logger.transtext("胸") in self.physics_type_ctrl.GetStringSelection() else 1

        diagonal_joint_rot = base_joint_rot * diagonal_joint_ratio
        diagonal_joint_y_rot = base_joint_y_rot * diagonal_joint_ratio

        # 制限角度が0の場合、ちょっとだけ動かす
        diagonal_joint_rot = 2 if not diagonal_joint_rot else diagonal_joint_rot
        diagonal_joint_y_rot = 2 if not diagonal_joint_y_rot else diagonal_joint_y_rot

        diagonal_spring_rot_ratio = 60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 20
        diagonal_spring_y_rot_ratio = (
            60 if self.physics_type_ctrl.GetStringSelection() == logger.transtext("胸") else 10
        )

        diagonal_spring_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(diagonal_spring_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )
        diagonal_spring_y_rot = (
            shape_maintenance_ratio * self.get_joint_coefficient(diagonal_spring_y_rot_ratio) * mass_coefficient
            if shape_maintenance_ratio < 1
            else 0
        )

        self.diagonal_joint_rot_x_min_spin.SetValue(-diagonal_joint_rot)
        self.diagonal_joint_rot_x_max_spin.SetValue(diagonal_joint_rot)
        self.diagonal_joint_rot_y_min_spin.SetValue(-diagonal_joint_y_rot)
        self.diagonal_joint_rot_y_max_spin.SetValue(diagonal_joint_y_rot)
        self.diagonal_joint_rot_z_min_spin.SetValue(-diagonal_joint_rot)
        self.diagonal_joint_rot_z_max_spin.SetValue(diagonal_joint_rot)

        self.diagonal_joint_spring_rot_x_spin.SetValue(diagonal_spring_rot)
        self.diagonal_joint_spring_rot_y_spin.SetValue(diagonal_spring_y_rot)
        self.diagonal_joint_spring_rot_z_spin.SetValue(diagonal_spring_rot)

        # 縦逆ジョイント
        self.advance_vertical_reverse_joint_coefficient_spin.SetValue(
            self.advance_vertical_joint_coefficient_spin.GetValue()
        )

        self.vertical_reverse_joint_mov_x_min_spin.SetValue(self.vertical_joint_mov_x_min_spin.GetValue())
        self.vertical_reverse_joint_mov_x_max_spin.SetValue(self.vertical_joint_mov_x_max_spin.GetValue())
        self.vertical_reverse_joint_mov_y_min_spin.SetValue(self.vertical_joint_mov_y_min_spin.GetValue())
        self.vertical_reverse_joint_mov_y_max_spin.SetValue(self.vertical_joint_mov_y_max_spin.GetValue())
        self.vertical_reverse_joint_mov_z_min_spin.SetValue(self.vertical_joint_mov_z_min_spin.GetValue())
        self.vertical_reverse_joint_mov_z_max_spin.SetValue(self.vertical_joint_mov_z_max_spin.GetValue())

        self.vertical_reverse_joint_rot_x_min_spin.SetValue(self.vertical_joint_rot_x_min_spin.GetValue())
        self.vertical_reverse_joint_rot_x_max_spin.SetValue(self.vertical_joint_rot_x_max_spin.GetValue())
        self.vertical_reverse_joint_rot_y_min_spin.SetValue(self.vertical_joint_rot_y_min_spin.GetValue())
        self.vertical_reverse_joint_rot_y_max_spin.SetValue(self.vertical_joint_rot_y_max_spin.GetValue())
        self.vertical_reverse_joint_rot_z_min_spin.SetValue(self.vertical_joint_rot_z_min_spin.GetValue())
        self.vertical_reverse_joint_rot_z_max_spin.SetValue(self.vertical_joint_rot_z_max_spin.GetValue())

        self.vertical_reverse_joint_spring_rot_x_spin.SetValue(self.vertical_joint_spring_rot_x_spin.GetValue())
        self.vertical_reverse_joint_spring_rot_y_spin.SetValue(self.vertical_joint_spring_rot_y_spin.GetValue())
        self.vertical_reverse_joint_spring_rot_z_spin.SetValue(self.vertical_joint_spring_rot_z_spin.GetValue())

        # 横逆ジョイント
        self.horizonal_reverse_joint_mov_x_min_spin.SetValue(self.horizonal_joint_mov_x_min_spin.GetValue())
        self.horizonal_reverse_joint_mov_x_max_spin.SetValue(self.horizonal_joint_mov_x_max_spin.GetValue())

        self.advance_horizonal_reverse_joint_coefficient_spin.SetValue(
            self.advance_horizonal_joint_coefficient_spin.GetValue()
        )

        self.horizonal_reverse_joint_mov_x_min_spin.SetValue(self.horizonal_joint_mov_x_min_spin.GetValue())
        self.horizonal_reverse_joint_mov_x_max_spin.SetValue(self.horizonal_joint_mov_x_max_spin.GetValue())
        self.horizonal_reverse_joint_mov_y_min_spin.SetValue(self.horizonal_joint_mov_y_min_spin.GetValue())
        self.horizonal_reverse_joint_mov_y_max_spin.SetValue(self.horizonal_joint_mov_y_max_spin.GetValue())
        self.horizonal_reverse_joint_mov_z_min_spin.SetValue(self.horizonal_joint_mov_z_min_spin.GetValue())
        self.horizonal_reverse_joint_mov_z_max_spin.SetValue(self.horizonal_joint_mov_z_max_spin.GetValue())

        self.horizonal_reverse_joint_rot_x_min_spin.SetValue(self.horizonal_joint_rot_x_min_spin.GetValue())
        self.horizonal_reverse_joint_rot_x_max_spin.SetValue(self.horizonal_joint_rot_x_max_spin.GetValue())
        self.horizonal_reverse_joint_rot_y_min_spin.SetValue(self.horizonal_joint_rot_y_min_spin.GetValue())
        self.horizonal_reverse_joint_rot_y_max_spin.SetValue(self.horizonal_joint_rot_y_max_spin.GetValue())
        self.horizonal_reverse_joint_rot_z_min_spin.SetValue(self.horizonal_joint_rot_z_min_spin.GetValue())
        self.horizonal_reverse_joint_rot_z_max_spin.SetValue(self.horizonal_joint_rot_z_max_spin.GetValue())

        self.horizonal_reverse_joint_spring_rot_x_spin.SetValue(self.horizonal_joint_spring_rot_x_spin.GetValue())
        self.horizonal_reverse_joint_spring_rot_y_spin.SetValue(self.horizonal_joint_spring_rot_y_spin.GetValue())
        self.horizonal_reverse_joint_spring_rot_z_spin.SetValue(self.horizonal_joint_spring_rot_z_spin.GetValue())

        if self.simple_shape_maintenance_slider.GetValue() > self.simple_shape_maintenance_slider.GetMax() * 0.7:
            # 一定以上の維持感であれば斜めも張る
            if logger.transtext("布") in self.simple_material_ctrl.GetStringSelection():
                # 斜めは布のみ
                self.advance_diagonal_joint_valid_check.SetValue(1)
        # 一度張ったらチェックは自動で外さない

        if (
            self.physics_type_ctrl.GetStringSelection() == logger.transtext("布")
            and self.simple_shape_maintenance_slider.GetValue() > self.simple_shape_maintenance_slider.GetMax() * 0.9
        ):
            # 一定以上の維持感であれば逆も張る
            self.advance_vertical_reverse_joint_valid_check.SetValue(1)
            self.advance_horizonal_reverse_joint_valid_check.SetValue(1)

        self.on_diagonal_joint(event)
        self.on_vertical_reverse_joint(event)

    def get_joint_coefficient(self, val: float, is_reverse=False):
        if self.joint_pos_type_ctrl.GetStringSelection() == logger.transtext("ボーン位置"):
            return val * (0.7 if is_reverse else 1.3)
        return val

    def on_clear(self, event: wx.Event):
        self.simple_material_ctrl.SetStringSelection("")
        self.simple_back_material_ctrl.SetStringSelection("")
        self.simple_edge_material_ctrl.SetStringSelection("")
        self.simple_primitive_ctrl.SetStringSelection("")
        self.simple_mass_slider.SetValue(1.5)
        self.simple_air_resistance_slider.SetValue(1.8)
        self.simple_shape_maintenance_slider.SetValue(1.5)
        self.simple_direction_ctrl.SetStringSelection(logger.transtext("下"))
        self.simple_exist_physics_clear_ctrl.SetStringSelection(logger.transtext("そのまま"))
        self.simple_special_shape_ctrl.SetStringSelection(logger.transtext("なし"))

        self.advance_rigidbody_shape_type_ctrl.SetStringSelection(logger.transtext("箱"))
        self.physics_type_ctrl.SetStringSelection(logger.transtext("布"))
        self.density_type_ctrl.SetStringSelection(logger.transtext("頂点"))
        self.parent_type_ctrl.SetStringSelection(logger.transtext("中心"))
        self.joint_pos_type_ctrl.SetStringSelection(logger.transtext("ボーン位置"))
        self.route_search_type_ctrl.SetStringSelection(logger.transtext("前頂点優先"))
        self.route_estimate_type_ctrl.SetStringSelection(logger.transtext("角度"))
        self.advance_comment_ctrl.SetValue(self.main_frame.file_hitories.get("advance_comment", 0))

        self.set_material_name(event)
        # self.set_fineness(event)
        self.set_mass(event)
        self.set_air_resistance(event)
        self.set_shape_maintenance(event)

        self.advance_vertical_joint_valid_check.SetValue(1)
        self.advance_horizonal_joint_valid_check.SetValue(1)
        self.advance_diagonal_joint_valid_check.SetValue(0)
        self.advance_vertical_reverse_joint_valid_check.SetValue(0)
        self.advance_horizonal_reverse_joint_valid_check.SetValue(0)

        self.advance_vertical_joint_coefficient_spin.SetValue(2.8)
        self.advance_horizonal_joint_coefficient_spin.SetValue(4.2)
        self.advance_diagonal_joint_coefficient_spin.SetValue(1)
        self.advance_vertical_reverse_joint_coefficient_spin.SetValue(1)
        self.advance_horizonal_reverse_joint_coefficient_spin.SetValue(1)

    def create_weighted_bone_names(self):
        self.weighted_bone_names = {}
        model = self.main_frame.file_panel_ctrl.org_model_file_ctrl.data
        material_name = self.simple_material_ctrl.GetStringSelection()

        if material_name not in model.material_vertices:
            return

        # ウェイトボーンリスト取得
        all_weighted_bone_names = {}
        for vertex_idx in model.material_vertices[material_name]:
            vertex = model.vertex_dict[vertex_idx]
            if type(vertex.deform) is Bdef1:
                if vertex.deform.index0 not in list(all_weighted_bone_names.values()):
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
            elif type(vertex.deform) is Bdef2:
                if vertex.deform.index0 not in list(all_weighted_bone_names.values()) and vertex.deform.weight0 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(all_weighted_bone_names.values()) and vertex.deform.weight0 < 1:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
            elif type(vertex.deform) is Bdef4:
                if vertex.deform.index0 not in list(all_weighted_bone_names.values()) and vertex.deform.weight0 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(all_weighted_bone_names.values()) and vertex.deform.weight1 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1
                if vertex.deform.index2 not in list(all_weighted_bone_names.values()) and vertex.deform.weight2 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index2]] = vertex.deform.index2
                if vertex.deform.index3 not in list(all_weighted_bone_names.values()) and vertex.deform.weight3 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index3]] = vertex.deform.index3
            elif type(vertex.deform) is Sdef:
                if vertex.deform.index0 not in list(all_weighted_bone_names.values()) and vertex.deform.weight0 > 0:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index0]] = vertex.deform.index0
                if vertex.deform.index1 not in list(all_weighted_bone_names.values()) and vertex.deform.weight0 < 1:
                    all_weighted_bone_names[model.bone_indexes[vertex.deform.index1]] = vertex.deform.index1

        # まず親子マップを作成する
        bone_links = {}
        for bone_name, bone in model.bones.items():
            bone_links[bone_name] = model.create_link_2_top_one(bone_name, is_defined=False)
            logger.debug("link[%s]: %s", bone_name, bone_links[bone_name].all().keys())

        target_bone_links = {}
        for bone_name, links in reversed(bone_links.items()):
            is_regist = True
            for bname, blinks in bone_links.items():
                if bname != bone_name and bone_name in blinks.all().keys():
                    # 他のボーンリストに含まている場合、登録対象外
                    is_regist = False
                    break
            if is_regist:
                target_bone_links[bone_name] = links

        target_bone_names = {}
        for tail_bone_name, links in target_bone_links.items():
            tail_bone_idx = model.bones[tail_bone_name].index
            target_bone_names[tail_bone_idx] = []
            is_regist = False
            for bone in links.all().values():
                if bone.name in list(all_weighted_bone_names.keys()):
                    is_regist = True
                    break
            if is_regist:
                for bone in links.all().values():
                    target_bone_names[tail_bone_idx].append(bone.name)

        registed_bone_names = []
        self.weighted_bone_names = {}
        for tail_bone_idx in sorted(list(target_bone_names.keys())):
            self.weighted_bone_names[tail_bone_idx] = []
            is_regist = False
            for bone_name in reversed(target_bone_names[tail_bone_idx]):
                if bone_name not in registed_bone_names:
                    is_regist = True
            if is_regist:
                for bone_name in target_bone_names[tail_bone_idx]:
                    if bone_name not in registed_bone_names:
                        self.weighted_bone_names[tail_bone_idx].append(bone_name)
                        registed_bone_names.append(bone_name)
            logger.debug("weighted_bone_names[%s]: %s", tail_bone_idx, self.weighted_bone_names[tail_bone_idx])

    def on_click_extend_edge(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.simple_extend_edge_choice_ctrl.ShowModal()

    def on_click_extend_back(self, event: wx.Event):
        self.main_frame.file_panel_ctrl.on_change_file(event)
        self.simple_extend_back_choice_ctrl.ShowModal()


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def truncate_double_byte_str(text, len):
    """全角・半角を区別して文字列を切り詰める"""
    count = 0
    sliced_text = ""
    for c in text:
        if unicodedata.east_asian_width(c) in "FWA":
            count += 2
        else:
            count += 1

        # lenと同じ長さになったときに抽出完了
        if count > len:
            break
        sliced_text += c
    return sliced_text
